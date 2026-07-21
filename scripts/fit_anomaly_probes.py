"""Binary anomaly probes on UCF-Crime DiFuMo latents.

Reads the 53 latents from extract_ucf_latents.py and benchmarks several
classifiers via stratified 5-fold CV. Picks the best one by mean AUC,
re-fits on a single 75/25 split, saves weights + per-clip predictions.

Why CV: with N=53 and dim=1024, single-split test scores swing ~10pp
from one seed to the next. K-fold gives a stable mean ± std that
reviewers and demo voiceovers can quote.

Usage
-----
    python scripts/fit_anomaly_probes.py
        Defaults to artifacts/latents/ + artifacts/labels/anomaly_labels.json.

Outputs
-------
    artifacts/probe_weights_anomaly.npz   weights of best model + scaler
    artifacts/results/probe_anomaly.json  CV scores, best model, per-clip pred
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--latents-dir", default="artifacts/latents")
    p.add_argument("--labels", default="artifacts/labels/anomaly_labels.json")
    p.add_argument("--output-weights", default="artifacts/probe_weights_anomaly.npz")
    p.add_argument("--output-metrics", default="artifacts/results/probe_anomaly.json")
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--C", type=float, default=1.0,
                   help="Logistic regression inverse-regularization")
    return p.parse_args()


def _load_data(latents_dir: Path, labels_path: Path):
    labels = json.loads(labels_path.read_text())["labels"]
    rows: list[tuple[str, np.ndarray, int]] = []
    for vid, lab in labels.items():
        npy = latents_dir / f"{vid}_difumo.npy"
        if not npy.exists():
            print(f"  [skip] no latent for {vid}")
            continue
        rows.append((vid, np.load(npy).astype(np.float32), int(lab)))
    if not rows:
        raise RuntimeError(f"No latents found in {latents_dir}")
    vids = [r[0] for r in rows]
    X = np.stack([r[1] for r in rows], axis=0)   # (N, 1024)
    y = np.array([r[2] for r in rows], dtype=np.int64)
    return vids, X, y


def main():
    args = _parse_args()
    latents_dir = Path(args.latents_dir)
    labels_path = Path(args.labels)

    vids, X, y = _load_data(latents_dir, labels_path)
    print(f"  [data] N={len(y)}  dim={X.shape[1]}  "
          f"anomaly={int(y.sum())}  normal={int((y==0).sum())}")

    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import (accuracy_score, roc_auc_score, recall_score,
                                  precision_score, confusion_matrix,
                                  classification_report)

    candidates = {
        "logreg_l2":    Pipeline([("sc", StandardScaler()),
                                   ("clf", LogisticRegression(
                                       C=args.C, max_iter=2000,
                                       class_weight="balanced",
                                       random_state=args.seed))]),
        "logreg_l1":    Pipeline([("sc", StandardScaler()),
                                   ("clf", LogisticRegression(
                                       C=args.C, penalty="l1", solver="liblinear",
                                       max_iter=2000, class_weight="balanced",
                                       random_state=args.seed))]),
        "svm_rbf":      Pipeline([("sc", StandardScaler()),
                                   ("clf", SVC(C=1.0, gamma="scale",
                                               probability=True,
                                               class_weight="balanced",
                                               random_state=args.seed))]),
        "mlp":          Pipeline([("sc", StandardScaler()),
                                   ("clf", MLPClassifier(
                                       hidden_layer_sizes=(64,), alpha=1e-3,
                                       max_iter=2000,
                                       random_state=args.seed))]),
        "gboost":       Pipeline([("sc", StandardScaler()),
                                   ("clf", GradientBoostingClassifier(
                                       n_estimators=100, max_depth=3,
                                       random_state=args.seed))]),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    cv_results: dict[str, dict] = {}

    print("\n=== STRATIFIED 5-FOLD CV ===")
    for name, pipe in candidates.items():
        accs: list[float] = []
        aucs: list[float] = []
        recalls: list[float] = []
        precisions: list[float] = []
        for fold, (tr, te) in enumerate(skf.split(X, y)):
            pipe.fit(X[tr], y[tr])
            pred = pipe.predict(X[te])
            try:
                proba = pipe.predict_proba(X[te])[:, 1]
            except Exception:
                proba = pipe.decision_function(X[te])
            accs.append(accuracy_score(y[te], pred))
            recalls.append(recall_score(y[te], pred, pos_label=1, zero_division=0))
            precisions.append(precision_score(y[te], pred, pos_label=1,
                                               zero_division=0))
            try:
                aucs.append(roc_auc_score(y[te], proba))
            except ValueError:
                aucs.append(float("nan"))
        cv_results[name] = {
            "acc_mean":       float(np.mean(accs)),
            "acc_std":        float(np.std(accs)),
            "auc_mean":       float(np.nanmean(aucs)),
            "auc_std":        float(np.nanstd(aucs)),
            "recall_mean":    float(np.mean(recalls)),
            "recall_std":     float(np.std(recalls)),
            "precision_mean": float(np.mean(precisions)),
            "precision_std":  float(np.std(precisions)),
            "fold_acc":       [float(a) for a in accs],
            "fold_auc":       [float(a) for a in aucs],
            "fold_recall":    [float(a) for a in recalls],
            "fold_precision": [float(a) for a in precisions],
        }
        r = cv_results[name]
        print(f"  {name:14s} "
              f"acc={r['acc_mean']:.3f}±{r['acc_std']:.3f}  "
              f"auc={r['auc_mean']:.3f}±{r['auc_std']:.3f}  "
              f"recall={r['recall_mean']:.3f}±{r['recall_std']:.3f}  "
              f"prec={r['precision_mean']:.3f}±{r['precision_std']:.3f}")

    best_name = max(cv_results, key=lambda k: cv_results[k]["auc_mean"])
    print(f"\n  [winner] {best_name}  "
          f"auc_mean={cv_results[best_name]['auc_mean']:.3f}")

    # Re-fit best model on a 75/25 hold-out for a single inspectable split
    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=args.test_size, stratify=y,
        random_state=args.seed)
    best = candidates[best_name]
    best.fit(X[train_idx], y[train_idx])
    test_pred  = best.predict(X[test_idx])
    try:
        test_proba = best.predict_proba(X[test_idx])[:, 1]
    except Exception:
        test_proba = best.decision_function(X[test_idx])
    test_acc = accuracy_score(y[test_idx], test_pred)
    try:
        test_auc = roc_auc_score(y[test_idx], test_proba)
    except ValueError:
        test_auc = float("nan")
    cm = confusion_matrix(y[test_idx], test_pred).tolist()

    print(f"\n=== {best_name} 75/25 HOLD-OUT ===")
    print(f"  test acc: {test_acc:.3f}  test AUC: {test_auc:.3f}")
    print(f"  confusion: {cm}")
    print(classification_report(y[test_idx], test_pred,
                                 target_names=["Normal", "Anomaly"]))

    # Save weights only when it's a linear pipeline (introspectable)
    Path(args.output_weights).parent.mkdir(parents=True, exist_ok=True)
    sc = best.named_steps["sc"]
    clf = best.named_steps["clf"]
    save_kwargs = {
        "scaler_mean":  sc.mean_.astype(np.float32),
        "scaler_scale": sc.scale_.astype(np.float32),
        "best_model":   np.asarray(best_name),
    }
    if hasattr(clf, "coef_"):
        save_kwargs["coef"] = clf.coef_.astype(np.float32)
        save_kwargs["intercept"] = clf.intercept_.astype(np.float32)
    np.savez(args.output_weights, **save_kwargs)
    print(f"  saved {args.output_weights}")

    Path(args.output_metrics).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_metrics).write_text(json.dumps({
        "n_total": int(len(y)),
        "n_anomaly": int(y.sum()),
        "n_normal": int((y == 0).sum()),
        "cv": cv_results,
        "best_model": best_name,
        "holdout_test_accuracy": float(test_acc),
        "holdout_test_auc": float(test_auc),
        "holdout_confusion": cm,
        "holdout_per_clip": [
            {"video_id": vids[i], "true": int(y[i]),
             "pred": int(test_pred[k]), "p_anomaly": float(test_proba[k])}
            for k, i in enumerate(test_idx)
        ],
        "config": {"C": args.C, "test_size": args.test_size, "seed": args.seed},
    }, indent=2))
    print(f"  saved {args.output_metrics}")


if __name__ == "__main__":
    main()
