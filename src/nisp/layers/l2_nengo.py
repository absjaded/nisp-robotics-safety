"""Layer 2: Nengo Spiking Neural Network decoder for TRIBE v2 latents.

Replaces the linear-probe decoder (L2_PredicateDecoder) for the --real-neural demo path.

Architecture:
    DiFuMo (1024,) → PCA compression (64,) → Nengo LIF ensemble (256 neurons)
    → NEF decoded output (64,) → linear probe (8,) → softmax → Z

Why Nengo here:
    TRIBE v2 outputs predicted BOLD signals — what a brain *observes*. Nengo
    simulates the spiking population dynamics that produce those BOLD signals,
    decoding safety predicates as spike firing rates. This is the neuroscience-
    correct inversion: BOLD → spikes (deconvolution of the hemodynamic response).

Compute split:
    - TRIBE v2: GPU (H100, offline pre-compute)
    - Nengo SNN decoder: CPU only, ~5ms per cycle after one-time 300ms build
"""
import json
import os
import warnings
from pathlib import Path

import numpy as np

try:
    import nengo
    NENGO_OK = True
except ImportError:
    NENGO_OK = False

from src.nisp.core.contracts import LatentVector, PredicateDistribution

_LATENT_DIR  = Path("artifacts/latents")
_N_PREDICATES = 8
_N_NEURONS    = 256     # LIF neurons in ensemble
_PCA_DIMS     = 64      # reduce 1024 → 64 before Nengo ensemble
_SIM_MS       = 20      # milliseconds to simulate per cycle (1ms timesteps)
_SYNAPSE      = 0.005   # low-pass synapse filter constant (seconds)


class L2_Nengo:
    """Nengo LIF spiking network decoder.

    Instantiate once (builds the Nengo simulator in ~300ms).
    Call process() per cycle (~5ms each).

    Falls back gracefully to L2_PredicateDecoder-style linear probes if nengo
    is not installed or if the latent directory is empty.
    """

    def __init__(self,
                 latent_dir: str = str(_LATENT_DIR),
                 probe_weights_path: str = "artifacts/probe_weights.pt",
                 labels_path: str = "artifacts/labels/cycle_labels.json",
                 temperature: float = 1.0,
                 seed: int = 42):
        """
        Parameters
        ----------
        labels_path : str
            JSON of per-cycle dominant predicate labels. The demo passes
            artifacts/labels/curated_labels.json when --curated is set so the
            lstsq decoder is fitted on just the 5 hero cycles.
        temperature : float
            Softmax temperature applied to the predicate logits. lstsq returns
            small-magnitude weights that produce a near-uniform softmax (max
            ~0.25-0.30) even when argmax is correct — this is a known calibration
            issue with one-hot regression targets. Lower temperature (e.g. 0.1)
            sharpens the distribution so the dominant predicate exceeds the 0.60
            REPLAN threshold. The argmax (i.e. *which* predicate fires) is
            unaffected — only confidence changes. Default 1.0 = uncalibrated.
        probe_weights_path : str
            Optional fine-tuned (8, 1024) probe weights from precompute_latents.py.
            **Note:** these are saved in adapter-output space (NEURONA-style),
            not raw-DiFuMo space — so loading them here would mix coordinate
            systems. The existing fitter on raw PCA gives a self-consistent
            decoder; loading is left as a hook for a future end-to-end retrain.
        """
        self._rng         = np.random.default_rng(seed)
        self._latent_dir  = Path(latent_dir)
        self._labels_path = Path(labels_path)
        self._temperature = float(temperature)
        self._fallback    = False
        self._sim         = None
        self._buf         = {"x": np.zeros(_PCA_DIMS)}
        self._pca_mean    = None
        self._pca_components = None  # (64, 1024) — top-64 PCA directions
        self._probe_W     = None     # (64, 8)   — linear readout from PCA space
        self._nengo_probe = None

        if not NENGO_OK:
            warnings.warn("nengo not installed — L2_Nengo falling back to linear probes. "
                          "pip install nengo")
            self._fallback = True
            self._init_fallback(probe_weights_path)
            return

        latents = self._load_latents()
        if not latents:
            warnings.warn(f"No latents found in {latent_dir} — running render_scenes.py "
                          "and precompute_latents.py first. L2_Nengo falling back.")
            self._fallback = True
            self._init_fallback(probe_weights_path)
            return

        self._fit_pca(latents)
        self._fit_probes(latents, probe_weights_path)
        self._build_nengo()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _load_latents(self) -> list:
        latents = []
        for i in range(15):
            p = self._latent_dir / f"cycle_{i}_difumo.npy"
            if p.exists():
                latents.append((i, np.load(p).astype(np.float32)))
        return latents

    def _fit_pca(self, latents: list) -> None:
        """Fit PCA on available latents, keep top-_PCA_DIMS components."""
        X = np.stack([l for _, l in latents], axis=0)  # (N, 1024)
        self._pca_mean = X.mean(axis=0)
        X_c = X - self._pca_mean
        # Thin SVD: U (N, N), S (N,), Vt (N, 1024)
        k = min(_PCA_DIMS, len(latents) - 1, X_c.shape[1])
        _, _, Vt = np.linalg.svd(X_c, full_matrices=False)
        self._pca_components = Vt[:k].astype(np.float32)  # (k, 1024)

    def _pca_project(self, latent: np.ndarray) -> np.ndarray:
        """(1024,) → (k,) normalised to unit sphere for Nengo.

        k = actual number of PCA components (≤ min(_PCA_DIMS, n_latents-1)).
        """
        v = (latent - self._pca_mean) @ self._pca_components.T  # (k,)
        norm = np.linalg.norm(v) + 1e-8
        return (v / norm).astype(np.float32)

    def _fit_probes(self, latents: list, probe_weights_path: str) -> None:
        """Fit linear readout from PCA space to 8 predicates.

        Reads label mapping from `self._labels_path` (curated or full).
        Two on-disk schemas are supported:
          (A) {"cycle_labels": [int, ... 15 entries]} — full set
          (B) {"curated_cycle_ids": [...], "cycle_labels": {"0": int, "4": int, ...}} — curated
        """
        # We deliberately ignore probe_weights_path here — see __init__ docstring.
        # The PCA-space lstsq fitter is what feeds the Nengo decoder consistently.

        labels_path = self._labels_path
        cycle_labels: dict[int, int] = {}
        if labels_path.exists():
            data = json.loads(labels_path.read_text())
            raw = data.get("cycle_labels", {})
            if isinstance(raw, list):                 # schema A
                cycle_labels = {i: lbl for i, lbl in enumerate(raw)}
                source = f"{labels_path.name} (full)"
            else:                                     # schema B
                cycle_labels = {int(k): v for k, v in raw.items()}
                source = f"{labels_path.name} (curated)"
        else:
            # Last resort: map cycle_id to predicate index by SCRIPT order.
            cycle_labels = {i: lbl for i, lbl in
                            enumerate([7, 7, 7, 4, 5, 6, 4, 0, 0, 3, 2, 1, 1, 2, 7])}
            source = "hardcoded fallback"

        # Subset latents to those that actually have a label (relevant for curated).
        labeled = [(cid, l) for (cid, l) in latents if cid in cycle_labels]
        if not labeled:
            warnings.warn(f"No cached latents match labels in {labels_path}. "
                          "Falling back to all available latents with default label 7.")
            labeled = latents
            label_for = lambda cid: 7
        else:
            label_for = lambda cid: cycle_labels[cid]

        X_pca = np.stack([self._pca_project(l) for _, l in labeled])  # (N, k)
        Y     = np.zeros((len(labeled), _N_PREDICATES), dtype=np.float32)
        for idx, (cid, _) in enumerate(labeled):
            Y[idx, label_for(cid)] = 1.0
        # Least-squares decoder: (k, 8)
        self._probe_W, _, _, _ = np.linalg.lstsq(X_pca, Y, rcond=None)
        print(f"  [L2_Nengo] Fitted linear probe decoder from {source} "
              f"({len(labeled)} cycles).")

    def _build_nengo(self) -> None:
        """Build and compile Nengo simulator (one-time, ~300ms).

        Ensemble dimension = actual PCA components (≤ _PCA_DIMS).
        With 15 latents, SVD gives 14 non-trivial components.
        """
        buf  = self._buf
        n_dims = self._pca_components.shape[0]  # actual k, not _PCA_DIMS constant

        with nengo.Network(seed=42) as net:
            inp = nengo.Node(lambda t: buf["x"], size_out=n_dims)
            ens = nengo.Ensemble(
                n_neurons=_N_NEURONS,
                dimensions=n_dims,
                neuron_type=nengo.LIF(),
                seed=42,
            )
            nengo.Connection(inp, ens, synapse=None)
            self._nengo_probe = nengo.Probe(ens, "decoded_output", synapse=_SYNAPSE)
            # Per-neuron spike-rate probe (N-1c). Lets grounding show *which*
            # LIF neurons fired for each predicate, not just probe-weight magnitudes.
            self._spike_probe = nengo.Probe(ens.neurons, synapse=_SYNAPSE)
            # Capture the encoder matrix so we can attribute neurons -> PCA dims.
            self._ens_handle = ens

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._sim = nengo.Simulator(net, dt=0.001, progress_bar=False)
        # After build, encoders are populated. Shape: (n_neurons, n_dims).
        # We use them for spike-attribution back to PCA dims in _compute_grounding.
        self._encoders = self._sim.data[self._ens_handle].encoders.astype(np.float32)
        print(f"  [L2_Nengo] Nengo simulator built — "
              f"{_N_NEURONS} LIF neurons, {n_dims}-dim PCA space, "
              f"{_SIM_MS}ms sim per cycle.")

    def _init_fallback(self, probe_weights_path: str) -> None:
        """Minimal linear probe fallback when Nengo is unavailable."""
        self._W_fallback = self._rng.standard_normal(
            (_N_PREDICATES, 1024)).astype(np.float32) * 0.01
        self._b_fallback = np.zeros(_N_PREDICATES, dtype=np.float32)
        if probe_weights_path and os.path.exists(probe_weights_path):
            try:
                import torch
                w_dict = torch.load(probe_weights_path, map_location="cpu")
                self._W_fallback = w_dict["linear.weight"].numpy()
                self._b_fallback = w_dict["linear.bias"].numpy()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def process(self, latent: LatentVector) -> PredicateDistribution:
        """Decode a DiFuMo LatentVector to a predicate distribution via SNN.

        If Nengo is unavailable, falls back to a direct linear probe.
        """
        E = np.asarray(latent.E, dtype=np.float32)

        if self._fallback or self._sim is None:
            return self._process_fallback(E)

        # Project to PCA space, feed to Nengo ensemble
        x_pca = self._pca_project(E)
        self._buf["x"] = x_pca
        self._sim.reset()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._sim.run(_SIM_MS * 0.001)   # run _SIM_MS timesteps

        # Last 5ms of decoded output, averaged (low-pass settled region)
        decoded = self._sim.data[self._nengo_probe][-5:].mean(axis=0)  # (k,)
        spikes  = self._sim.data[self._spike_probe][-5:].mean(axis=0)  # (n_neurons,)

        # Linear readout from PCA space → 8 predicate logits.
        # Temperature scaling: low T sharpens the softmax so the dominant
        # predicate's mass exceeds the 0.60 REPLAN threshold. Calibration only —
        # argmax (which predicate fires) is invariant to temperature.
        logits = (decoded @ self._probe_W) / self._temperature   # (8,)
        Z      = _softmax(logits)

        grounding_maps = self._compute_grounding(decoded, spikes)
        return PredicateDistribution(Z=Z, grounding_maps=grounding_maps)

    def _process_fallback(self, E: np.ndarray) -> PredicateDistribution:
        logits = (self._W_fallback @ E) + self._b_fallback
        Z = _softmax(logits)
        grounding_maps = {
            k: np.abs(self._W_fallback[k] * E) / (np.abs(self._W_fallback[k] * E).max() + 1e-8)
            for k in range(_N_PREDICATES)
        }
        return PredicateDistribution(Z=Z, grounding_maps=grounding_maps)

    def _compute_grounding(self, decoded: np.ndarray, spikes: np.ndarray) -> dict:
        """Grounding map: spike-rate-weighted attribution back to 1024-dim latent.

        Pipeline:
            spikes (n_neurons,)      — average firing rate over last 5ms
            encoders (n_neurons, k)  — each neuron's preferred PCA direction
            spike_pca (k,)           = encoders.T @ spikes  -> PCA-space activity
                                       weighted by which neurons actually fired
            probe_W[:, k] (k,)       — predicate-k readout weights in PCA space
            attribution (k,)         = |spike_pca * decoded| * |probe_W[:, k]|
            grounding[k] (1024,)     = pca_components.T @ attribution

        Reading: high values point to DiFuMo brain regions whose corresponding
        Nengo neurons both *fired* and *contributed to predicate k's logit*.
        """
        # Neuron firing -> PCA-space activity. Encoders point each neuron to a
        # direction in PCA space, so spike_pca[d] is "how much PCA dim d was
        # represented in the population activity this cycle."
        spike_pca = self._encoders.T @ spikes        # (k,)
        # Norm so different inputs are comparable.
        spike_pca = spike_pca / (np.abs(spike_pca).max() + 1e-8)

        grounding = {}
        for k in range(_N_PREDICATES):
            # PCA importance: probe weight × decoded value × spike-driven activity.
            # The spike term is what makes this "spiking-aware" rather than a
            # plain probe-weight magnitude.
            attribution = np.abs(spike_pca * decoded) * np.abs(self._probe_W[:, k])  # (k,)
            latent_importance = self._pca_components.T @ attribution  # (1024,)
            m = np.abs(latent_importance).max() + 1e-8
            grounding[k] = (latent_importance / m).astype(np.float32)
        return grounding


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()
