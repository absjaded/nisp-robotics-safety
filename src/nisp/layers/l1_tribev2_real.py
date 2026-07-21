"""Layer 1: file-lookup TRIBE v2 latent loader for the --real-neural demo path.

The whitepaper architecture is:

    MuJoCo render -> TRIBE v2 (7B params, GPU-only) -> fsaverage5 (T, 20484)
                  -> DiFuMo projection (1024,) -> L2

Step 2 — TRIBE v2 forward — costs ~30s/clip on H100 and tens of seconds per
clip on consumer GPUs. We never run it live. Instead, scripts/precompute_latents.py
runs TRIBE v2 once on the 15 MuJoCo renders and caches the DiFuMo latents to
artifacts/latents/cycle_{i}_difumo.npy. This module is the runtime side: a
sim_time -> cycle_id mapping plus a `np.load`.

Falls back to TribevX (Gaussian Sampler) if the precomputed latent is missing,
so the demo still runs correctly when artifacts/latents is empty (e.g. before
the H100 session).
"""
from pathlib import Path
from typing import Optional

import numpy as np

from src.nisp.core.contracts import LatentVector, RiskState, TelemetryState


_DEFAULT_LATENT_DIR = Path("artifacts/latents")
_DEFAULT_DT_SECS    = 0.02   # demo cycles are spaced 20ms apart in sim time
_LATENT_DIM         = 1024


class L1_TribeV2_Real:
    """File-lookup L1 layer that reads precomputed DiFuMo latents.

    Parameters
    ----------
    latent_dir : Path | str
        Directory containing cycle_{i}_difumo.npy files.
    dt : float
        Simulated cycle period — used to map TelemetryState.sim_time to cycle_id.
    fallback : Optional[object]
        Synthetic L1 (e.g. TribevX) used when no .npy exists for the cycle.
        If None, raises FileNotFoundError on a miss.
    """

    P = _LATENT_DIM  # contract: outputs 1024-dim DiFuMo latents

    def __init__(self,
                 latent_dir: Path | str = _DEFAULT_LATENT_DIR,
                 dt: float = _DEFAULT_DT_SECS,
                 fallback: Optional[object] = None):
        self._latent_dir = Path(latent_dir)
        self._dt = dt
        self._fallback = fallback
        self._cache: dict[int, np.ndarray] = {}
        self._n_cycles = self._count_available()

    def _count_available(self) -> int:
        if not self._latent_dir.exists():
            return 0
        return sum(1 for p in self._latent_dir.glob("cycle_*_difumo.npy"))

    def _cycle_id(self, sim_time: float) -> int:
        """Map sim_time -> cycle_id, wrapping if more cycles than latents."""
        n = max(self._n_cycles, 1)
        return int(round(sim_time / self._dt)) % n

    def _load(self, cycle_id: int) -> Optional[np.ndarray]:
        if cycle_id in self._cache:
            return self._cache[cycle_id]
        path = self._latent_dir / f"cycle_{cycle_id}_difumo.npy"
        if not path.exists():
            return None
        E = np.load(path).astype(np.float32)
        if E.shape != (_LATENT_DIM,):
            raise ValueError(
                f"{path} has shape {E.shape}, expected ({_LATENT_DIM},). "
                "Re-run scripts/precompute_latents.py."
            )
        self._cache[cycle_id] = E
        return E

    def _infer_risk_state(self, telemetry: TelemetryState) -> RiskState:
        """Heuristic risk_state for the LatentVector metadata field.

        TRIBE v2 doesn't natively output risk_state — we tag it from telemetry
        so the Narrator can colour-code its banner. This metadata never feeds
        L2 inference.
        """
        if telemetry.min_dist < 0.5 or telemetry.tcp_velocity > 0.25:
            return RiskState.CRITICAL
        if telemetry.min_dist < 0.7 or telemetry.tcp_velocity > 0.12:
            return RiskState.HESITANT
        return RiskState.NOMINAL

    def process(self, telemetry: TelemetryState) -> LatentVector:
        cycle_id = self._cycle_id(telemetry.sim_time)
        E = self._load(cycle_id)

        if E is None:
            if self._fallback is not None:
                return self._fallback.process(telemetry)
            raise FileNotFoundError(
                f"No precomputed latent for cycle {cycle_id} in {self._latent_dir} "
                "and no fallback L1 provided. Run scripts/precompute_latents.py "
                "or pass fallback=TribevX(...)."
            )

        return LatentVector(
            E=E,
            risk_state=self._infer_risk_state(telemetry),
            timestamp=telemetry.sim_time,
        )
