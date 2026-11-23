"""1D consensus Kd predictor combining pyccscore models via Ridge regression."""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pyccscore as cc

DEFAULT_CALIB_PATH = Path(__file__).resolve().parent.parent / "configs" / "consensus_calibration.json"

MODELS = {
    "iCipa_nter": cc.icipa_nter_core,
    "iCipa_vert": cc.icipa_core_vert,
    "bCIPA": cc.score_bcipa,
    "qCIPA": cc.score_qcipa,
    "Potapov": cc.score_potapov,
}
CONSENSUS_MODELS = ["iCipa_nter", "iCipa_vert", "bCIPA", "Potapov"]

_REG_MAP = {"f": 0, "g": 1, "a": 2, "b": 3, "c": 4, "d": 5, "e": 6}
_REGS = ["f", "g", "a", "b", "c", "d", "e"]

_AN4_PAD = "-EIAALEQEIAALEKENAALEWEIAALEQ"
_BN4_PAD = "-KIAALKQKIAALKYKNAALKKKIAALKQ"
_AN3_PAD = "-EIAALEKENAALEWEIAALEQ"


def _call_scorer(fn, p1_pad: str, p2_pad: str, shifts, orient) -> Tuple[float, int]:
    try:
        res = fn(p1_pad, p2_pad, shifts, False, orient)
    except Exception:
        res = fn(p1_pad, p2_pad, [0], False, orient)
    if isinstance(res, (tuple, list)):
        return float(res[0]), int(res[1]) if len(res) > 1 else 0
    return float(res), 0


def detect_score_polarity() -> Dict[str, bool]:
    polarity = {}
    for name, fn in MODELS.items():
        s_strong, _ = _call_scorer(fn, _AN4_PAD, _BN4_PAD, [0], cc.Orientation.PARALLEL)
        s_weak, _ = _call_scorer(fn, _AN3_PAD, _AN3_PAD, [0], cc.Orientation.PARALLEL)
        polarity[name] = (s_strong < s_weak)
    return polarity


def pad_register(seq: str, start_reg: str = "g") -> str:
    """Pad sequence with leading '-' so pyccscore starts at register 'f'."""
    clean = "".join(ch for ch in str(seq).strip().upper() if ch.isalpha())
    return "-" * _REG_MAP[start_reg] + clean


def infer_register_and_pad(seq: str, default_reg: str = "g") -> str:
    """Heuristically infer starting heptad register and pad to 'f'."""
    clean = "".join(ch for ch in str(seq).strip().upper() if ch.isalpha())
    if not clean:
        return "-"
    best_score, best_r = -1e9, default_reg
    for r_idx, r in enumerate(_REGS):
        sc = 0.05 if r == default_reg else 0.0
        for i, aa in enumerate(clean):
            pos = _REGS[(r_idx + i) % 7]
            if pos == "d" and aa == "L":
                sc += 2.0
            elif pos == "a" and aa in ("I", "V", "N", "L"):
                sc += 1.0
            elif pos in ("e", "g") and aa in ("E", "K", "Q", "R"):
                sc += 0.3
            elif pos == "d" and aa in ("E", "K", "D", "R"):
                sc -= 1.0
        if sc > best_score:
            best_score, best_r = sc, r
    return "-" * _REG_MAP[best_r] + clean


def ensure_padded(seq: str, start_reg: Optional[str] = None) -> str:
    s = str(seq).strip().upper()
    if start_reg is not None:
        return pad_register(s.lstrip("-"), start_reg)
    if s.startswith("-"):
        return s
    return infer_register_and_pad(s)


def score_raw_models(
    p1_pad: str,
    p2_pad: str,
    lower_is_stronger: Dict[str, bool],
    scan_shifts: bool = True,
    models: Optional[list] = None,
) -> Dict[str, float]:
    """Compute sign-normalized stability scores (higher = stronger)."""
    shifts = list(range(-7, 8)) if scan_shifts else [0]
    out: Dict[str, float] = {}
    for name in (models or list(MODELS.keys())):
        fn = MODELS[name]
        lower = lower_is_stronger[name]
        sign = -1.0 if lower else 1.0
        if scan_shifts:
            s_par, sh_par = _call_scorer(fn, p1_pad, p2_pad, shifts, cc.Orientation.PARALLEL)
            s_ant, sh_ant = _call_scorer(fn, p1_pad, p2_pad, shifts, cc.Orientation.ANTIPARALLEL)
            if lower:
                best_raw, best_sh, best_or = (s_par, sh_par, "PAR") if s_par <= s_ant else (s_ant, sh_ant, "ANTI")
            else:
                best_raw, best_sh, best_or = (s_par, sh_par, "PAR") if s_par >= s_ant else (s_ant, sh_ant, "ANTI")
            out[name] = sign * best_raw
            if name == "iCipa_nter":
                out["iCipa_best_shift"] = best_sh
                out["iCipa_best_orient"] = best_or
        else:
            s0, _ = _call_scorer(fn, p1_pad, p2_pad, [0], cc.Orientation.PARALLEL)
            out[name] = sign * s0
    return out


class ConsensusModel:
    def __init__(self, calib_path: Path = DEFAULT_CALIB_PATH):
        with open(calib_path, "r") as f:
            calib = json.load(f)
        self.calib = calib
        self.models = calib["consensus_models"]
        self.lower_is_stronger = calib["lower_is_stronger"]
        self.mean = np.array(calib["scaler_mean"], dtype=float)
        self.scale = np.array(calib["scaler_scale"], dtype=float)
        self.coef = np.array(calib["ridge_coef_log10_kd"], dtype=float)
        self.intercept = float(calib["ridge_intercept_log10_kd"])
        self.loo_rmse = float(calib.get("loo_rmse", 0.62))
        self._cache: Dict[Tuple[str, str], dict] = {}

    def predict_pair(
        self,
        seq1: str,
        seq2: str,
        reg1: Optional[str] = None,
        reg2: Optional[str] = None,
    ) -> dict:
        p1 = ensure_padded(seq1, reg1)
        p2 = ensure_padded(seq2, reg2)
        key = (p1, p2) if p1 <= p2 else (p2, p1)
        if key in self._cache:
            return dict(self._cache[key])

        res = score_raw_models(p1, p2, self.lower_is_stronger, scan_shifts=True, models=self.models)
        feats = np.array([res[m] for m in self.models], dtype=float)
        x_std = (feats - self.mean) / self.scale
        log10_kd = float(np.dot(x_std, self.coef) + self.intercept)
        res["log10_kd"] = log10_kd
        res["pKd"] = -log10_kd
        res["Kd_nM"] = (10.0 ** log10_kd) * 1e9
        self._cache[key] = res
        return dict(res)

    def slippage_margin(
        self,
        seq1: str,
        seq2: str,
        reg1: Optional[str] = None,
        reg2: Optional[str] = None,
    ) -> Tuple[float, str]:
        """Gap in iCipa_nter between canonical (shift=0, PAR) and best off-target pose."""
        p1 = ensure_padded(seq1, reg1)
        p2 = ensure_padded(seq2, reg2)
        fn = MODELS["iCipa_nter"]
        sign = -1.0 if self.lower_is_stronger["iCipa_nter"] else 1.0
        s0 = sign * _call_scorer(fn, p1, p2, [0], cc.Orientation.PARALLEL)[0]
        nz_shifts = [s for s in range(-7, 8) if s != 0]
        s_par_nz = sign * _call_scorer(fn, p1, p2, nz_shifts, cc.Orientation.PARALLEL)[0]
        s_anti = sign * _call_scorer(fn, p1, p2, list(range(-7, 8)), cc.Orientation.ANTIPARALLEL)[0]
        best_wrong = max(s_par_nz, s_anti)
        return float(s0 - best_wrong), ("PAR_SLIP" if s_par_nz >= s_anti else "ANTI")
