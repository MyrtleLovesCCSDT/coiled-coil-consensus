"""Thermodynamic YES-gate simulation via EQTK and Monte Carlo robustness testing."""

from typing import Dict, Optional
import numpy as np
import eqtk

from .consensus import ConsensusModel

# Species: I, Q, S, IQ, QS, IS, II, QQ, SS
STOICH_N = np.array([
    [-1, -1,  0,  1,  0,  0,  0,  0,  0],
    [ 0, -1, -1,  0,  1,  0,  0,  0,  0],
    [-1,  0, -1,  0,  0,  1,  0,  0,  0],
    [-2,  0,  0,  0,  0,  0,  1,  0,  0],
    [ 0, -2,  0,  0,  0,  0,  0,  1,  0],
    [ 0,  0, -2,  0,  0,  0,  0,  0,  1],
], dtype=float)


def _newton_dimers_fallback(c0_rows: np.ndarray, k_vec: np.ndarray) -> np.ndarray:
    k_iq, k_qs, k_is, k_ii, k_qq, k_ss = k_vec
    out = np.zeros_like(c0_rows)
    for idx, row in enumerate(c0_rows):
        it, qt, st = row[0], row[1], row[2]
        x = np.array([max(it, 1e-12), max(qt, 1e-12), max(st, 1e-12)], dtype=float)
        for _ in range(80):
            i, q, s = x
            f_vec = np.array([
                i + k_iq * i * q + k_is * i * s + 2 * k_ii * i * i - it,
                q + k_iq * i * q + k_qs * q * s + 2 * k_qq * q * q - qt,
                s + k_qs * q * s + k_is * i * s + 2 * k_ss * s * s - st,
            ])
            if np.max(np.abs(f_vec)) < 1e-14:
                break
            jac = np.array([
                [1 + k_iq * q + k_is * s + 4 * k_ii * i, k_iq * i, k_is * i],
                [k_iq * q, 1 + k_iq * i + k_qs * s + 4 * k_qq * q, k_qs * q],
                [k_is * s, k_qs * s, 1 + k_qs * q + k_is * i + 4 * k_ss * s],
            ])
            dx = np.linalg.solve(jac, -f_vec)
            step = 1.0
            while np.any(x + step * dx <= 0) and step > 1e-6:
                step *= 0.5
            x = np.maximum(x + step * dx, 1e-18)
        i, q, s = x
        out[idx] = [i, q, s, k_iq * i * q, k_qs * q * s, k_is * i * s, k_ii * i * i, k_qq * q * q, k_ss * s * s]
    return out


def _check_eqtk() -> bool:
    try:
        c_test = np.zeros((1, 9))
        c_test[0, 0] = 1e-6
        c_test[0, 1] = 1e-6
        sol = np.asarray(eqtk.solve(c_test, N=STOICH_N, K=np.array([1e6, 1.0, 1.0, 1.0, 1.0, 1.0])))
        return abs(sol[0, 3] - 0.38196601e-6) < 1e-9
    except Exception:
        return False


_USE_EQTK = _check_eqtk()


def solve_equilibrium(c0_rows: np.ndarray, kd_vec: np.ndarray) -> np.ndarray:
    kd_safe = np.maximum(np.asarray(kd_vec, dtype=float), 1e-12)
    k_vec = 1.0 / kd_safe
    if _USE_EQTK:
        return np.asarray(eqtk.solve(c0_rows, N=STOICH_N, K=k_vec))
    return _newton_dimers_fallback(c0_rows, k_vec)


def simulate_yes_gate(
    kd_dict: Dict[str, float],
    s0_uM: float = 0.5,
    q0_uM: float = 1.0,
    i_on_uM: float = 2.5,
) -> dict:
    """Simulate OFF ([I]=0) and ON ([I]=i_on_uM) states for I + QS <-> IQ + S."""
    c0 = np.zeros((2, 9))
    c0[:, 1] = q0_uM * 1e-6
    c0[:, 2] = s0_uM * 1e-6
    c0[1, 0] = i_on_uM * 1e-6

    kd_vec = np.array([
        kd_dict["IQ"], kd_dict["QS"], kd_dict["IS"],
        kd_dict["II"], kd_dict["QQ"], kd_dict["SS"],
    ])
    eq = solve_equilibrium(c0, kd_vec)
    s0_m = s0_uM * 1e-6
    q0_m = q0_uM * 1e-6

    s_free_off = float(eq[0, 2] / s0_m)
    s_free_on = float(eq[1, 2] / s0_m)
    return {
        "S_free_OFF": s_free_off,
        "S_free_ON": s_free_on,
        "QS_OFF": float(eq[0, 4] / s0_m),
        "QS_ON": float(eq[1, 4] / s0_m),
        "Delta_S_free": s_free_on - s_free_off,
        "Fold_ON_OFF": s_free_on / max(s_free_off, 0.005),
        "IS_leak_ON": float(eq[1, 5] / s0_m),
        "QQ_loss_OFF": float(2.0 * eq[0, 7] / q0_m),
        "SS_loss_OFF": float(2.0 * eq[0, 8] / s0_m),
    }


def mc_robustness(
    log10_kds_base: np.ndarray,
    s0_uM: float = 0.5,
    q0_uM: float = 1.0,
    i_on_uM: float = 2.5,
    n_mc: int = 200,
    sigma: float = 0.62,
    seed: int = 42,
) -> dict:
    """Monte Carlo stress test perturbing all 6 log10(Kd) values by N(0, sigma)."""
    rng = np.random.default_rng(seed)
    c0 = np.zeros((2, 9))
    c0[:, 1] = q0_uM * 1e-6
    c0[:, 2] = s0_uM * 1e-6
    c0[1, 0] = i_on_uM * 1e-6
    s0_m = s0_uM * 1e-6

    base = np.minimum(np.asarray(log10_kds_base, dtype=float), -2.0)
    deltas = np.empty(n_mc)
    folds = np.empty(n_mc)
    for idx in range(n_mc):
        kd_vec = 10.0 ** (base + rng.normal(0.0, sigma, size=6))
        eq = solve_equilibrium(c0, kd_vec)
        s_off = eq[0, 2] / s0_m
        s_on = eq[1, 2] / s0_m
        deltas[idx] = s_on - s_off
        folds[idx] = s_on / max(s_off, 0.005)

    return {
        "MC_Delta_med": float(np.median(deltas)),
        "MC_Delta_p10": float(np.percentile(deltas, 10)),
        "MC_Fold_med": float(np.median(folds)),
        "MC_Pass_rate": float(np.mean((deltas >= 0.40) & (folds >= 3.0))),
    }


def evaluate_triplet(
    model: ConsensusModel,
    i_seq: str,
    q_seq: str,
    s_seq: str,
    i_reg: Optional[str] = None,
    q_reg: Optional[str] = None,
    s_reg: Optional[str] = None,
    s0_uM: float = 0.5,
    q0_uM: float = 1.0,
    i_on_uM: float = 2.5,
    n_mc: int = 200,
    sigma: Optional[float] = None,
) -> dict:
    """Score all 6 dimers of an (I, Q, S) triplet and run EQTK + Monte Carlo checks."""
    pairs = {
        "IQ": model.predict_pair(i_seq, q_seq, i_reg, q_reg),
        "QS": model.predict_pair(q_seq, s_seq, q_reg, s_reg),
        "IS": model.predict_pair(i_seq, s_seq, i_reg, s_reg),
        "II": model.predict_pair(i_seq, i_seq, i_reg, i_reg),
        "QQ": model.predict_pair(q_seq, q_seq, q_reg, q_reg),
        "SS": model.predict_pair(s_seq, s_seq, s_reg, s_reg),
    }
    kd_dict = {k: 10.0 ** v["log10_kd"] for k, v in pairs.items()}
    eq_res = simulate_yes_gate(kd_dict, s0_uM=s0_uM, q0_uM=q0_uM, i_on_uM=i_on_uM)
    log10_vec = np.array([pairs[k]["log10_kd"] for k in ("IQ", "QS", "IS", "II", "QQ", "SS")])
    mc_res = mc_robustness(
        log10_vec,
        s0_uM=s0_uM,
        q0_uM=q0_uM,
        i_on_uM=i_on_uM,
        n_mc=n_mc,
        sigma=model.loo_rmse if sigma is None else sigma,
    )
    m_iq, _ = model.slippage_margin(i_seq, q_seq, i_reg, q_reg)
    m_qs, _ = model.slippage_margin(q_seq, s_seq, q_reg, s_reg)

    return {
        "Kd_IQ_nM": pairs["IQ"]["Kd_nM"],
        "Kd_QS_uM": pairs["QS"]["Kd_nM"] / 1000.0,
        "Kd_IS_uM": pairs["IS"]["Kd_nM"] / 1000.0,
        "Kd_II_uM": pairs["II"]["Kd_nM"] / 1000.0,
        "Kd_QQ_uM": pairs["QQ"]["Kd_nM"] / 1000.0,
        "Kd_SS_uM": pairs["SS"]["Kd_nM"] / 1000.0,
        "SlipMargin_IQ": m_iq,
        "SlipMargin_QS": m_qs,
        **eq_res,
        **mc_res,
    }
