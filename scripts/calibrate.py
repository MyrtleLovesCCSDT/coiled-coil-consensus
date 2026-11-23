"""Calibrate the 4-model 1D consensus on Thomas (2013) + Aronsson (2015) Kd data."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.anchors import (
    ARONSSON_PAIRS,
    ARONSSON_PEPTIDES,
    CROOKS_PAIRS,
    CROOKS_PEPTIDES,
    THOMAS_PAIRS,
    THOMAS_PEPTIDES,
)
from src.consensus import CONSENSUS_MODELS, detect_score_polarity, pad_register, score_raw_models


def build_anchor_table(lower_is_stronger: dict) -> pd.DataFrame:
    rows = []
    sources = [
        ("Thomas_2013", THOMAS_PEPTIDES, THOMAS_PAIRS),
        ("Aronsson_2015", ARONSSON_PEPTIDES, ARONSSON_PAIRS),
        ("Crooks_2017", CROOKS_PEPTIDES, CROOKS_PAIRS),
    ]
    for dataset, pep_dict, pair_list in sources:
        for item in pair_list:
            if dataset == "Crooks_2017":
                p1, p2, tm, grp = item
                kd = None
            else:
                p1, p2, tm, kd, grp = item
            s1, r1 = pep_dict[p1]
            s2, r2 = pep_dict[p2]
            scores = score_raw_models(
                pad_register(s1, r1),
                pad_register(s2, r2),
                lower_is_stronger,
                scan_shifts=True,
            )
            rows.append({
                "dataset": dataset,
                "group": grp,
                "p1": p1,
                "p2": p2,
                "tm_exp": np.nan if tm is None else float(tm),
                "kd_exp_M": np.nan if kd is None else float(kd),
                "log10_kd": np.log10(kd) if (kd is not None and kd > 0) else np.nan,
                **scores,
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Fit 1D Kd consensus calibration.")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "configs" / "consensus_calibration.json",
    )
    parser.add_argument("--save-csv", action="store_true")
    args = parser.parse_args()

    lower_is_stronger = detect_score_polarity()
    df = build_anchor_table(lower_is_stronger)

    kd_df = df[df["log10_kd"].notna()].copy()
    x_kd = kd_df[CONSENSUS_MODELS].values
    y_kd = kd_df["log10_kd"].values

    pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 40)))
    y_loo = cross_val_predict(pipe, x_kd, y_kd, cv=len(kd_df))
    pipe.fit(x_kd, y_kd)

    r_loo, _ = stats.pearsonr(y_kd, y_loo)
    rho_loo, _ = stats.spearmanr(y_kd, y_loo)
    rmse_loo = float(np.sqrt(np.mean((y_kd - y_loo) ** 2)))
    print(f"LOO-CV Kd (n={len(kd_df)}): r={r_loo:+.3f}, R2={r_loo**2:.3f}, rho={rho_loo:+.3f}, RMSE={rmse_loo:.3f}")

    df["Consensus_pKd"] = -pipe.predict(df[CONSENSUS_MODELS].values)
    cr = df[df["dataset"] == "Crooks_2017"]
    r_cr, _ = stats.pearsonr(cr["Consensus_pKd"], cr["tm_exp"])
    sl_tm, in_tm, _, _, _ = stats.linregress(cr["Consensus_pKd"], cr["tm_exp"])
    print(f"Zero-shot Crooks Tm (n={len(cr)}): r={r_cr:+.3f}, R2={r_cr**2:.3f}")

    scaler = pipe.named_steps["standardscaler"]
    ridge = pipe.named_steps["ridgecv"]
    payload = {
        "consensus_models": CONSENSUS_MODELS,
        "lower_is_stronger": lower_is_stronger,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "ridge_coef_log10_kd": ridge.coef_.tolist(),
        "ridge_intercept_log10_kd": float(ridge.intercept_),
        "ridge_alpha": float(ridge.alpha_),
        "loo_r": float(r_loo),
        "loo_rmse": rmse_loo,
        "tm_from_pKd_slope": float(sl_tm),
        "tm_from_pKd_intercept": float(in_tm),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved calibration to {args.out}")

    if args.save_csv:
        res_dir = ROOT / "results"
        res_dir.mkdir(exist_ok=True)
        df.to_csv(res_dir / "anchor_scored.csv", index=False)


if __name__ == "__main__":
    main()
