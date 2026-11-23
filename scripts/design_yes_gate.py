"""De novo coiled-coil YES-gate design with EQTK equilibrium and Monte Carlo filtering."""

import argparse
import itertools
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.consensus import ConsensusModel, DEFAULT_CALIB_PATH
from src.yes_gate import mc_robustness, simulate_yes_gate, solve_equilibrium

# Концентрация, при которой подбирались исходные пороги по гомодимерам.
REF_Q0_uM = 1.0


def net_charge(seq: str) -> int:
    return seq.count("K") + seq.count("R") - seq.count("E") - seq.count("D")


def calc_eps280(seq: str) -> int:
    s = seq.strip().lstrip("-")
    return 5500 * s.count("W") + 1490 * s.count("Y")


def build_soluble_4mer(a_tuple, g_tuple, e_tuple, charge_sign: int):
    bc = [("A", "S"), ("S", "A"), ("A", "S"), ("S", "A")]
    opp = "K" if charge_sign < 0 else "E"
    eg_dom = sum(1 for x in g_tuple + e_tuple if x == ("E" if charge_sign < 0 else "K"))
    f_list = [opp, opp, "W", opp] if eg_dom >= 6 else [opp, "S", "W", opp]
    raw = "".join(f"{g_tuple[h]}{a_tuple[h]}{bc[h][0]}{bc[h][1]}L{e_tuple[h]}{f_list[h]}" for h in range(4))
    return "-" + raw, raw


def build_soluble_35mer(a_tuple3, g_tuple3, e_tuple3, half_g_e=("E",)):
    bc = [("S", "A"), ("A", "S"), ("S", "A")]
    f_list = ["S", "W", "K"]
    lead = f"L{half_g_e[0]}K"
    body = "".join(f"{g_tuple3[h]}{a_tuple3[h]}{bc[h][0]}{bc[h][1]}L{e_tuple3[h]}{f_list[h]}" for h in range(3))
    raw = lead + body
    return "-----" + raw, raw


def build_soluble_3mer(a_tuple3, g_tuple3, e_tuple3):
    bc = [("S", "A"), ("A", "S"), ("S", "A")]
    f_list = ["S", "W", "K"]
    raw = "".join(f"{g_tuple3[h]}{a_tuple3[h]}{bc[h][0]}{bc[h][1]}L{e_tuple3[h]}{f_list[h]}" for h in range(3))
    return "-" + raw, raw


def generate_pools(model: ConsensusModel, homo_min_nM: float = 30_000.0):
    q_pool, seen_q = [], set()
    for a_tup in [("I", "I", "N", "I"), ("I", "N", "I", "I"), ("V", "I", "N", "I")]:
        for comb in itertools.product(["K", "Q"], repeat=8):
            if comb.count("K") < 5:
                continue
            pad, raw = build_soluble_4mer(a_tup, comb[:4], comb[4:], charge_sign=+1)
            chg = net_charge(raw)
            if not (3 <= chg <= 6) or raw in seen_q:
                continue
            seen_q.add(raw)
            sc_homo = model.predict_pair(pad, pad)
            if sc_homo["Kd_nM"] < homo_min_nM:
                continue
            q_pool.append({
                "raw": raw, "pad": pad, "len": len(raw), "charge": chg,
                "core_a": "".join(a_tup), "homo_pKd": sc_homo["pKd"],
                "homo_Kd_uM": sc_homo["Kd_nM"] / 1000.0,
            })

    acid_pool, seen_a = [], set()
    a_cores_4 = [
        ("I", "I", "N", "I"), ("I", "N", "I", "I"), ("V", "V", "N", "V"),
        ("I", "V", "N", "V"), ("V", "N", "V", "V"), ("V", "I", "N", "I"),
    ]
    for a_tup in a_cores_4:
        for comb in itertools.product(["E", "Q", "A"], repeat=8):
            n_e, n_a = comb.count("E"), comb.count("A")
            if n_e < 4 or n_a > 2 or ("V" in a_tup and n_e < 5):
                continue
            pad, raw = build_soluble_4mer(a_tup, comb[:4], comb[4:], charge_sign=-1)
            chg = net_charge(raw)
            if not (-6 <= chg <= -2) or raw in seen_a:
                continue
            seen_a.add(raw)
            sc_homo = model.predict_pair(pad, pad)
            if sc_homo["Kd_nM"] < homo_min_nM:
                continue
            acid_pool.append({
                "raw": raw, "pad": pad, "len": len(raw), "charge": chg,
                "core_a": "".join(a_tup), "homo_pKd": sc_homo["pKd"],
                "homo_Kd_uM": sc_homo["Kd_nM"] / 1000.0,
            })

    for a_tup3 in [("I", "N", "I"), ("V", "N", "V"), ("I", "N", "V")]:
        for comb6 in itertools.product(["E", "Q"], repeat=6):
            if comb6.count("E") < 4:
                continue
            for pad, raw in [
                build_soluble_35mer(a_tup3, comb6[:3], comb6[3:]),
                build_soluble_3mer(a_tup3, comb6[:3], comb6[3:]),
            ]:
                chg = net_charge(raw)
                if not (-6 <= chg <= -3) or raw in seen_a:
                    continue
                seen_a.add(raw)
                sc_homo = model.predict_pair(pad, pad)
                if sc_homo["Kd_nM"] < homo_min_nM:
                    continue
                acid_pool.append({
                    "raw": raw, "pad": pad, "len": len(raw), "charge": chg,
                    "core_a": "".join(a_tup3), "homo_pKd": sc_homo["pKd"],
                    "homo_Kd_uM": sc_homo["Kd_nM"] / 1000.0,
                })

    return q_pool, acid_pool


def classify_design(r) -> str:
    if r["S_len"] < 28:
        return f"Truncated_{int(r['S_len'])}aa"
    if "V" in str(r["S_core"]):
        return "4mer_Val_core"
    return "4mer_Ile_core_QA"


def plot_titration(df_short: pd.DataFrame, out_path: Path, s0_uM: float, q0_uM: float, i_on_uM: float):
    if df_short.empty:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=150)
    i_grid = np.linspace(0, i_on_uM * 1.4, 80)
    for _, r in df_short.drop_duplicates("Class").head(4).iterrows():
        kd_vec = np.array([
            r["Kd_IQ_nM"] * 1e-9, r["Kd_QS_uM"] * 1e-6, r["Kd_IS_uM"] * 1e-6,
            r["Kd_II_uM"] * 1e-6, r["Kd_QQ_uM"] * 1e-6, r["Kd_SS_uM"] * 1e-6,
        ])
        c0 = np.zeros((len(i_grid), 9))
        c0[:, 0] = i_grid * 1e-6
        c0[:, 1] = q0_uM * 1e-6
        c0[:, 2] = s0_uM * 1e-6
        eq = solve_equilibrium(c0, kd_vec)
        lbl = f"{r['Class']}: IQ={r['Kd_IQ_nM']:.1f}nM, QS={r['Kd_QS_uM']:.2f}uM (MC {100*r['MC_Pass_rate']:.0f}%)"
        ax.plot(i_grid, eq[:, 2] / (s0_uM * 1e-6), lw=2, label=lbl)
    ax.axvline(q0_uM, color="gray", ls="--", lw=1, label=f"[Q]0 = {q0_uM:.1f} uM")
    ax.set_xlabel("[I]0 (uM)")
    ax.set_ylabel("[S]free / [S]0")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(f"Designed YES-gate titration curves (S0={s0_uM} uM, Q0={q0_uM} uM)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path)


def main():
    parser = argparse.ArgumentParser(description="Search and audit de novo YES-gate triplets.")
    parser.add_argument("--calib", type=Path, default=DEFAULT_CALIB_PATH)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--n-mc", type=int, default=200)
    parser.add_argument("--S0", type=float, default=0.5, help="[S]0 in uM")
    parser.add_argument("--Q0", type=float, default=5.0, help="[Q]0 in uM")
    parser.add_argument("--Ion", type=float, default=12.5, help="[I]on in uM")
    args = parser.parse_args()

    model = ConsensusModel(args.calib)

    # Пороги ниже были откалиброваны при Q0 = 1 uM. Доля мономера, уходящая
    # в паразитный комплекс, идёт как ~2*C0/Kd, поэтому при другой рабочей
    # концентрации их надо масштабировать, иначе жёсткость фильтра плывёт.
    conc_scale = args.Q0 / REF_Q0_uM
    homo_min_nM = 30_000.0 * conc_scale
    safe_min_uM = 300.0 * conc_scale
    print(f"Concentrations: S0={args.S0} Q0={args.Q0} Ion={args.Ion} uM "
          f"-> scale={conc_scale:.2f}x, homo cutoff={homo_min_nM/1000:.0f} uM, "
          f"safe cutoff={safe_min_uM:.0f} uM")

    t0 = time.time()
    q_pool, acid_pool = generate_pools(model, homo_min_nM=homo_min_nM)
    print(f"Pools ready in {time.time() - t0:.1f}s: Q={len(q_pool)}, Acidic={len(acid_pool)}")

    triplets = []
    t1 = time.time()
    for q_idx, Q in enumerate(q_pool, 1):
        if q_idx % 15 == 0 or q_idx == len(q_pool):
            print(f"  [{q_idx}/{len(q_pool)}] triplets={len(triplets)} "
                  f"({time.time() - t1:.0f}s)", flush=True)
        strong_i, mod_s = [], []
        for A in acid_pool:
            sc = model.predict_pair(Q["pad"], A["pad"])
            if sc["pKd"] >= 7.4 and sc["iCipa_best_orient"] == "PAR" and sc["iCipa_best_shift"] == 0:
                strong_i.append((A, sc))
            elif 5.60 <= sc["pKd"] <= 6.85 and sc["iCipa_best_orient"] == "PAR":
                mod_s.append((A, sc))
        if not strong_i or not mod_s:
            continue

        strong_i.sort(key=lambda x: x[1]["pKd"], reverse=True)
        for I, sc_iq in strong_i[:6]:
            for S, sc_qs in mod_s:
                if I["raw"] == S["raw"] or (sc_iq["pKd"] - sc_qs["pKd"]) < 1.5:
                    continue
                sc_is = model.predict_pair(I["pad"], S["pad"])
                if sc_is["Kd_nM"] < homo_min_nM:
                    continue
                agree_iq_qs = sum(sc_iq[m] > sc_qs[m] for m in model.models)
                agree_qs_is = sum(sc_qs[m] > sc_is[m] for m in model.models)
                if agree_iq_qs < 4 or agree_qs_is < 3:
                    continue

                kd_dict = {
                    "IQ": 10.0 ** (-sc_iq["pKd"]), "QS": 10.0 ** (-sc_qs["pKd"]), "IS": 10.0 ** (-sc_is["pKd"]),
                    "II": 10.0 ** (-I["homo_pKd"]), "QQ": 10.0 ** (-Q["homo_pKd"]), "SS": 10.0 ** (-S["homo_pKd"]),
                }
                eq_res = simulate_yes_gate(kd_dict, s0_uM=args.S0, q0_uM=args.Q0, i_on_uM=args.Ion)
                if eq_res["Delta_S_free"] < 0.45 or eq_res["IS_leak_ON"] > 0.05:
                    continue

                triplets.append({
                    "Q_seq": Q["raw"], "S_seq": S["raw"], "I_seq": I["raw"],
                    "Q_pad": Q["pad"], "S_pad": S["pad"], "I_pad": I["pad"],
                    "Q_len": Q["len"], "S_len": S["len"], "I_len": I["len"],
                    "Q_chg": Q["charge"], "S_chg": S["charge"], "I_chg": I["charge"],
                    "Q_core": Q["core_a"], "S_core": S["core_a"], "I_core": I["core_a"],
                    "Kd_IQ_nM": sc_iq["Kd_nM"],
                    "Kd_QS_uM": sc_qs["Kd_nM"] / 1000.0,
                    "Kd_IS_uM": sc_is["Kd_nM"] / 1000.0,
                    "Kd_II_uM": I["homo_Kd_uM"],
                    "Kd_QQ_uM": Q["homo_Kd_uM"],
                    "Kd_SS_uM": S["homo_Kd_uM"],
                    "QS_shift": sc_qs["iCipa_best_shift"],
                    **eq_res,
                })

    df_des = pd.DataFrame(triplets)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if df_des.empty:
        print("No triplets matched filters.")
        return

    df_des = df_des.sort_values(["Delta_S_free", "Fold_ON_OFF", "Kd_IS_uM"], ascending=False).reset_index(drop=True)
    df_des.to_csv(args.out_dir / "designed_yes_gates.csv", index=False)

    safe = df_des[
        (df_des["Kd_QQ_uM"] >= safe_min_uM) & (df_des["Kd_SS_uM"] >= safe_min_uM) &
        (df_des["Kd_II_uM"] >= safe_min_uM) & (df_des["Kd_IS_uM"] >= safe_min_uM) &
        (df_des["QS_shift"] == 0)
    ].copy()
    safe["Class"] = safe.apply(classify_design, axis=1)
    safe["Regime"] = np.where(safe["Kd_IQ_nM"] >= 3.0, "Reversible_3-25nM", "HighContrast_<3nM")

    shortlist = []
    for (cls, reg), grp in safe.groupby(["Class", "Regime"]):
        pool = grp.sort_values(["Delta_S_free", "Kd_QQ_uM"], ascending=False).drop_duplicates("S_seq").head(25)
        scored = []
        for _, r in pool.iterrows():
            m_iq, _ = model.slippage_margin(r["I_pad"], r["Q_pad"])
            m_qs, _ = model.slippage_margin(r["Q_pad"], r["S_pad"])
            if m_iq < 10.0 or m_qs < 8.0:
                continue
            log10_vec = np.array([
                np.log10(r["Kd_IQ_nM"] * 1e-9), np.log10(r["Kd_QS_uM"] * 1e-6),
                np.log10(r["Kd_IS_uM"] * 1e-6), np.log10(r["Kd_II_uM"] * 1e-6),
                np.log10(r["Kd_QQ_uM"] * 1e-6), np.log10(r["Kd_SS_uM"] * 1e-6),
            ])
            mc = mc_robustness(
                log10_vec,
                s0_uM=args.S0,
                q0_uM=args.Q0,
                i_on_uM=args.Ion,
                n_mc=args.n_mc,
                sigma=model.loo_rmse,
            )
            scored.append({
                "Class": cls, "Regime": reg,
                "Kd_IQ_nM": r["Kd_IQ_nM"], "Kd_QS_uM": r["Kd_QS_uM"], "Kd_IS_uM": r["Kd_IS_uM"],
                "Kd_II_uM": r["Kd_II_uM"], "Kd_QQ_uM": r["Kd_QQ_uM"], "Kd_SS_uM": r["Kd_SS_uM"],
                "SlipMargin_IQ": m_iq, "SlipMargin_QS": m_qs,
                "Delta_S": r["Delta_S_free"], "Fold": r["Fold_ON_OFF"],
                **mc,
                "Q_seq": r["Q_seq"], "S_seq": r["S_seq"], "I_seq": r["I_seq"],
                "eps280_Q": calc_eps280(r["Q_seq"]),
                "eps280_S": calc_eps280(r["S_seq"]),
                "eps280_I": calc_eps280(r["I_seq"]),
            })
        if scored:
            shortlist.append(pd.DataFrame(scored).sort_values(["MC_Pass_rate", "MC_Delta_med"], ascending=False).head(2))

    df_short = pd.concat(shortlist, ignore_index=True) if shortlist else pd.DataFrame()
    if not df_short.empty:
        df_short.to_csv(args.out_dir / "shortlist_yes_gates.csv", index=False)
        plot_titration(df_short, args.out_dir / "yes_gate_titration.png", args.S0, args.Q0, args.Ion)
        print(f"Shortlist saved ({len(df_short)} triplets) to {args.out_dir / 'shortlist_yes_gates.csv'}")


if __name__ == "__main__":
    main()
