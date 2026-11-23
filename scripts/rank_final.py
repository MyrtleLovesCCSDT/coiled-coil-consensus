import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shortlist", type=Path, default=ROOT / "results" / "shortlist_yes_gates.csv")
    parser.add_argument("--manifest", type=Path, default=ROOT / "results" / "3d_inputs" / "manifest.csv")
    parser.add_argument("--rosetta-summary", type=Path, default=ROOT / "results" / "rosetta_3d_summary.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--ref-homo-dg", type=float, default=None)
    parser.add_argument("--homo-dg-tol", type=float, default=1.5)
    parser.add_argument("--max-dg-qs", type=float, default=-50.0)
    parser.add_argument("--min-gap-iq-qs", type=float, default=2.5)
    parser.add_argument("--min-gap-qs-is", type=float, default=3.5)
    parser.add_argument("--min-gap-homo", type=float, default=2.5)
    parser.add_argument("--min-gap-qs-ss", type=float, default=2.0)
    parser.add_argument("--max-dimer-unsat", type=float, default=1.0)
    parser.add_argument("--min-trimer-contacts", type=int, default=25)
    parser.add_argument("--min-trimer-iptm", type=float, default=0.70)
    parser.add_argument("--max-clean-trimer-unsat", type=float, default=1.0)
    parser.add_argument("--min-clean-trimer-pack", type=float, default=0.60)
    parser.add_argument("--min-clean-trimer-sc", type=float, default=0.78)
    parser.add_argument("--min-mc-pass", type=float, default=0.65)
    args = parser.parse_args()

    des = pd.read_csv(args.shortlist)
    man = pd.read_csv(args.manifest)
    ros = pd.read_csv(args.rosetta_summary)
    ros_idx = ros.set_index("job_id").to_dict(orient="index")

    role_map = {
        (str(r["design_id"]), str(r["role"])): str(r["job_id"])
        for _, r in man[man["design_id"] != "CTRL"].iterrows()
    }

    ctrl_rows = (
        man[(man["design_id"] == "CTRL") & man["exp_log10_kd"].notna()][["job_id", "role", "exp_log10_kd"]]
        .drop_duplicates("job_id")
    )
    c_data = []
    ref_homo_dg = args.ref_homo_dg
    for _, cr in ctrl_rows.iterrows():
        jid = cr["job_id"]
        if jid in ros_idx and pd.notna(ros_idx[jid].get("dG_separated")):
            dg = float(ros_idx[jid]["dG_separated"])
            c_data.append({"job_id": jid, "role": cr["role"], "exp_log10Kd": float(cr["exp_log10_kd"]), "dG": dg})
            if ref_homo_dg is None and cr["role"] == "BN4:BN4":
                ref_homo_dg = dg

    if ref_homo_dg is None:
        raise RuntimeError(
            "Reference homodimer control 'BN4:BN4' not found in rosetta_summary. "
            "Run export_3d_jobs.py with controls or pass --ref-homo-dg explicitly."
        )

    cdf = pd.DataFrame(c_data)
    has_calib = len(cdf) >= 3
    if has_calib:
        sl_ros, in_ros, r_ros, _, _ = stats.linregress(cdf["dG"], cdf["exp_log10Kd"])
        rho_ros, _ = stats.spearmanr(cdf["dG"], cdf["exp_log10Kd"])
        print(f"Rosetta calibration (n={len(cdf)}): r={r_ros:+.3f}, rho={rho_ros:+.3f} | log10(Kd) = {sl_ros:.3f}*dG {in_ros:+.2f}")

    def get_val(dtag: str, role: str, field: str, default=np.nan):
        jid = role_map.get((dtag, role))
        if jid and jid in ros_idx:
            return ros_idx[jid].get(field, default)
        return default

    master = []
    for idx, r in des.iterrows():
        dtag = f"D{idx + 1:02d}"
        dg_iq = get_val(dtag, "dimer_IQ", "dG_separated")
        dg_qs = get_val(dtag, "dimer_QS", "dG_separated")
        dg_is = get_val(dtag, "dimer_IS", "dG_separated")
        dg_ii = get_val(dtag, "homo_II", "dG_separated")
        dg_qq = get_val(dtag, "homo_QQ", "dG_separated")
        dg_ss = get_val(dtag, "homo_SS", "dG_separated")

        gap_iq_qs = dg_qs - dg_iq
        gap_qs_is = dg_is - dg_qs
        gap_iq_qq = dg_qq - dg_iq
        gap_iq_ii = dg_ii - dg_iq
        gap_qs_ss = dg_ss - dg_qs

        unsat_iq = get_val(dtag, "dimer_IQ", "delta_unsatHbonds")
        unsat_qs = get_val(dtag, "dimer_QS", "delta_unsatHbonds")
        unsat_tri = float(np.nanmean([
            get_val(dtag, "trimer_IQQ", "delta_unsatHbonds"),
            get_val(dtag, "trimer_IIQ", "delta_unsatHbonds"),
            get_val(dtag, "trimer_QQQ", "delta_unsatHbonds"),
        ]))
        pack_dim = float(np.nanmean([get_val(dtag, "dimer_IQ", "packstat"), get_val(dtag, "dimer_QS", "packstat")]))
        pack_tri = float(np.nanmean([get_val(dtag, "trimer_IQQ", "packstat"), get_val(dtag, "trimer_IIQ", "packstat")]))
        sc_dim = float(np.nanmean([get_val(dtag, "dimer_IQ", "sc_value"), get_val(dtag, "dimer_QS", "sc_value")]))
        sc_tri = float(np.nanmean([get_val(dtag, "trimer_IQQ", "sc_value"), get_val(dtag, "trimer_IIQ", "sc_value")]))
        asn_iiq = get_val(dtag, "trimer_IIQ", "min_Asn_CA")

        trimer_traps = []
        for trole in ("trimer_IQQ", "trimer_IIQ", "trimer_QQQ"):
            c_cont = get_val(dtag, trole, "C_to_AB_contacts")
            m_piptm = get_val(dtag, trole, "min_pair_iptm")
            t_unsat = get_val(dtag, trole, "delta_unsatHbonds")
            t_pack = get_val(dtag, trole, "packstat")
            t_sc = get_val(dtag, trole, "sc_value")
            if any(pd.isna(v) for v in (c_cont, m_piptm, t_unsat, t_pack, t_sc)):
                trimer_traps.append(True)
                continue
            assembled = (c_cont >= args.min_trimer_contacts) and (m_piptm >= args.min_trimer_iptm)
            clean_core = (
                (t_unsat <= args.max_clean_trimer_unsat)
                and (t_pack >= args.min_clean_trimer_pack)
                and (t_sc >= args.min_clean_trimer_sc)
            )
            trimer_traps.append(bool(assembled and clean_core))
        trimer_safe = not any(trimer_traps)

        qq_pen = max(0.0, ref_homo_dg - dg_qq) if pd.notna(dg_qq) else 0.0
        ii_pen = max(0.0, ref_homo_dg - dg_ii) if pd.notna(dg_ii) else 0.0
        score_3d = (
            2.0 * gap_iq_qs
            + 1.5 * gap_qs_is
            + 1.0 * gap_qs_ss
            - 2.0 * (qq_pen + ii_pen)
            - 1.0 * (unsat_iq if pd.notna(unsat_iq) else 0.0)
            + 10.0 * float(r["MC_Pass_rate"])
            + 1.5 * (unsat_tri - (unsat_iq if pd.notna(unsat_iq) else 0.0))
        )

        req_vals = (dg_iq, dg_qs, dg_is, dg_ii, dg_qq, dg_ss, unsat_iq, unsat_qs)
        passed_3d = bool(
            all(pd.notna(v) for v in req_vals)
            and dg_qs <= args.max_dg_qs
            and gap_iq_qs >= args.min_gap_iq_qs
            and gap_qs_is >= args.min_gap_qs_is
            and gap_iq_qq >= args.min_gap_homo
            and gap_iq_ii >= args.min_gap_homo
            and gap_qs_ss >= args.min_gap_qs_ss
            and dg_qq >= ref_homo_dg - args.homo_dg_tol
            and dg_ii >= ref_homo_dg - args.homo_dg_tol
            and unsat_iq <= args.max_dimer_unsat
            and unsat_qs <= args.max_dimer_unsat
            and trimer_safe
            and float(r["MC_Pass_rate"]) >= args.min_mc_pass
        )

        master.append({
            "ID": dtag, "Class": r["Class"], "Regime": r["Regime"],
            "1D_IQ_nM": r["Kd_IQ_nM"], "1D_QS_uM": r["Kd_QS_uM"], "1D_QQ_uM": r["Kd_QQ_uM"],
            "MC_Pass": r["MC_Pass_rate"], "MC_Fold": r["MC_Fold_med"],
            "dG_IQ": dg_iq, "dG_QS": dg_qs, "dG_IS": dg_is,
            "dG_II": dg_ii, "dG_QQ": dg_qq, "dG_SS": dg_ss,
            "Gap_IQ_QS": gap_iq_qs, "Gap_QS_IS": gap_qs_is,
            "Gap_IQ_QQ": gap_iq_qq, "Gap_IQ_II": gap_iq_ii, "Gap_QS_SS": gap_qs_ss,
            "unsat_IQ": unsat_iq, "unsat_QS": unsat_qs, "unsat_Tri_avg": unsat_tri,
            "pack_Dim": pack_dim, "pack_Tri": pack_tri,
            "sc_Dim": sc_dim, "sc_Tri": sc_tri,
            "Asn_IIQ_A": asn_iiq,
            "Trimer_Safe": trimer_safe,
            "Passed_3D": passed_3d,
            "Total_Score": score_3d,
            "Q_seq": r["Q_seq"], "S_seq": r["S_seq"], "I_seq": r["I_seq"],
        })

    mdf = pd.DataFrame(master).sort_values(["Passed_3D", "Total_Score"], ascending=[False, False]).reset_index(drop=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    mdf.to_csv(args.out_dir / "final_master_scorecard.csv", index=False)

    show = [
        "ID", "Class", "Regime", "Trimer_Safe", "Passed_3D", "1D_IQ_nM", "1D_QS_uM", "MC_Pass",
        "dG_IQ", "dG_QS", "dG_IS", "dG_II", "dG_QQ", "dG_SS",
        "Gap_IQ_QS", "Gap_QS_IS", "unsat_Tri_avg", "Total_Score",
    ]
    print("\n" + mdf[show].to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    winners = mdf[mdf["Passed_3D"]]
    if not winners.empty:
        print("\nShared (Q, I) sets among passed candidates:")
        for (q, i_seq), grp in winners.groupby(["Q_seq", "I_seq"]):
            if len(grp) >= 2:
                print(f"  {list(grp['ID'])}: Q={q} | I={i_seq}")
                for _, gr in grp.iterrows():
                    print(f"    -> S ({gr['ID']}, dG_QS={gr['dG_QS']:.2f}): {gr['S_seq']}")

    plot_df = winners.head(5) if not winners.empty else mdf.head(5)
    if not plot_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=150)

        ax = axes[0]
        if has_calib:
            ax.scatter(cdf["dG"], cdf["exp_log10Kd"], s=80, color="#2b8cbe", zorder=4, label="Experimental controls")
            for _, cr in cdf.iterrows():
                ax.annotate(cr["role"], (cr["dG"], cr["exp_log10Kd"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
            xs = np.linspace(cdf["dG"].min() - 3, cdf["dG"].max() + 3, 50)
            ax.plot(xs, sl_ros * xs + in_ros, "k--", lw=1.2, label=f"r={r_ros:.2f}, rho={rho_ros:.2f}")
        top1 = plot_df.iloc[0]
        for role_lbl, dg_val, col in [
            ("Top1 I:Q", top1["dG_IQ"], "#1b9e77"),
            ("Top1 Q:S", top1["dG_QS"], "#d95f02"),
            ("Top1 I:S", top1["dG_IS"], "#e41a1c"),
            ("Top1 I:I", top1["dG_II"], "#377eb8"),
            ("Top1 Q:Q", top1["dG_QQ"], "#7570b3"),
        ]:
            if pd.notna(dg_val):
                ax.axvline(dg_val, color=col, ls=":", lw=1.5, label=f"{role_lbl} ({dg_val:.1f} REU)")
        ax.set_title("PyRosetta Calibration & Top-1 Hierarchy")
        ax.set_xlabel("Rosetta Interface dG_separated (REU)")
        ax.set_ylabel("Experimental log10(Kd [M])")
        ax.legend(fontsize=7.5)

        ax = axes[1]
        x_pos = np.arange(len(plot_df))
        for role_col, lbl, col, off in [
            ("dG_IQ", "I:Q (target strong)", "#1b9e77", -0.32),
            ("dG_QS", "Q:S (signal mod)", "#d95f02", -0.19),
            ("dG_IS", "I:S (crosstalk)", "#e41a1c", -0.06),
            ("dG_II", "I:I (homo)", "#377eb8", 0.07),
            ("dG_QQ", "Q:Q (homo)", "#7570b3", 0.20),
            ("dG_SS", "S:S (homo)", "#666666", 0.33),
        ]:
            ax.bar(x_pos + off, -plot_df[role_col], width=0.12, label=lbl, color=col)
        ax.axhline(-ref_homo_dg, color="purple", ls="--", lw=1, label=f"BN4 homo ({ref_homo_dg:.1f} REU)")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"{r['ID']}\n({r['Class'][:10]})" for _, r in plot_df.iterrows()], fontsize=8)
        ax.set_ylabel("-dG_separated (REU, higher = stronger)")
        ax.set_title("Top Candidates: 3D Rosetta Energy Profile")
        ax.legend(fontsize=7.5)

        plt.tight_layout()
        plt.savefig(args.out_dir / "final_3d_master_summary.png")
        plt.close(fig)

    print(f"\nSaved scorecard and plot to {args.out_dir}")


if __name__ == "__main__":
    main()
