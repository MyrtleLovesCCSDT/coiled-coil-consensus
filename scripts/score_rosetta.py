import argparse
import sys
import time
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
import pyrosetta
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def cif_to_pdb(cif_path: Path, pdb_path: Path) -> None:
    st = gemmi.read_structure(str(cif_path))
    st.write_pdb(str(pdb_path))


def run_iam(pose, jump: int, sfxn) -> dict:
    p = pose.clone()
    iam = pyrosetta.rosetta.protocols.analysis.InterfaceAnalyzerMover(jump, False, sfxn, True, False, True)
    if hasattr(iam, "set_compute_interface_sc"):
        iam.set_compute_interface_sc(True)
    iam.apply(p)
    return dict(p.scores)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boltz-summary", type=Path, default=ROOT / "results" / "boltz_3d_summary.csv")
    parser.add_argument("--manifest", type=Path, default=ROOT / "results" / "3d_inputs" / "manifest.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "rosetta")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "rosetta_3d_summary.csv")
    args = parser.parse_args()

    pyrosetta.init("-mute all -ignore_unrecognized_res -ex1 -ex2")
    sfxn = pyrosetta.get_fa_scorefxn()
    relax = pyrosetta.rosetta.protocols.relax.FastRelax(sfxn, 1)

    boltz_df = pd.read_csv(args.boltz_summary)
    manifest = pd.read_csv(args.manifest)
    ctrl_meta = manifest[["job_id", "exp_log10_kd"]].drop_duplicates("job_id")

    tmp_dir = args.out_dir / "tmp_pdb"
    rel_dir = args.out_dir / "relaxed_pdb"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    rel_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    t0 = time.time()
    n_total = len(boltz_df)

    for idx, row in boltz_df.iterrows():
        job_id = row["job_id"]
        cif_path = Path(row["cif_path"])
        if not cif_path.exists():
            continue

        pdb_in = tmp_dir / f"{job_id}.pdb"
        pdb_rel = rel_dir / f"{job_id}_relaxed.pdb"
        cif_to_pdb(cif_path, pdb_in)

        pose = pyrosetta.pose_from_pdb(str(pdb_in))
        n_chains = pose.num_chains()
        if n_chains < 2:
            continue

        relax.apply(pose)
        pose.dump_pdb(str(pdb_rel))

        total_score = float(sfxn(pose))
        energies = pose.energies().total_energies()
        fa_elec = float(energies[pyrosetta.rosetta.core.scoring.fa_elec])
        fa_sol = float(energies[pyrosetta.rosetta.core.scoring.fa_sol])
        hbond_sc = float(energies[pyrosetta.rosetta.core.scoring.hbond_sc])

        iface = "A_B" if n_chains == 2 else "AB_C"
        jump = 1 if n_chains == 2 else 2
        sc = run_iam(pose, jump, sfxn)

        rows.append({
            "job_id": job_id,
            "interface_checked": iface,
            "total_score": total_score,
            "fa_elec": fa_elec,
            "fa_sol": fa_sol,
            "hbond_sc": hbond_sc,
            "dG_separated": sc.get("dG_separated", np.nan),
            "dG_per_SASA100": sc.get("dG_separated/dSASAx100", np.nan),
            "dSASA_int": sc.get("dSASA_int", np.nan),
            "dSASA_hphobic": sc.get("dSASA_hphobic", np.nan),
            "dSASA_polar": sc.get("dSASA_polar", np.nan),
            "hbonds_int": sc.get("hbonds_int", np.nan),
            "delta_unsatHbonds": sc.get("delta_unsatHbonds", np.nan),
            "packstat": sc.get("packstat", np.nan),
            "sc_value": sc.get("sc_value", np.nan),
        })
        print(
            f"[{idx + 1:2d}/{n_total}] {job_id:38s} | dG={rows[-1]['dG_separated']:7.2f} | "
            f"unsatHB={rows[-1]['delta_unsatHbonds']:4.1f} | sc={rows[-1]['sc_value']:.2f} ({time.time() - t0:.1f}s)",
            flush=True,
        )

    boltz_cols = [
        "job_id", "category", "iptm", "min_pair_iptm", "plddt",
        "ipde", "min_Asn_CA", "C_to_AB_contacts",
    ]
    res_df = (
        pd.DataFrame(rows)
        .merge(boltz_df[boltz_cols], on="job_id", how="left")
        .merge(ctrl_meta, on="job_id", how="left")
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(args.out, index=False)

    ctrl = res_df[res_df["exp_log10_kd"].notna()]
    if len(ctrl) >= 4:
        print("\nControl correlations:")
        for col in ["dG_separated", "dG_per_SASA100", "hbonds_int", "sc_value"]:
            sub = ctrl[[col, "exp_log10_kd"]].dropna()
            if len(sub) >= 4:
                r_v, _ = stats.pearsonr(sub[col], sub["exp_log10_kd"])
                rho_v, _ = stats.spearmanr(sub[col], sub["exp_log10_kd"])
                print(f"  {col:18s}: r={r_v:+.3f} | rho={rho_v:+.3f}")

    print(f"\nSaved {len(res_df)} rows to {args.out}")


if __name__ == "__main__":
    main()
