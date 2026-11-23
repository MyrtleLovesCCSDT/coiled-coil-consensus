import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.structure_3d import analyze_cif


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boltz-dir", type=Path, default=ROOT / "results" / "boltz_out")
    parser.add_argument("--manifest", type=Path, default=ROOT / "results" / "3d_inputs" / "manifest.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "boltz_3d_summary.csv")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    meta = manifest[["job_id", "category"]].drop_duplicates("job_id")

    rows = []
    pattern = str(args.boltz_dir / "**" / "confidence_*_model_0.json")
    for jp in sorted(glob.glob(pattern, recursive=True)):
        jp_path = Path(jp)
        with open(jp_path, "r") as f:
            conf = json.load(f)
        folder = jp_path.parent
        job_id = folder.name
        cif_path = folder / f"{job_id}_model_0.cif"
        if not cif_path.exists():
            cifs = sorted(folder.glob("*_model_0.cif"))
            if not cifs:
                continue
            cif_path = cifs[0]

        geom = analyze_cif(cif_path)

        pc = conf.get("pair_chains_iptm", {})
        off_diag_iptm = [
            float(val)
            for k1, sub in pc.items()
            for k2, val in sub.items()
            if k1 != k2
        ]
        min_pair_iptm = min(off_diag_iptm) if off_diag_iptm else np.nan
        max_pair_iptm = max(off_diag_iptm) if off_diag_iptm else np.nan

        inter_pae = np.nan
        pae_files = sorted(folder.glob("pae_*_model_0.npz"))
        if pae_files:
            try:
                npz = np.load(pae_files[0])
                pae = npz[list(npz.keys())[0]]
                if pae.ndim == 3:
                    pae = pae[0]
                lens = geom["chain_lens"]
                if len(lens) >= 2 and pae.shape[0] >= sum(lens):
                    l0, l1 = lens[0], lens[1]
                    b1 = pae[:l0, l0 : l0 + l1]
                    b2 = pae[l0 : l0 + l1, :l0]
                    inter_pae = float(0.5 * (np.mean(b1) + np.mean(b2)))
            except Exception:
                pass

        rows.append({
            "job_id": job_id,
            "cif_path": str(cif_path),
            "iptm": conf.get("iptm", np.nan),
            "min_pair_iptm": min_pair_iptm,
            "max_pair_iptm": max_pair_iptm,
            "plddt": conf.get("complex_plddt", np.nan),
            "ipde": conf.get("complex_ipde", np.nan),
            "inter_pae": inter_pae,
            **{k: v for k, v in geom.items() if k != "chain_lens"},
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print(f"No Boltz-2 predictions found in {args.boltz_dir}")
        return

    df = df.merge(meta, on="job_id", how="left")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    cols = [
        "job_id", "category", "iptm", "min_pair_iptm", "plddt",
        "cos_AB", "min_Asn_CA", "AB_contacts_45A", "C_to_AB_contacts", "salt_bridges",
    ]
    print(df.sort_values(["category", "job_id"])[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\nSaved {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
