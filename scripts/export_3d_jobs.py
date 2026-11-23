import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.anchors import ARONSSON_PAIRS, ARONSSON_PEPTIDES, THOMAS_PAIRS, THOMAS_PEPTIDES


def _clean_seq(seq: str) -> str:
    return str(seq).strip().lstrip("-")


def _safe_tag(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")


def write_3d_job(out_dir: Path, job_id: str, chains: List[str]) -> None:
    fasta_dir = out_dir / "colabfold_fasta"
    yaml_dir = out_dir / "boltz_yaml"
    fasta_dir.mkdir(parents=True, exist_ok=True)
    yaml_dir.mkdir(parents=True, exist_ok=True)

    clean = [_clean_seq(s) for s in chains]
    with open(fasta_dir / f"{job_id}.fasta", "w") as f:
        f.write(f">{job_id}\n" + ":".join(clean) + "\n")

    chain_ids = ["A", "B", "C", "D"]
    with open(yaml_dir / f"{job_id}.yaml", "w") as f:
        f.write("version: 1\nsequences:\n")
        for cid, seq in zip(chain_ids, clean):
            f.write(f"  - protein:\n      id: {cid}\n      sequence: {seq}\n      msa: empty\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shortlist", type=Path, default=ROOT / "results" / "shortlist_yes_gates.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "3d_inputs")
    parser.add_argument("--no-controls", action="store_true")
    args = parser.parse_args()

    df_short = pd.read_csv(args.shortlist)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    seen_jobs: Dict[Tuple[str, ...], str] = {}

    for idx, r in df_short.iterrows():
        dtag = f"D{idx + 1:02d}"
        q_seq = _clean_seq(r["Q_seq"])
        s_seq = _clean_seq(r["S_seq"])
        i_seq = _clean_seq(r["I_seq"])

        jobs = [
            ("dimer_IQ", [i_seq, q_seq]),
            ("dimer_QS", [q_seq, s_seq]),
            ("dimer_IS", [i_seq, s_seq]),
            ("homo_II", [i_seq, i_seq]),
            ("homo_QQ", [q_seq, q_seq]),
            ("homo_SS", [s_seq, s_seq]),
            ("trimer_IQQ", [i_seq, q_seq, q_seq]),
            ("trimer_IIQ", [i_seq, i_seq, q_seq]),
            ("trimer_QQQ", [q_seq, q_seq, q_seq]),
        ]
        for role, chains in jobs:
            key = tuple(chains)
            if key not in seen_jobs:
                job_id = f"{dtag}__{role}"
                seen_jobs[key] = job_id
                write_3d_job(args.out_dir, job_id, chains)
            else:
                job_id = seen_jobs[key]

            rows.append({
                "job_id": job_id,
                "design_id": dtag,
                "role": role,
                "category": "design",
                "seqs": ":".join(chains),
                "exp_log10_kd": np.nan,
            })

    if not args.no_controls:
        for dataset, pep_dict, pair_list in [
            ("Thomas_2013", THOMAS_PEPTIDES, THOMAS_PAIRS),
            ("Aronsson_2015", ARONSSON_PEPTIDES, ARONSSON_PAIRS),
        ]:
            for p1, p2, _, kd, _ in pair_list:
                if kd is None or kd <= 0:
                    continue
                s1 = _clean_seq(pep_dict[p1][0])
                s2 = _clean_seq(pep_dict[p2][0])
                chains = [s1, s2]
                key = tuple(chains)
                job_id = _safe_tag(f"CTRL_{dataset}_{p1}__{p2}")
                if key not in seen_jobs:
                    seen_jobs[key] = job_id
                    write_3d_job(args.out_dir, job_id, chains)
                else:
                    job_id = seen_jobs[key]

                rows.append({
                    "job_id": job_id,
                    "design_id": "CTRL",
                    "role": f"{p1}:{p2}",
                    "category": dataset,
                    "seqs": ":".join(chains),
                    "exp_log10_kd": float(np.log10(kd)),
                })

    manifest = pd.DataFrame(rows)
    manifest_path = args.out_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"Exported {len(seen_jobs)} unique 3D jobs to {args.out_dir}")
    print(f"Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
