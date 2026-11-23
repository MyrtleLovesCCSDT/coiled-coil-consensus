from pathlib import Path
from typing import Dict, List
import numpy as np


def parse_cif_atoms(cif_path: Path) -> List[Dict[str, str]]:
    cols: List[str] = []
    atoms: List[Dict[str, str]] = []
    in_atom = False
    with open(cif_path, "r") as f:
        for line in f:
            s = line.strip()
            if s.startswith("_atom_site."):
                in_atom = True
                cols.append(s.split(".")[1])
            elif in_atom and s.startswith("ATOM"):
                parts = s.split()
                if len(parts) >= len(cols):
                    atoms.append(dict(zip(cols, parts)))
            elif in_atom and s.startswith("#"):
                in_atom = False
    return atoms


def _dist(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.linalg.norm(p - q))


def analyze_cif(cif_path: Path) -> dict:
    atoms = parse_cif_atoms(cif_path)
    ca_by_chain = {}
    heavy_by_chain = {}
    asn_by_chain = {}
    salt_pos, salt_neg = {}, {}

    for a in atoms:
        seq_str = a.get("label_seq_id", "")
        if not seq_str.isdigit():
            continue
        ch = a.get("auth_asym_id") or a.get("label_asym_id")
        at = a.get("label_atom_id", "")
        elem = a.get("type_symbol", "")
        res = a.get("label_comp_id", "")
        seq_id = int(seq_str)
        xyz = np.array([float(a["Cartn_x"]), float(a["Cartn_y"]), float(a["Cartn_z"])], dtype=float)

        if elem != "H" and not at.startswith("H"):
            heavy_by_chain.setdefault(ch, []).append(xyz)
        if at == "CA":
            ca_by_chain.setdefault(ch, []).append((seq_id, xyz))
            if res == "ASN":
                asn_by_chain.setdefault(ch, []).append((seq_id, xyz))
        if res == "LYS" and at == "NZ":
            salt_pos.setdefault(ch, []).append(xyz)
        elif res == "ARG" and at in ("NH1", "NH2"):
            salt_pos.setdefault(ch, []).append(xyz)
        elif res == "GLU" and at in ("OE1", "OE2"):
            salt_neg.setdefault(ch, []).append(xyz)
        elif res == "ASP" and at in ("OD1", "OD2"):
            salt_neg.setdefault(ch, []).append(xyz)

    ch_ids = sorted(ca_by_chain.keys())
    for c in ch_ids:
        ca_by_chain[c].sort(key=lambda x: x[0])

    hel_scores = []
    for c in ch_ids:
        coords = [x[1] for x in ca_by_chain[c]]
        if len(coords) >= 5:
            ok = sum(4.5 <= _dist(coords[i], coords[i + 3]) <= 5.9 for i in range(len(coords) - 3))
            hel_scores.append(ok / (len(coords) - 3))
    helix_frac = float(np.mean(hel_scores)) if hel_scores else np.nan

    cos_ab, nn_dist, nc_dist = np.nan, np.nan, np.nan
    if len(ch_ids) >= 2:
        cA = [x[1] for x in ca_by_chain[ch_ids[0]]]
        cB = [x[1] for x in ca_by_chain[ch_ids[1]]]
        vA, vB = cA[-1] - cA[0], cB[-1] - cB[0]
        cos_ab = float(np.dot(vA, vB) / (np.linalg.norm(vA) * np.linalg.norm(vB) + 1e-9))
        nn_dist = _dist(cA[0], cB[0])
        nc_dist = _dist(cA[0], cB[-1])

    min_asn_ca = np.nan
    if len(ch_ids) >= 2 and ch_ids[0] in asn_by_chain and ch_ids[1] in asn_by_chain:
        dists = [_dist(x1, x2) for _, x1 in asn_by_chain[ch_ids[0]] for _, x2 in asn_by_chain[ch_ids[1]]]
        if dists:
            min_asn_ca = float(min(dists))

    ab_contacts = 0
    third_chain_contacts = np.nan
    if len(ch_ids) >= 2:
        hA = np.array(heavy_by_chain[ch_ids[0]])
        hB = np.array(heavy_by_chain[ch_ids[1]])
        ab_contacts = int(np.sum(np.linalg.norm(hA[:, None, :] - hB[None, :, :], axis=-1) < 4.5))
    if len(ch_ids) >= 3:
        hAB = np.vstack([heavy_by_chain[ch_ids[0]], heavy_by_chain[ch_ids[1]]])
        hC = np.array(heavy_by_chain[ch_ids[2]])
        third_chain_contacts = int(np.sum(np.linalg.norm(hAB[:, None, :] - hC[None, :, :], axis=-1) < 4.5))

    salt_bridges, same_charge_close = 0, 0
    for i in range(len(ch_ids)):
        for j in range(i + 1, len(ch_ids)):
            ci, cj = ch_ids[i], ch_ids[j]
            for p_list, n_list in [
                (salt_pos.get(ci, []), salt_neg.get(cj, [])),
                (salt_pos.get(cj, []), salt_neg.get(ci, [])),
            ]:
                if p_list and n_list:
                    dm = np.linalg.norm(np.array(p_list)[:, None, :] - np.array(n_list)[None, :, :], axis=-1)
                    salt_bridges += int(np.sum(dm < 4.5))
            for s1, s2 in [
                (salt_pos.get(ci, []), salt_pos.get(cj, [])),
                (salt_neg.get(ci, []), salt_neg.get(cj, [])),
            ]:
                if s1 and s2:
                    dm = np.linalg.norm(np.array(s1)[:, None, :] - np.array(s2)[None, :, :], axis=-1)
                    same_charge_close += int(np.sum(dm < 5.5))

    return {
        "n_chains": len(ch_ids),
        "chain_lens": [len(ca_by_chain[c]) for c in ch_ids],
        "helix_frac": helix_frac,
        "cos_AB": cos_ab,
        "dist_NN": nn_dist,
        "dist_NC": nc_dist,
        "min_Asn_CA": min_asn_ca,
        "AB_contacts_45A": ab_contacts,
        "C_to_AB_contacts": third_chain_contacts,
        "salt_bridges": salt_bridges,
        "same_chg_close": same_charge_close,
    }
