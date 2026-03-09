#!/usr/bin/env python3
"""
Create a variant of an existing supercell by shuffling species outside the ROI.

Reads original from {data_root}/{id5}/{id5}_structure_unrelaxed.cif
Writes variant to {data_root_var}/{id5}/{id5}_var_structure_unrelaxed.cif
              and {data_root_var}/{id5}/{id5}_var_meta.json
"""

import argparse
import json
import os
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

import yaml
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter


def nm_to_A(x):
    return float(x) * 10.0


def atomic_write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def frac_wrap_half(frac):
    return frac - np.round(frac)


def roi_mask_centered(struct, roi_nm_xyz, center_frac=(0.5, 0.5, 0.5)):
    roi_A = np.array([nm_to_A(x) for x in roi_nm_xyz], dtype=float)
    half_A = 0.5 * roi_A

    f = np.array(struct.frac_coords, dtype=float)
    c = np.array(center_frac, dtype=float)
    df = frac_wrap_half(f - c)
    dcart = struct.lattice.get_cartesian_coords(df)

    m = (
        (np.abs(dcart[:, 0]) <= half_A[0])
        & (np.abs(dcart[:, 1]) <= half_A[1])
        & (np.abs(dcart[:, 2]) <= half_A[2])
    )
    return m


def shuffle_species_outside_roi(struct, roi_mask, seed):
    rng = np.random.default_rng(seed)

    n = len(struct)
    roi_idx = np.where(roi_mask)[0]
    out_idx = np.where(~roi_mask)[0]

    out_species = [str(struct[i].specie) for i in out_idx]
    out_species_arr = np.array(out_species, dtype=object)

    uniq, cnt = np.unique(out_species_arr, return_counts=True)
    comp_before = {str(u): int(c) for u, c in zip(uniq, cnt)}

    perm = rng.permutation(len(out_idx))
    out_species_perm = out_species_arr[perm]

    out_struct = struct.copy()
    for k, site_i in enumerate(out_idx):
        sp = out_species_perm[k]
        out_struct[site_i].species = {sp: 1.0}

    out_species_after = np.array(
        [str(out_struct[i].specie) for i in out_idx], dtype=object
    )
    uniq2, cnt2 = np.unique(out_species_after, return_counts=True)
    comp_after = {str(u): int(c) for u, c in zip(uniq2, cnt2)}

    if comp_before != comp_after:
        raise RuntimeError(
            f"Outside composition changed! before={comp_before}, after={comp_after}"
        )

    info = {
        "num_atoms_total": int(n),
        "num_atoms_roi": int(len(roi_idx)),
        "num_atoms_outside": int(len(out_idx)),
        "outside_composition_before": comp_before,
        "outside_composition_after": comp_after,
        "shuffle_seed": int(seed),
    }
    return {"structure": out_struct, "info": info}


def composition_counts(struct):
    elems = [str(site.specie) for site in struct.sites]
    uniq, cnt = np.unique(np.array(elems, dtype=object), return_counts=True)
    return {str(u): int(c) for u, c in zip(uniq, cnt)}


def run(cfg, idx):
    """Run variant generation for one sample index."""
    project_cfg = cfg["project"]
    variant_cfg = cfg["variant"]

    id5 = f"{idx:05d}"

    in_dir = Path(project_cfg["data_root"]) / id5
    out_dir = Path(project_cfg["data_root_var"]) / id5
    out_dir.mkdir(parents=True, exist_ok=True)

    cif_in = in_dir / f"{id5}_structure_unrelaxed.cif"
    meta_in = in_dir / f"{id5}_meta.json"

    cif_out = out_dir / f"{id5}_var_structure_unrelaxed.cif"
    meta_out = out_dir / f"{id5}_var_meta.json"

    if not cif_in.exists():
        raise FileNotFoundError(f"Missing input CIF: {cif_in}")
    if not meta_in.exists():
        raise FileNotFoundError(f"Missing input meta: {meta_in}")

    s = Structure.from_file(str(cif_in))

    roi_nm = tuple(float(x) for x in variant_cfg["roi_nm"])
    mask = roi_mask_centered(s, roi_nm_xyz=roi_nm, center_frac=(0.5, 0.5, 0.5))

    n_roi = int(np.sum(mask))
    if n_roi <= 0:
        raise RuntimeError("ROI mask selected 0 atoms.")

    seed = cfg["samples"]["base_seed"] + idx
    result = shuffle_species_outside_roi(s, roi_mask=mask, seed=seed)
    s_var = result["structure"]
    shuffle_info = result["info"]

    # Verify ROI unchanged
    cart0 = np.array(s.cart_coords, dtype=float)
    cart1 = np.array(s_var.cart_coords, dtype=float)
    if not np.allclose(cart0[mask], cart1[mask], atol=1e-8):
        raise RuntimeError("ROI cart_coords changed unexpectedly.")

    sp0 = np.array([str(s[i].specie) for i in range(len(s))], dtype=object)
    sp1 = np.array([str(s_var[i].specie) for i in range(len(s_var))], dtype=object)
    if not np.array_equal(sp0[mask], sp1[mask]):
        raise RuntimeError("ROI species changed unexpectedly.")

    comp0 = composition_counts(s)
    comp1 = composition_counts(s_var)
    if comp0 != comp1:
        raise RuntimeError(f"Total composition changed! in={comp0}, out={comp1}")

    CifWriter(s_var).write_file(str(cif_out))

    # Build variant meta from original meta
    meta_orig = json.loads(meta_in.read_text())
    meta_var = dict(meta_orig)

    meta_var["var_pipeline"] = {
        "status": "done",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": str(project_cfg["data_root"]),
        "output_root": str(project_cfg["data_root_var"]),
        "input_cif": cif_in.name,
        "output_cif": cif_out.name,
        "roi_nm_centered": list(roi_nm),
        "roi_center_frac": [0.5, 0.5, 0.5],
        "roi_num_atoms": int(n_roi),
        "shuffle_mode": "species_permutation_outside_roi",
        "shuffle_seed": int(seed),
        "composition_total_in": comp0,
        "composition_total_out": comp1,
        **shuffle_info,
    }

    atomic_write_json(meta_out, meta_var)
    print(f"[variant] {id5} | ROI atoms={n_roi} | seed={seed}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--idx", type=int, required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run(cfg, args.idx)


if __name__ == "__main__":
    main()
