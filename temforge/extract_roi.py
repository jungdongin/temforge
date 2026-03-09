#!/usr/bin/env python3
"""
Extract a centered cuboid ROI from an input CIF.

For original pipeline:
  Input:  {data_root}/{id5}/{id5}_structure.cif
  Output: {data_root}/{id5}/{id5}_structure_roi.cif

For variant pipeline:
  Input:  {data_root_var}/{id5}/{id5}_var_structure.cif
  Output: {data_root_var}/{id5}/{id5}_var_structure_roi.cif
"""

import argparse
import json
import os
import numpy as np
from pathlib import Path

import yaml
from pymatgen.core import Structure, Lattice


def wrap01(x):
    return x % 1.0


def pbc_dist(d):
    d = d % 1.0
    if d > 0.5:
        d -= 1.0
    return d


def extract_roi(in_path, out_path, roi_nm, min_atoms):
    """Extract ROI and write output CIF. Returns number of atoms kept."""
    s = Structure.from_file(str(in_path))
    lat = s.lattice
    a_len, b_len, c_len = lat.abc

    roi_A = np.array(roi_nm, dtype=float) * 10.0
    dfrac = roi_A / np.array([a_len, b_len, c_len], dtype=float)

    center = np.array([0.5, 0.5, 0.5], dtype=float)
    frac = np.array([site.frac_coords for site in s.sites], dtype=float)
    frac_wrapped = np.mod(frac, 1.0)

    keep_idx = []
    for i, f in enumerate(frac_wrapped):
        df = np.array([pbc_dist(f[j] - center[j]) for j in range(3)], dtype=float)
        if all(abs(df[j]) <= dfrac[j] / 2.0 for j in range(3)):
            keep_idx.append(i)

    if len(keep_idx) < min_atoms:
        raise RuntimeError(
            f"Extracted {len(keep_idx)} atoms < min_atoms={min_atoms} for {in_path}"
        )

    lo = center - dfrac / 2.0

    new_species = []
    new_fcoords = []
    for i in keep_idx:
        f = frac_wrapped[i].copy()
        f_rel = (f - lo) / dfrac
        f_rel = np.array([wrap01(x) for x in f_rel], dtype=float)
        new_species.append(s[i].specie)
        new_fcoords.append(f_rel)

    new_fcoords = np.array(new_fcoords, dtype=float)

    amat, bmat, cmat = lat.matrix
    a_hat = amat / np.linalg.norm(amat)
    b_hat = bmat / np.linalg.norm(bmat)
    c_hat = cmat / np.linalg.norm(cmat)

    new_a = a_hat * roi_A[0]
    new_b = b_hat * roi_A[1]
    new_c = c_hat * roi_A[2]
    new_lat = Lattice(np.vstack([new_a, new_b, new_c]))

    s_roi = Structure(new_lat, new_species, new_fcoords, coords_are_cartesian=False)
    s_roi.to(filename=str(out_path))

    return len(keep_idx)


def run(cfg, idx, pipeline):
    """Run ROI extraction for one sample on the specified pipeline."""
    project_cfg = cfg["project"]
    roi_cfg = cfg["roi"]
    id5 = f"{idx:05d}"

    roi_nm = [float(x) for x in roi_cfg["roi_nm"]]
    min_atoms = int(roi_cfg["min_atoms"])

    if pipeline == "original":
        folder = Path(project_cfg["data_root"]) / id5
        cif_in = folder / f"{id5}_structure.cif"
        cif_out = folder / f"{id5}_structure_roi.cif"
        meta_path = folder / f"{id5}_meta.json"
    else:
        folder = Path(project_cfg["data_root_var"]) / id5
        cif_in = folder / f"{id5}_var_structure.cif"
        cif_out = folder / f"{id5}_var_structure_roi.cif"
        meta_path = folder / f"{id5}_var_meta.json"

    if not cif_in.exists():
        raise FileNotFoundError(f"Missing relaxed CIF: {cif_in}")

    n_kept = extract_roi(cif_in, cif_out, roi_nm, min_atoms)

    # Update meta
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        meta = {"id": id5}

    meta["roi_pipeline"] = {
        "status": "done",
        "roi_nm": roi_nm,
        "min_atoms": min_atoms,
        "input_cif": cif_in.name,
        "output_cif": cif_out.name,
        "num_atoms_roi": n_kept,
    }

    tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
    tmp.write_text(json.dumps(meta, indent=2) + "\n")
    os.replace(tmp, meta_path)

    print(f"[roi] {id5} | kept {n_kept} atoms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--idx", type=int, required=True)
    ap.add_argument("--pipeline", required=True, choices=["original", "variant"])
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run(cfg, args.idx, args.pipeline)


if __name__ == "__main__":
    main()
