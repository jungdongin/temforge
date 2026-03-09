#!/usr/bin/env python3
"""
Simulate electron diffraction pattern tilt series using abTEM.

For original pipeline:
  Input:  {data_root}/{id5}/{id5}_structure.cif
  Output: {data_root}/{id5}/{id5}_dp_convAngle_*/
  Meta:   {data_root}/{id5}/{id5}_meta.json

For variant pipeline:
  Input:  {data_root_var}/{id5}/{id5}_var_structure.cif
  Output: {data_root_var}/{id5}/{id5}_var_dp_convAngle_*/
  Meta:   {data_root_var}/{id5}/{id5}_var_meta.json
"""

import argparse
import json
import os
import shutil
import numpy as np
from pathlib import Path

import yaml
from ase.io import read

import abtem
from abtem import Probe, Potential
from abtem.scan import GridScan
from abtem.inelastic.phonons import FrozenPhonons

import zarr
from numcodecs import Blosc


def build_simulation_cell(atoms_in, L_sim_xy_nm, L_sim_z_nm):
    atoms = atoms_in.copy()
    cell = np.zeros((3, 3))
    cell[0, 0] = L_sim_xy_nm * 10
    cell[1, 1] = L_sim_xy_nm * 10
    cell[2, 2] = L_sim_z_nm * 10
    atoms.set_cell(cell, scale_atoms=False)
    atoms.center()
    return atoms


def apply_tilt(atoms, tilt_deg, axis):
    atoms = atoms.copy()
    center = atoms.cell.sum(axis=0) / 2
    atoms.rotate(tilt_deg, axis, center=center, rotate_cell=False)
    return atoms


def simulate_dp_onepos(atoms, tilt_deg, dp_cfg, semiangle_mrad):
    atoms_t = apply_tilt(atoms, tilt_deg, dp_cfg["tilt_axis"])

    frozen = FrozenPhonons(
        atoms_t,
        num_configs=dp_cfg["n_frozen"],
        sigmas=dp_cfg["sigmas_A"],
        seed=dp_cfg["seed"],
    )

    pot = Potential(
        frozen,
        sampling=dp_cfg["potential_sampling_A"],
        slice_thickness=dp_cfg["slice_thickness_A"],
        parametrization=dp_cfg["parametrization"],
    )

    probe = Probe(
        energy=dp_cfg["energy_eV"],
        semiangle_cutoff=semiangle_mrad,
        sampling=dp_cfg["probe_sampling_A"],
    )

    Lx, Ly = pot.extent
    cx, cy = Lx / 2, Ly / 2
    scan = GridScan(
        start=(cx - 1e-3, cy - 1e-3),
        end=(cx + 1e-3, cy + 1e-3),
        gpts=(1, 1),
    )

    waves = probe.multislice(pot, scan=scan)
    dp = waves.diffraction_patterns(max_angle=dp_cfg["max_angle_mrad"])
    arr = np.asarray(dp.array, dtype=np.float32)
    return arr.mean(axis=tuple(range(arr.ndim - 2)))


def run(cfg, idx, pipeline):
    """Run DP simulation for one sample on the specified pipeline."""
    project_cfg = cfg["project"]
    dp_cfg = cfg["dp"]
    id5 = f"{idx:05d}"

    if pipeline == "original":
        folder = Path(project_cfg["data_root"]) / id5
        cif_relaxed = folder / f"{id5}_structure.cif"
        meta_path = folder / f"{id5}_meta.json"
        prefix = id5
    else:
        folder = Path(project_cfg["data_root_var"]) / id5
        cif_relaxed = folder / f"{id5}_var_structure.cif"
        meta_path = folder / f"{id5}_var_meta.json"
        prefix = f"{id5}_var"

    if not cif_relaxed.exists():
        raise FileNotFoundError(f"Missing relaxed CIF: {cif_relaxed}")

    atoms0 = read(str(cif_relaxed))
    atoms_sim = build_simulation_cell(
        atoms0, dp_cfg["L_sim_xy_nm"], dp_cfg["L_sim_z_nm"]
    )

    tilts = np.arange(
        dp_cfg["tilt_min_deg"],
        dp_cfg["tilt_max_deg"] + 1e-9,
        dp_cfg["tilt_step_deg"],
    )

    conv_angles = dp_cfg["conv_angles_mrad"]
    tag_map = dp_cfg["conv_angle_tags"]
    overwrite = dp_cfg["overwrite"]

    # Load meta
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        meta = {"id": id5}

    meta.setdefault("dp_settings", {})
    meta["dp_settings"]["structure_source"] = cif_relaxed.name

    for ca in conv_angles:
        dp_list = []
        for t in tilts:
            dp_list.append(simulate_dp_onepos(atoms_sim, t, dp_cfg, ca))
        stack = np.stack(dp_list)

        tag = tag_map[ca]
        out_store = str(folder / f"{prefix}_dp_convAngle_{tag}")

        if os.path.exists(out_store) and overwrite:
            shutil.rmtree(out_store)

        store = zarr.DirectoryStore(out_store)
        root = zarr.group(store=store, overwrite=True)
        root.create_dataset(
            dp_cfg["zarr_dataset_name"],
            data=stack,
            chunks=(1, *stack.shape[1:]),
            dtype=np.float32,
            compressor=Blosc(cname="zstd", clevel=3),
        )

        meta.setdefault("files", {})
        meta["files"][f"dp_convAngle_{tag}"] = {
            "path": os.path.basename(out_store),
            "shape": list(stack.shape),
            "dtype": "float32",
        }

    # Save meta
    tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
    tmp.write_text(json.dumps(meta, indent=2) + "\n")
    os.replace(tmp, meta_path)

    print(f"[dp] {id5} | {len(tilts)} tilts x {len(conv_angles)} angles")


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
