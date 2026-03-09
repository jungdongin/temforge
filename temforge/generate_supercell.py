#!/usr/bin/env python3
"""
Generate a random Cu/Au supercell with randomly assigned species.

Reads configuration from YAML. Produces:
  {data_root}/{id5}/{id5}_structure_unrelaxed.cif
  {data_root}/{id5}/{id5}_meta.json
"""

import argparse
import json
import os
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

import yaml
from pymatgen.core import Structure, Lattice
from pymatgen.io.cif import CifWriter


def nm_to_A(x):
    return x * 10.0


def scale_lattice(lattice, target_lengths_A):
    mat = lattice.matrix
    new_mat = np.zeros_like(mat)
    for i in range(3):
        v = mat[i]
        new_mat[i] = v / np.linalg.norm(v) * target_lengths_A[i]
    return Lattice(new_mat)


def embed_center(struct, sim_cell_nm, wrap=True):
    target_A = [nm_to_A(x) for x in sim_cell_nm]
    new_lat = scale_lattice(struct.lattice, target_A)

    cart = struct.cart_coords
    atom_center = 0.5 * (cart.min(axis=0) + cart.max(axis=0))
    new_center = 0.5 * np.sum(new_lat.matrix, axis=0)
    shifted = cart + (new_center - atom_center)

    out = Structure(
        lattice=new_lat,
        species=struct.species,
        coords=shifted,
        coords_are_cartesian=True,
    )

    if wrap:
        out.translate_sites(
            range(len(out)), [0, 0, 0],
            frac_coords=False, to_unit_cell=True,
        )
    return out


def random_assign(struct, seed, species_A, species_B):
    rng = np.random.default_rng(seed)
    frac_A = rng.uniform(0.0, 1.0)

    n = len(struct)
    nA = int(round(frac_A * n))
    idxA = set(rng.choice(n, size=nA, replace=False))

    out = struct.copy()
    for i in range(n):
        out[i].species = {species_A: 1.0} if i in idxA else {species_B: 1.0}

    return out, frac_A, nA


def run(cfg, idx):
    """Run supercell generation for one sample index."""
    struct_cfg = cfg["structure"]
    project_cfg = cfg["project"]

    # Resolve base CIF path relative to project root
    base_cif_path = struct_cfg["base_cif"]
    if not os.path.isabs(base_cif_path):
        project_root = os.environ.get(
            "TEMFORGE_ROOT",
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        base_cif_path = os.path.join(project_root, base_cif_path)

    base = Structure.from_file(base_cif_path)
    supercell = tuple(struct_cfg["supercell"])
    base.make_supercell(supercell)

    id5 = f"{idx:05d}"
    data_root = project_cfg["data_root"]
    out_dir = Path(data_root) / id5
    out_dir.mkdir(parents=True, exist_ok=True)

    out_cif = out_dir / f"{id5}_structure_unrelaxed.cif"
    meta_path = out_dir / f"{id5}_meta.json"

    seed = cfg["samples"]["base_seed"] + idx

    s_rand, frac_draw, nA = random_assign(
        base, seed,
        struct_cfg["species_a"],
        struct_cfg["species_b"],
    )

    sim_cell_nm = tuple(struct_cfg["sim_cell_nm"])
    s_final = embed_center(s_rand, sim_cell_nm)
    CifWriter(s_final).write_file(str(out_cif))

    n = len(s_final)
    frac_actual = nA / n

    meta = {
        "id": id5,
        "seed": seed,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "supercell": list(supercell),
        "sim_cell_nm": list(sim_cell_nm),
        "species_A": struct_cfg["species_a"],
        "species_B": struct_cfg["species_b"],
        "a_frac_random_draw": float(frac_draw),
        "a_frac_actual": float(frac_actual),
        "num_atoms_total": int(n),
        "num_atoms_A": int(nA),
        "cif_filename": out_cif.name,
    }

    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[generate] {id5} | {struct_cfg['species_a']} frac={frac_actual:.4f} | seed={seed}")


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
