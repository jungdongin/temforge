#!/usr/bin/env python3
"""
Download a crystal structure from the Materials Project and save CIF + POSCAR.

Usage:
  python fetch_structure.py --mp-id mp-81 --output-dir structures/
"""

import argparse
import os

from mp_api.client import MPRester
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import write


def fetch_structure(mp_id, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with MPRester() as mpr:
        doc = mpr.summary.get_data_by_id(
            mp_id,
            fields=["material_id", "formula_pretty", "structure", "symmetry"],
        )

        if doc is None:
            raise ValueError(f"No structure found for material ID: {mp_id}")

        print(f"Got material: {doc.material_id} {doc.formula_pretty}")

        ase_atoms = AseAtomsAdaptor.get_atoms(doc.structure)
        print(f"Chemical formula: {ase_atoms.get_chemical_formula()}")
        print(f"Cell:\n{ase_atoms.get_cell()}")

        formula = doc.formula_pretty.replace(" ", "")
        sg_symbol = getattr(doc.symmetry, "symbol", None)
        if sg_symbol is None:
            base_name = formula
        else:
            sg_symbol_safe = sg_symbol.replace(" ", "").replace("/", "_")
            base_name = f"{formula}_{sg_symbol_safe}"

        cif_path = os.path.join(output_dir, f"{base_name}.cif")
        poscar_path = os.path.join(output_dir, f"{base_name}.POSCAR")

        write(cif_path, ase_atoms, format="cif")
        print(f"Wrote {cif_path}")

        write(poscar_path, ase_atoms, format="vasp")
        print(f"Wrote {poscar_path}")

        return ase_atoms


def main():
    ap = argparse.ArgumentParser(
        description="Download crystal structure from Materials Project"
    )
    ap.add_argument("--mp-id", required=True, help="Materials Project ID (e.g., mp-81)")
    ap.add_argument(
        "--output-dir",
        default="structures/",
        help="Output directory (default: structures/)",
    )
    args = ap.parse_args()

    fetch_structure(args.mp_id, args.output_dir)


if __name__ == "__main__":
    main()
