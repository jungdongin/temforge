#!/usr/bin/env python3
"""
LAMMPS energy minimization using a KIM interatomic potential.

For original pipeline:
  Input:  {data_root}/{id5}/{id5}_structure_unrelaxed.cif
  Output: {data_root}/{id5}/{id5}_structure.cif
  Meta:   {data_root}/{id5}/{id5}_meta.json

For variant pipeline:
  Input:  {data_root_var}/{id5}/{id5}_var_structure_unrelaxed.cif
  Output: {data_root_var}/{id5}/{id5}_var_structure.cif
  Meta:   {data_root_var}/{id5}/{id5}_var_meta.json
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import List

from ase.io import read, write
from ase.io.lammpsdata import write_lammps_data


MASS = {"Cu": 63.546, "Au": 196.96657}
ALLOWED = set(MASS.keys())
BASE_SPECORDER = ["Cu", "Au"]
LAMMPS_UNITS = "metal"


def atomic_write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def load_json(path):
    return json.loads(Path(path).read_text())


def tail_text(path, n_lines=120):
    path = Path(path)
    if not path.exists():
        return ""
    lines = path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-n_lines:])


def resolve_potential_dir(relax_cfg):
    """Resolve potential_dir from config (relative to TEMFORGE_ROOT or absolute)."""
    potential_dir = relax_cfg.get("potential_dir", "")
    if not potential_dir:
        return None
    if not os.path.isabs(potential_dir):
        project_root = os.environ.get(
            "TEMFORGE_ROOT",
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        potential_dir = os.path.join(project_root, potential_dir)
    return Path(potential_dir).resolve()


def find_kim_model(kim_model, potential_dir=None):
    """Find KIM model .so file, checking project potentials/ first, then ~/.kim-api/."""
    # Check project potentials/ directory first
    if potential_dir:
        project_so = potential_dir / kim_model / "libkim-api-portable-model.so"
        if project_so.exists():
            return str(project_so)

    # Fall back to ~/.kim-api/
    pattern = str(
        Path.home()
        / ".kim-api"
        / "*"
        / "portable-models-dir"
        / kim_model
        / "libkim-api-portable-model.so"
    )
    hits = glob.glob(pattern)
    if hits:
        return hits[0]

    return None


def setup_kim_env(potential_dir):
    """Add project potentials/ to KIM API search path via environment variable."""
    if potential_dir and potential_dir.exists():
        existing = os.environ.get("KIM_API_USER_PORTABLE_MODELS_DIR", "")
        if existing:
            os.environ["KIM_API_USER_PORTABLE_MODELS_DIR"] = f"{potential_dir}:{existing}"
        else:
            os.environ["KIM_API_USER_PORTABLE_MODELS_DIR"] = str(potential_dir)


def choose_specorder(species):
    order = [s for s in BASE_SPECORDER if s in species]
    if not order:
        raise RuntimeError(f"No usable species. species={sorted(species)}")
    return order


def type_to_symbol_map(specorder):
    return {i + 1: el for i, el in enumerate(specorder)}


def write_lammps_input(in_path, data_name, out_data_name, specorder, relax_cfg):
    kim_model = relax_cfg["kim_model"]
    lines = [
        f"kim init {kim_model} {LAMMPS_UNITS}",
        "",
        "atom_style atomic",
        "boundary p p p",
        "neighbor 2.0 bin",
        "neigh_modify delay 0 every 1 check yes",
        "",
        f"read_data {data_name}",
        "",
        "# masses",
    ]
    for i, el in enumerate(specorder, start=1):
        lines.append(f"mass {i} {MASS[el]}")

    lines += [
        "",
        f"kim interactions {' '.join(specorder)}",
        "",
        "thermo 100",
        "thermo_style custom step pe etotal press pxx pyy pzz lx ly lz",
        "",
    ]

    if relax_cfg["relax_cell"]:
        lines += [f"fix 1 all box/relax aniso 0.0 vmax {relax_cfg['vmax']}", ""]

    lines += [
        f"min_style {relax_cfg['min_style']}",
        f"minimize {relax_cfg['etol']} {relax_cfg['ftol']} {relax_cfg['maxiter']} {relax_cfg['maxeval']}",
        "",
    ]

    if relax_cfg["relax_cell"]:
        lines += ["unfix 1", ""]

    lines += [
        f"write_data {out_data_name}",
        "write_dump all custom relaxed.dump id type x y z",
        'print "DONE"',
        "",
    ]
    in_path.write_text("\n".join(lines))


def run_lammps(cmd, cwd, potential_dir=None):
    logp = cwd / "run.log"
    env = os.environ.copy()
    env.pop("SLURM_CPU_BIND", None)
    # Ensure LAMMPS subprocess can find KIM models in potentials/
    if potential_dir and Path(potential_dir).exists():
        existing = env.get("KIM_API_USER_PORTABLE_MODELS_DIR", "")
        if existing:
            env["KIM_API_USER_PORTABLE_MODELS_DIR"] = f"{potential_dir}:{existing}"
        else:
            env["KIM_API_USER_PORTABLE_MODELS_DIR"] = str(potential_dir)

    with open(logp, "w") as f:
        f.write(f"PWD={cwd}\n")
        f.write(f"CMD={' '.join(cmd)}\n")
        f.write(f"SLURM_JOB_ID={os.environ.get('SLURM_JOB_ID', '')}\n")
        f.write(f"SLURM_NTASKS={os.environ.get('SLURM_NTASKS', '')}\n")
        f.write("\n===== OUTPUT (stdout+stderr) =====\n")
        f.flush()

        p = subprocess.run(
            cmd, cwd=str(cwd),
            stdout=f, stderr=subprocess.STDOUT,
            text=True, env=env,
        )
        f.write(f"\n===== RETURN CODE: {p.returncode} =====\n")
        f.flush()

    return int(p.returncode)


def relax_one(cif_in, cif_out, meta_path, run_dir, relax_cfg, nproc, launcher):
    """Run LAMMPS relaxation for one structure."""
    cif_in = Path(cif_in).resolve()
    cif_out = Path(cif_out).resolve()
    meta_path = Path(meta_path).resolve()
    run_dir = Path(run_dir).resolve()

    if not cif_in.exists():
        raise RuntimeError(f"missing input cif: {cif_in}")
    if not meta_path.exists():
        raise RuntimeError(f"missing meta: {meta_path}")

    overwrite = relax_cfg.get("overwrite", False)
    if cif_out.exists() and not overwrite:
        print(f"[relax] SKIP already relaxed: {cif_out}")
        return

    kim_model = relax_cfg["kim_model"]
    potential_dir = resolve_potential_dir(relax_cfg)
    so_path = find_kim_model(kim_model, potential_dir)
    if not so_path:
        search_locations = ["potentials/"]
        if potential_dir:
            search_locations = [str(potential_dir)]
        search_locations.append("~/.kim-api/*/portable-models-dir/")
        raise RuntimeError(
            f"KIM model .so not found in {search_locations}. Try:\n"
            f"  kim-api-collections-management install user {kim_model}\n"
            f"  or copy the model to potentials/{kim_model}/\n"
        )

    # Make KIM API find the model from potentials/
    setup_kim_env(potential_dir)

    atoms = read(str(cif_in))
    species = set(atoms.get_chemical_symbols())
    if not species or not species.issubset(ALLOWED):
        raise RuntimeError(f"unexpected species {sorted(species)} in {cif_in}")

    specorder = choose_specorder(species)

    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Mark running
    meta = load_json(meta_path)
    meta.setdefault("relax", {})
    meta["relax"].update({
        "status": "running",
        "kim_id": kim_model,
        "specorder_runtime": specorder,
        "relax_cell": relax_cfg["relax_cell"],
        "vmax": relax_cfg["vmax"],
        "min_style": relax_cfg["min_style"],
        "etol": relax_cfg["etol"],
        "ftol": relax_cfg["ftol"],
        "maxiter": relax_cfg["maxiter"],
        "maxeval": relax_cfg["maxeval"],
        "launcher": launcher,
        "nproc": nproc,
        "work_dir": str(run_dir),
        "cif_unrelaxed": cif_in.name,
        "cif_relaxed": cif_out.name,
    })
    atomic_write_json(meta_path, meta)

    try:
        data_path = run_dir / "structure.data"
        with open(data_path, "w") as f:
            write_lammps_data(f, atoms, specorder=specorder, atom_style="atomic")

        in_path = run_dir / "in.min"
        out_data = run_dir / "relaxed.data"
        write_lammps_input(in_path, data_path.name, out_data.name, specorder, relax_cfg)

        lmp = relax_cfg["lmp_command"]
        if launcher == "srun":
            cmd = ["srun", "--cpu-bind=none", "-n", str(nproc), lmp, "-in", in_path.name]
        else:
            cmd = ["mpirun", "-np", str(nproc), lmp, "-in", in_path.name]

        rc = run_lammps(cmd, cwd=run_dir, potential_dir=potential_dir)
        if rc != 0:
            raise RuntimeError(f"LAMMPS failed rc={rc}")

        dump_path = run_dir / "relaxed.dump"
        if not dump_path.exists():
            raise RuntimeError("missing relaxed.dump")

        relaxed = read(str(dump_path), format="lammps-dump-text")
        if "type" not in relaxed.arrays:
            raise RuntimeError("ASE did not load 'type' from relaxed.dump")

        t2s = type_to_symbol_map(specorder)
        types = relaxed.arrays["type"].astype(int)
        symbols = [t2s[int(t)] for t in types]
        relaxed.set_chemical_symbols(symbols)

        relaxed.set_cell(atoms.get_cell())
        relaxed.set_pbc(atoms.get_pbc())

        write(str(cif_out), relaxed)

        meta = load_json(meta_path)
        meta.setdefault("relax", {})
        meta["relax"].update({
            "status": "done",
            "run_log_tail": tail_text(run_dir / "run.log", 60),
        })
        atomic_write_json(meta_path, meta)

        print(f"[relax] OK wrote {cif_out}")

    except Exception as e:
        meta = load_json(meta_path)
        meta.setdefault("relax", {})
        meta["relax"].update({
            "status": "failed",
            "error": str(e),
            "run_log_tail": tail_text(run_dir / "run.log", 120),
        })
        atomic_write_json(meta_path, meta)
        raise


def run(cfg, idx, pipeline):
    """Run relaxation for one sample on the specified pipeline."""
    project_cfg = cfg["project"]
    relax_cfg = cfg["relax"]
    id5 = f"{idx:05d}"

    nproc = int(os.environ.get("SLURM_NTASKS", "1"))
    launcher = "srun" if os.environ.get("SLURM_JOB_ID") else "mpirun"

    if pipeline == "original":
        data_root = project_cfg["data_root"]
        folder = Path(data_root) / id5
        cif_in = folder / f"{id5}_structure_unrelaxed.cif"
        cif_out = folder / f"{id5}_structure.cif"
        meta_path = folder / f"{id5}_meta.json"
    else:
        data_root = project_cfg["data_root_var"]
        folder = Path(data_root) / id5
        cif_in = folder / f"{id5}_var_structure_unrelaxed.cif"
        cif_out = folder / f"{id5}_var_structure.cif"
        meta_path = folder / f"{id5}_var_meta.json"

    work_root = project_cfg["work_root"]
    run_dir = Path(work_root) / f"{pipeline}_{id5}"

    relax_one(cif_in, cif_out, meta_path, run_dir, relax_cfg, nproc, launcher)


def main():
    import yaml

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
