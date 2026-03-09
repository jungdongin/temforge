#!/usr/bin/env python3
"""
TEMForge pipeline orchestrator.

Called by run_pipeline.sh with --stages to group work by conda environment:
  --stages generate    : generate_supercell + generate_variant  (abtem env)
  --stages relax       : relax                                  (lammps env)
  --stages post_relax  : extract_roi + simulate_dp              (abtem env)

Reads the YAML config to determine which pipelines (original/variant) are
enabled and calls the appropriate stage scripts.
"""

import argparse
import sys
import os

import yaml


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def stage_generate(cfg, idx):
    """Run generate_supercell (if original enabled) + generate_variant (if variant enabled)."""
    orig_enabled = cfg["pipeline"]["original"]["enabled"]
    var_enabled = cfg["pipeline"]["variant"]["enabled"]

    if orig_enabled:
        from temforge.generate_supercell import run as run_generate
        run_generate(cfg, idx)

    if var_enabled:
        from temforge.generate_variant import run as run_variant
        run_variant(cfg, idx)


def stage_relax(cfg, idx):
    """Run relaxation for enabled pipelines."""
    from temforge.relax import run as run_relax

    orig_enabled = cfg["pipeline"]["original"]["enabled"]
    var_enabled = cfg["pipeline"]["variant"]["enabled"]

    if orig_enabled:
        run_relax(cfg, idx, "original")
    if var_enabled:
        run_relax(cfg, idx, "variant")


def stage_post_relax(cfg, idx):
    """Run extract_roi + simulate_dp for enabled pipelines."""
    from temforge.extract_roi import run as run_roi
    from temforge.simulate_dp import run as run_dp

    orig_enabled = cfg["pipeline"]["original"]["enabled"]
    var_enabled = cfg["pipeline"]["variant"]["enabled"]

    if orig_enabled:
        run_roi(cfg, idx, "original")
        run_dp(cfg, idx, "original")
    if var_enabled:
        run_roi(cfg, idx, "variant")
        run_dp(cfg, idx, "variant")


STAGE_MAP = {
    "generate": stage_generate,
    "relax": stage_relax,
    "post_relax": stage_post_relax,
}


def main():
    ap = argparse.ArgumentParser(description="TEMForge pipeline orchestrator")
    ap.add_argument("--config", required=True, help="Path to YAML config file")
    ap.add_argument("--idx", type=int, required=True, help="Sample index")
    ap.add_argument(
        "--stages",
        required=True,
        choices=list(STAGE_MAP.keys()),
        help="Stage group to run",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)

    # Set TEMFORGE_ROOT so scripts can resolve relative paths
    project_root = os.path.dirname(os.path.abspath(args.config))
    # If config is in config/ subdirectory, go up one level
    if os.path.basename(project_root) == "config":
        project_root = os.path.dirname(project_root)
    os.environ["TEMFORGE_ROOT"] = project_root

    # Add project root to Python path so temforge package is importable
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    stage_fn = STAGE_MAP[args.stages]
    stage_fn(cfg, args.idx)


if __name__ == "__main__":
    main()
