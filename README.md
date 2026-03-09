# TEMForge

Automated pipeline for generating simulated electron diffraction pattern (DP) training data from random Cu/Au alloy supercells on NERSC Perlmutter.

## Pipeline Stages

1. **Generate Supercell** — Create random Cu/Au FCC supercells from a base CIF, with randomly assigned species fractions
2. **Generate Variant** — Shuffle species labels outside a central ROI while preserving the ROI structure
3. **Relax** — LAMMPS energy minimization using a KIM EAM potential
4. **Extract ROI** — Crop the relaxed structure to a centered cuboid ROI
5. **Simulate DP** — abTEM multislice electron diffraction pattern tilt series

## Quick Start

```bash
# 1. Fetch a base crystal structure (requires MP API key)
python temforge/fetch_structure.py --mp-id mp-81 --output-dir structures/

# 2. Edit config
vim config/default.yaml

# 3. Submit the full pipeline
bash run_pipeline.sh --start 1 --end 4000
```

## Configuration

All physics, paths, and pipeline toggles are in `config/default.yaml`. SLURM settings (queue, time, nodes) stay in `run_pipeline.sh`.

### Pipeline Toggles

```yaml
pipeline:
  original:
    enabled: true    # Generate + relax + ROI + DP on original structures
  variant:
    enabled: true    # Generate variant + relax + ROI + DP on shuffled structures
```

| original | variant | Behavior |
|----------|---------|----------|
| true | true | Full pipeline for both original and variant |
| true | false | Original only (no variant generation) |
| false | true | Skip generate (assumes originals exist), create variant → relax → ROI → DP |
| false | false | Nothing to do |

### Key Config Sections

- **`project`** — Data output paths (`data_root`, `data_root_var`, `work_root`)
- **`structure`** — Base CIF, supercell size, simulation cell, species
- **`variant`** — ROI box size for variant generation
- **`relax`** — KIM model, minimization settings
- **`roi`** — ROI extraction box size, minimum atom count
- **`dp`** — Tilt series, beam energy, convergence angles, abTEM simulation parameters

## Project Structure

```
temforge/
├── config/
│   └── default.yaml              # All parameters
├── structures/                   # Base crystal structure files (CIF, POSCAR)
├── potentials/                   # Interatomic potential files
├── temforge/
│   ├── fetch_structure.py        # Download CIF from Materials Project
│   ├── generate_supercell.py     # Random Cu/Au supercell
│   ├── generate_variant.py       # ROI-fixed variant
│   ├── relax.py                  # LAMMPS relaxation
│   ├── extract_roi.py            # ROI extraction
│   └── simulate_dp.py            # abTEM DP simulation
├── run_pipeline.py               # Orchestrator (called by run_pipeline.sh)
├── run_pipeline.sh               # SLURM batch script
└── logs/                         # SLURM logs
```

### Data Output

```
{data_root}/{id5}/                 # Original pipeline
├── {id5}_structure_unrelaxed.cif
├── {id5}_structure.cif
├── {id5}_structure_roi.cif
├── {id5}_dp_convAngle_*/
└── {id5}_meta.json

{data_root_var}/{id5}/             # Variant pipeline
├── {id5}_var_structure_unrelaxed.cif
├── {id5}_var_structure.cif
├── {id5}_var_structure_roi.cif
├── {id5}_var_dp_convAngle_*/
└── {id5}_var_meta.json
```

## Requirements

Two conda environments are needed:

- **abtem** — `pip install -r requirements_abtem.txt`
- **lammps** — `pip install -r requirements_lammps.txt`

The `run_pipeline.sh` script switches between them automatically.

## Usage Examples

```bash
# Run samples 1-100
bash run_pipeline.sh --start 1 --end 100

# Run with custom config
bash run_pipeline.sh --config config/my_config.yaml --start 1 --end 500

# Limit concurrent SLURM tasks
bash run_pipeline.sh --start 1 --end 4000 --throttle 200

# Fetch a new base structure
python temforge/fetch_structure.py --mp-id mp-5229 --output-dir structures/
```
