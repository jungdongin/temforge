#!/bin/bash
#SBATCH -J temforge
#SBATCH -A m3828
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -t 03:00:00
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH -o /pscratch/sd/d/dongin/temforge/logs/%x_%A_%a.out
#SBATCH -e /pscratch/sd/d/dongin/temforge/logs/%x_%A_%a.err

# =============================================================================
# TEMForge Pipeline - SLURM Batch Script
#
# Usage:
#   bash run_pipeline.sh                  # self-submits with defaults below
#   bash run_pipeline.sh --start 1 --end 100
#   sbatch --array=1-100 run_pipeline.sh  # manual submission
# =============================================================================

set -eo pipefail

# ================== EDIT THESE ==================
START=${START:-3024}
END=${END:-3024}
CONFIG=${CONFIG:-config/default.yaml}
THROTTLE=${THROTTLE:-500}

CONDA_ENV_ABTEM="abtem"
CONDA_ENV_LAMMPS="lammps"
# ================================================

# Project root — resolved on login node, exported to compute nodes
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  TEMFORGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
# TEMFORGE_ROOT is exported via --export=ALL during sbatch
cd "${TEMFORGE_ROOT}"

# ---------- self-submit mode (login node) ----------
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  mkdir -p logs
  # Parse optional CLI args
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --start) START="$2"; shift 2 ;;
      --end)   END="$2";   shift 2 ;;
      --config) CONFIG="$2"; shift 2 ;;
      --throttle) THROTTLE="$2"; shift 2 ;;
      *) echo "[ERROR] Unknown arg: $1"; exit 2 ;;
    esac
  done

  if [[ "${END}" -lt "${START}" ]]; then
    echo "[ERROR] END < START (${END} < ${START})"
    exit 2
  fi

  total=$(( END - START + 1 ))
  echo "[TEMForge] Submitting ${total} tasks (${START}-${END}%${THROTTLE})"
  echo "[TEMForge] Config: ${CONFIG}"

  sbatch \
    --array="${START}-${END}%${THROTTLE}" \
    --export=ALL,TEMFORGE_ROOT="${TEMFORGE_ROOT}",CONFIG="${CONFIG}",CONDA_ENV_ABTEM="${CONDA_ENV_ABTEM}",CONDA_ENV_LAMMPS="${CONDA_ENV_LAMMPS}" \
    "${BASH_SOURCE[0]}"
  exit 0
fi

# ---------- running under Slurm ----------
IDX="${SLURM_ARRAY_TASK_ID}"
echo "=========================================="
echo "[TEMForge] ID=${IDX} | JobID=${SLURM_JOB_ID} | $(date)"
echo "=========================================="

# Initialize conda (disable -eu; conda scripts use unbound vars and may return non-zero)
set +eu
if [[ -f /global/common/software/nersc/pe/conda/24.10.0/Miniforge3-24.7.1-0/etc/profile.d/conda.sh ]]; then
  source /global/common/software/nersc/pe/conda/24.10.0/Miniforge3-24.7.1-0/etc/profile.d/conda.sh
else
  module load python 2>/dev/null || true
fi

# Convert YAML config to JSON (abtem env has pyyaml; lammps env may not)
CONFIG_JSON="/tmp/temforge_config_${SLURM_JOB_ID}_${IDX}.json"

# --- Stage 1: Generate supercell + variant (abtem env) ---
echo "[TEMForge] Stage: generate (env=${CONDA_ENV_ABTEM})"
conda activate "${CONDA_ENV_ABTEM}"
set -eo pipefail
python -c "import yaml, json, sys; json.dump(yaml.safe_load(open(sys.argv[1])), open(sys.argv[2], 'w'))" \
  "${TEMFORGE_ROOT}/${CONFIG}" "${CONFIG_JSON}"
python "${TEMFORGE_ROOT}/run_pipeline.py" --config "${CONFIG_JSON}" --idx "${IDX}" --stages generate

# --- Stage 2: LAMMPS relaxation (lammps env) ---
echo "[TEMForge] Stage: relax (env=${CONDA_ENV_LAMMPS})"
set +eu
conda deactivate
conda activate "${CONDA_ENV_LAMMPS}"
set -eo pipefail
python "${TEMFORGE_ROOT}/run_pipeline.py" --config "${CONFIG_JSON}" --idx "${IDX}" --stages relax

# --- Stage 3: ROI extraction + DP simulation (abtem env) ---
echo "[TEMForge] Stage: post_relax (env=${CONDA_ENV_ABTEM})"
set +eu
conda deactivate
conda activate "${CONDA_ENV_ABTEM}"
set -eo pipefail
python "${TEMFORGE_ROOT}/run_pipeline.py" --config "${CONFIG_JSON}" --idx "${IDX}" --stages post_relax

# Clean up temp config
rm -f "${CONFIG_JSON}"

echo "=========================================="
echo "[TEMForge] DONE ID=${IDX} | $(date)"
echo "=========================================="
