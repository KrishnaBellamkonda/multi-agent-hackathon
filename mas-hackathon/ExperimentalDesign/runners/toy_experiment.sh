#!/usr/bin/env bash
# SBATCH directives can remain here for cluster submission via sbatch
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name="pn"
#SBATCH --output=./slurm_stdout/slurm-pn-%j.out
#SBATCH --error=./slurm_stdout/slurm-pn-%j.err
#SBATCH --array=0-19
#SBATCH --nodelist=oat17

# Purpose: make this script usable both on a Slurm cluster and locally.
# The script detects whether it's running under Slurm (presence of SLURM_JOB_ID)
# and adapts conda activation and how it launches the Python job (srun vs direct).

set -euo pipefail

# Detect mode and scheduler
# If running under Slurm, SLURM_JOB_ID will be set. If running under LSF, LSB_JOBID will be set.
SCHEDULER="none"
if [ -n "${SLURM_JOB_ID:-}" ]; then
    SCHEDULER="slurm"
elif [ -n "${LSB_JOBID:-}" ]; then
    SCHEDULER="lsf"
else
    SCHEDULER="local"
fi

# Ensure we have an array task id to index seeds; support both SLURM and LSF ARRAY ids
# SLURM uses SLURM_ARRAY_TASK_ID; LSF may set LSB_JOBINDEX for array jobs
SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
LSF_JOB_INDEX=${LSB_JOBINDEX:-}

# Initialize FORCE_LOCAL (will be set by CLI args if --local flag is used)
FORCE_LOCAL=0

# Parse CLI args early to check for --local flag before mode detection
for _arg in "$@"; do
    case "$_arg" in
        --local|--force-local)
            FORCE_LOCAL=1
            break
            ;;
    esac
done

# Mode detection: respect FORCE_LOCAL flag, otherwise detect from scheduler
if [ "$FORCE_LOCAL" -eq 1 ]; then
    MODE="local"
    TASK_IDX=0
elif [ "$SCHEDULER" = "slurm" ]; then
    MODE="cluster"
    TASK_IDX=${SLURM_ARRAY_TASK_ID:-0}
elif [ "$SCHEDULER" = "lsf" ]; then
    MODE="cluster"
    TASK_IDX=${LSF_JOB_INDEX:-0}
else
    MODE="local"
    TASK_IDX=0
fi

# Cluster-specific activation
if [ "$SCHEDULER" = "slurm" ]; then
    # Keep original cluster conda paths when available (Slurm cluster)
    export CONDA_ENVS_PATH=/scratch-ssd/pastin/conda_envs
    export CONDA_PKGS_DIRS=/scratch-ssd/pastin/conda_pkgs
    # Preferred activation on the Slurm cluster
    if [ -f "/scratch-ssd/oatml/miniconda3/bin/activate" ]; then
        # shellcheck source=/dev/null
        source /scratch-ssd/oatml/miniconda3/bin/activate activehit
    else
        echo "Warning: expected Slurm cluster conda activation script not found; continuing without explicit activation"
    fi
elif [ "$SCHEDULER" = "lsf" ]; then
    # LSF cluster: load the module then activate the Disco environment
    if command -v module >/dev/null 2>&1; then
        # attempt to load the cond module - some LSF installations use 'module'
        module load cellgen/conda || echo "Warning: 'module load cellgen/conda' failed"
    else
        # If 'module' is not available, still try to activate if conda is present
        echo "Note: 'module' command not found; attempting to use conda directly"
    fi
    if command -v conda >/dev/null 2>&1; then
        # Initialize conda if necessary and activate Disco
        CONDA_BASE=$(conda info --base 2>/dev/null || true)
        if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
            # shellcheck source=/dev/null
            . "$CONDA_BASE/etc/profile.d/conda.sh"
            if ! conda activate Disco >/dev/null 2>&1; then
                echo "Warning: could not activate conda env 'Disco' on LSF; continuing with current python"
            fi
        else
            if ! source activate Disco >/dev/null 2>&1; then
                echo "Warning: could not activate conda env 'Disco' on LSF (no conda.sh); continuing with current python"
            fi
        fi
    else
        echo "Warning: 'conda' not found after module load; proceeding with current python on LSF"
    fi
else
    # Local: try to activate `Disco` conda env if conda is available
    if command -v conda >/dev/null 2>&1; then
        # Initialize conda for this shell session if possible
        CONDA_BASE=$(conda info --base 2>/dev/null || true)
        if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
            # shellcheck source=/dev/null
            . "$CONDA_BASE/etc/profile.d/conda.sh"
            if ! conda activate Disco >/dev/null 2>&1; then
                echo "Warning: could not activate conda env 'Disco' locally; continuing with current python"
            fi
        else
            # fallback attempt
            if ! source activate Disco >/dev/null 2>&1; then
                echo "Warning: could not activate conda env 'Disco' locally (no conda.sh); continuing with current python"
            fi
        fi
    else
        echo "Warning: 'conda' not found in PATH; proceeding with system python (local mode)"
    fi
fi

# Common environment
export PYTHONPATH=${PYTHONPATH:-}:$(pwd)
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Data location
cache_directory=./data

# Setup - fixed parameters across experiments
model_name="gp" #"bayesian_mlp"
acquisition_batch_size=20 # Size of acquisition batch at each cycle
num_active_learning_cycles=5 # Number of total acquisition cycles

# Seeds: declare as a bash array; user can edit this list
declare -a seed=(
    0
    1
    2)

# Datasets: declare as a bash array to loop over
# Manuscript-approved datasets: sine (1D), sine2d (2D), branin (2D), perturbation_simple (6D), perturbation_pathways (4D)
declare -a datasets=(
    "sine"
    "sine2d"
    "branin"
    "perturbation_simple"
    "perturbation_pathways"
)

# Create datetime-based log directory structure (will be set per dataset in the loop)
RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_BASE_DIR="./logs/toy_sweep_${RUN_TIMESTAMP}"
mkdir -p "${LOG_BASE_DIR}"

# Acquisition functions: prefer a bash array so we can pass them as multiple flags
# This allows the Python script to receive each acquisition function separately
declare -a acquisition_function=(
    greedy_kg_sum
    top_k
    thompson_sampling
    thompson_ig
    thompson_hit
    probability_hit
    random
)

### Parse CLI args for optional debug flag
PDB_MODE=0
USER_ARGS=()
for _arg in "$@"; do
    case "$_arg" in
        --pdb|-p|--debug)
            PDB_MODE=1
            ;;
        --local|--force-local)
            FORCE_LOCAL=1
            ;;
        --help|-h)
            echo "Usage: $0 [--pdb] [--local] [-- any additional args passed to python script]"
            echo "  --pdb, -p, --debug    Run the python script under pdb (local only)"
            echo "  --local, --force-local    Force local mode (run all seeds, no srun)"
            exit 0
            ;;
        *)
            USER_ARGS+=("$_arg")
            ;;
    esac
done

# Resolve seed index safely using scheduler-derived TASK_IDX
seed_idx=$TASK_IDX
# Ensure seed_idx is an integer >=0
if ! [[ "$seed_idx" =~ ^[0-9]+$ ]]; then
    echo "Warning: task index ('$seed_idx') is not an integer; defaulting to 0"
    seed_idx=0
fi

seed_count=${#seed[@]}
if [ "$seed_count" -eq 0 ]; then
    echo "Error: seed array is empty" >&2
    exit 1
fi
if [ "$seed_idx" -ge "$seed_count" ] || [ "$seed_idx" -lt 0 ]; then
    echo "Warning: seed index ($seed_idx) out of range (0..$((seed_count-1))); defaulting to 0"
    seed_idx=0
fi
selected_seed=${seed[$seed_idx]}

echo "Running in mode: $MODE (scheduler: $SCHEDULER)"
echo "Using seed index: $seed_idx -> seed: $selected_seed"
if [ "$PDB_MODE" -eq 1 ]; then
    echo "Debug mode: running under pdb (local only)"
fi

echo "=============================================="
echo "Toy Experiment Sweep Configuration:"
echo "  Datasets: ${datasets[*]}"
echo "  Acquisition methods: ${acquisition_function[*]}"
echo "  Seeds: ${seed[*]}"
echo "  Batch size: ${acquisition_batch_size}"
echo "  Cycles: ${num_active_learning_cycles}"
echo "  Output base: ${LOG_BASE_DIR}"
echo "=============================================="

# Loop over all datasets
for dataset_name in "${datasets[@]}"; do
    echo ""
    echo "====== Starting dataset: ${dataset_name} ======"
    
    # Create per-dataset output directories
    output_directory="${LOG_BASE_DIR}/${dataset_name}/output"
    save_fig_dir="${LOG_BASE_DIR}/${dataset_name}/figures"
    mkdir -p "${output_directory}"
    mkdir -p "${save_fig_dir}"
    
    # Build Python command for this dataset
    PY_CMD=(python3 activehit/apps/toy_experiment.py
        --cache_directory="${cache_directory}"
        --output_directory="${output_directory}"
        --model_name="${model_name}"
        --acquisition_batch_size="${acquisition_batch_size}"
        --num_active_learning_cycles="${num_active_learning_cycles}"
        --dataset_name="${dataset_name}"
        --save_fig_dir="${save_fig_dir}"
    )

    # Append any user-provided args safely (avoid unbound var expansion when empty)
    if [ ${#USER_ARGS[@]} -gt 0 ]; then
        PY_CMD+=("${USER_ARGS[@]}")
    fi

    # Append acquisition functions as multiple flags so Python can parse them individually
    for af in "${acquisition_function[@]}"; do
        PY_CMD+=(--acquisition_function_name "$af")
    done

    # Append seeds: in cluster mode, run a single seed per task; locally, pass all seeds
    if [ "$MODE" = "cluster" ]; then
        PY_CMD+=(--seed "${selected_seed}")
    else
        for s in "${seed[@]}"; do
            PY_CMD+=(--seed "$s")
        done
    fi

    # Launch for this dataset
    if [ "$MODE" = "cluster" ]; then
        echo "Launching with srun for dataset ${dataset_name}..."
        srun "${PY_CMD[@]}"
    else
        if [ "$PDB_MODE" -eq 1 ]; then
            echo "Launching ${dataset_name} locally under pdb..."
            python3 -m pdb "${PY_CMD[@]:1}"
        else
            echo "Launching ${dataset_name} locally..."
            "${PY_CMD[@]}"
        fi
    fi
    
    echo "====== Finished dataset: ${dataset_name} ======"
done

echo ""
echo "=============================================="
echo "All datasets completed!"
echo "Results saved to: ${LOG_BASE_DIR}"
echo "=============================================="

# Optional: Create a combined summary across all datasets
echo "Creating combined summary across all datasets..."
COMBINED_SUMMARY="${LOG_BASE_DIR}/combined_summary.csv"
FIRST_FILE=1
for dataset_name in "${datasets[@]}"; do
    SUMMARY_FILE="${LOG_BASE_DIR}/${dataset_name}/output/summary_by_seed.csv"
    if [ -f "$SUMMARY_FILE" ]; then
        if [ "$FIRST_FILE" -eq 1 ]; then
            # Include header from first file, add dataset column header
            head -1 "$SUMMARY_FILE" | sed 's/^/dataset,/' > "$COMBINED_SUMMARY"
            FIRST_FILE=0
        fi
        # Append data rows with dataset name prepended
        tail -n +2 "$SUMMARY_FILE" | sed "s/^/${dataset_name},/" >> "$COMBINED_SUMMARY"
    fi
done

if [ -f "$COMBINED_SUMMARY" ]; then
    echo "Combined summary written to: ${COMBINED_SUMMARY}"
else
    echo "Warning: No summary files found to combine."
fi