#!/bin/bash
#BSUB -J maehd_v7_llm_policy
#BSUB -q gpu-lotfollahi
#BSUB -M 16000
#BSUB -R "select[mem>16000 && ngpus>0] rusage[mem=16000]"
#BSUB -gpu "num=1"
#BSUB -n 2
#BSUB -o logs/v7_llm_policy_%J.out
#BSUB -e logs/v7_llm_policy_%J.err

set -euo pipefail
cd /lustre/scratch126/cellgen/lotfollahi/ar36/AEHD
source .venv/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE

mkdir -p logs results/v7_llm_policy
# Update log paths to match LSF job ID
exec > >(tee -a logs/v7_llm_policy_${LSB_JOBID}.out) 2>&1

# ── Start Ollama in background ──
export OLLAMA_MODELS="$PWD/ollama_models"
export OLLAMA_HOST="127.0.0.1:11434"
./ollama_dist/bin/ollama serve &
OLLAMA_PID=$!
sleep 10

# Verify ollama
echo "Checking Ollama..."
./ollama_dist/bin/ollama list || { echo "Ollama failed to start"; kill $OLLAMA_PID 2>/dev/null; exit 1; }

echo "=================================================="
echo " MAEHD v7 — LLM-as-Policy Benchmark"
echo " Agent IS the acquisition function"
echo "=================================================="
echo ""

# ── v7 Innovation: LLM-as-Policy ──
# The LLM sees top-100 GP candidates WITH gene names and directly
# selects which 30 to query, using biology knowledge to break ties.
#
# Comparison arms:
#   classic_max_poh  — pure GP math (baseline)
#   llm_agent        — LLM chooses which math formula (v6 approach)
#   llm_policy       — LLM directly picks genes (v7 innovation)

# ── 1. T-cell (achilles) — our standard benchmark ──
echo ""
echo "=== Setup: t_cell / achilles ==="
python -m maehd.benchmark_real \
    --setup t_cell \
    --feature-set achilles \
    --seeds 42 43 44 45 46 \
    --n-cycles 15 \
    --batch-size 30 \
    --with-agent \
    --no-batch-scoring \
    --no-gene-embeddings \
    --output-dir results/v7_llm_policy \
    2>&1 | tee logs/v7_t_cell_achilles.log

# ── 2. T-cell FUSION (achilles+string) — best v6 result ──
echo ""
echo "=== Setup: t_cell / fusion ==="
python -m maehd.benchmark_real \
    --setup t_cell \
    --fusion \
    --seeds 42 43 44 45 46 \
    --n-cycles 15 \
    --batch-size 30 \
    --with-agent \
    --no-batch-scoring \
    --no-gene-embeddings \
    --output-dir results/v7_llm_policy \
    2>&1 | tee logs/v7_t_cell_fusion.log

# ── 3. Cross-biology: tau + sars (completely different biology) ──
echo ""
echo "=== Setup: cross_biology / achilles ==="
python -m maehd.benchmark_real \
    --setup cross_biology \
    --feature-set achilles \
    --seeds 42 43 44 45 46 \
    --n-cycles 15 \
    --batch-size 30 \
    --with-agent \
    --no-batch-scoring \
    --no-gene-embeddings \
    --output-dir results/v7_llm_policy \
    2>&1 | tee logs/v7_cross_biology_achilles.log

# ── 4. Cross-biology FUSION ──
echo ""
echo "=== Setup: cross_biology / fusion ==="
python -m maehd.benchmark_real \
    --setup cross_biology \
    --fusion \
    --seeds 42 43 44 45 46 \
    --n-cycles 15 \
    --batch-size 30 \
    --with-agent \
    --no-batch-scoring \
    --no-gene-embeddings \
    --output-dir results/v7_llm_policy \
    2>&1 | tee logs/v7_cross_biology_fusion.log

# ── 5. Neuro-immune: ifng + tau ──
echo ""
echo "=== Setup: neuro_immune / achilles ==="
python -m maehd.benchmark_real \
    --setup neuro_immune \
    --feature-set achilles \
    --seeds 42 43 44 45 46 \
    --n-cycles 15 \
    --batch-size 30 \
    --with-agent \
    --no-batch-scoring \
    --no-gene-embeddings \
    --output-dir results/v7_llm_policy \
    2>&1 | tee logs/v7_neuro_immune_achilles.log

# ── 6. Full diverse (all 5 screens) ──
echo ""
echo "=== Setup: full_diverse / achilles ==="
python -m maehd.benchmark_real \
    --setup full_diverse \
    --feature-set achilles \
    --seeds 42 43 44 45 46 \
    --n-cycles 15 \
    --batch-size 30 \
    --with-agent \
    --no-batch-scoring \
    --no-gene-embeddings \
    --output-dir results/v7_llm_policy \
    2>&1 | tee logs/v7_full_diverse_achilles.log

# Cleanup
kill $OLLAMA_PID 2>/dev/null || true

echo ""
echo "=================================================="
echo " v7 benchmark complete!"
echo "=================================================="
echo " Results in: results/v7_llm_policy/"
echo " Logs in: logs/"
