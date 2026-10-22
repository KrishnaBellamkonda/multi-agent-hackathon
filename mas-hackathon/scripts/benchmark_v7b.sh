#!/bin/bash
#BSUB -J maehd_v7b_hybrid
#BSUB -G team361
#BSUB -q gpu-lotfollahi
#BSUB -M 16000
#BSUB -R "select[mem>16000 && ngpus>0] rusage[mem=16000]"
#BSUB -gpu "num=1"
#BSUB -n 2
#BSUB -o logs/v7b_hybrid_%J.out
#BSUB -e logs/v7b_hybrid_%J.err

set -euo pipefail
cd /lustre/scratch126/cellgen/lotfollahi/ar36/AEHD
source .venv/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE

mkdir -p logs results/v7b_hybrid

# ── Start Ollama in background ──
export OLLAMA_MODELS="$PWD/ollama_models"
export OLLAMA_HOST="127.0.0.1:11434"
./ollama_dist/bin/ollama serve &
OLLAMA_PID=$!
sleep 10

echo "Checking Ollama..."
./ollama_dist/bin/ollama list || { echo "Ollama failed to start"; kill $OLLAMA_PID 2>/dev/null; exit 1; }

echo "=================================================="
echo " MAEHD v7b — Hybrid LLM Policy"
echo " GP exploitation + LLM exploration + cycle memory"
echo "=================================================="
echo ""

# v7b improvements over v7:
#   - GP picks 70% of batch (pure max_poh — floors performance)
#   - LLM picks 30% from tiebreaker pool (borderline candidates)
#   - LLM gets cycle memory (which gene picks were hits)
#   - Narrower tiebreaker pool (60 genes, not 100)
#
# This bounds downside: worst case = 70% optimal + 30% random ≈ 90% of classic
# Best case: LLM biology picks extra hits from the tiebreaker pool

# ── 1. T-cell achilles — standard benchmark ──
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
    --output-dir results/v7b_hybrid \
    2>&1 | tee logs/v7b_t_cell_achilles.log

# ── 2. T-cell fusion ──
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
    --output-dir results/v7b_hybrid \
    2>&1 | tee logs/v7b_t_cell_fusion.log

# ── 3. Cross-biology achilles ──
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
    --output-dir results/v7b_hybrid \
    2>&1 | tee logs/v7b_cross_biology_achilles.log

# ── 4. Cross-biology fusion ──
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
    --output-dir results/v7b_hybrid \
    2>&1 | tee logs/v7b_cross_biology_fusion.log

# ── 5. Neuro-immune achilles ──
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
    --output-dir results/v7b_hybrid \
    2>&1 | tee logs/v7b_neuro_immune_achilles.log

# ── 6. Full diverse achilles ──
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
    --output-dir results/v7b_hybrid \
    2>&1 | tee logs/v7b_full_diverse_achilles.log

# Cleanup
kill $OLLAMA_PID 2>/dev/null || true

echo ""
echo "=================================================="
echo " v7b benchmark complete!"
echo "=================================================="
echo " Results in: results/v7b_hybrid/"
