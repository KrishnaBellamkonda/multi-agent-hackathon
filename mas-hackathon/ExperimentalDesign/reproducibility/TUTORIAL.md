# Reproducibility Tutorial

This package reproduces all experiments from the paper. It is organised into two tracks — **toy (synthetic) datasets** and **real GeneDisco datasets** — each with an ablation study and a complexity analysis.

---

## Package Structure

```
reproducibility/
├── TUTORIAL.md                   ← you are here
├── README.md                     ← high-level overview, method/metric tables
├── requirements.txt              ← Python dependencies
├── setup_environment.sh          ← conda environment setup
├── Makefile                      ← convenience targets (quick, full, test)
├── config/
│   └── ablation_config.yaml      ← all default hyperparameters
├── shared/                       ← shared utilities used by both tracks
│   ├── datasets.py               ← generate / load toy datasets
│   ├── acquisition.py            ← all acquisition functions (P(Hit), Thompson, etc.)
│   ├── experiment.py             ← single-experiment runner and ExperimentConfig
│   └── metrics.py                ← CHR, SMAPE, Spearman, HitClusters
├── toy/                          ← synthetic dataset experiments
│   ├── ablation/
│   │   └── run_ablation.py       ← toy ablation entry point
│   └── complexity/
│       ├── analyze_complexity.py         ← complexity analysis
│       └── analyze_dataset_complexity.py ← extended analysis with visualisations
├── genedisco/                    ← real biological screen experiments
│   ├── ablation/
│   │   └── run_ablation.py       ← GeneDisco ablation entry point
│   ├── complexity/
│   │   ├── analyze_complexity.py         ← GeneDisco complexity analysis
│   │   └── analyze_complexity_real.py    ← extended analysis with plots
│   ├── run_single_config.py      ← run one experiment from CLI (debugging / parallelism)
│   ├── run_ablation.sh           ← bash launcher for full GeneDisco sweep
│   └── compute_smape_posthoc.py  ← post-hoc SMAPE from saved models
└── tests/
    └── test_core.py              ← unit tests for datasets, acquisition, metrics
```

---

## 1. Setup

```bash
# From the repo root
cd reproducibility
./setup_environment.sh        # creates conda env "ablation_repro"
conda activate ablation_repro
# or if using existing pip venv:
pip install -r requirements.txt
```

---

## 2. Quick sanity check (≈2 min)

```bash
# From the repo root (ExperimentalDesign/)
python -m reproducibility.toy.ablation.run_ablation \
    --output-dir output/quick_test \
    --datasets sine \
    --methods probability_hit thompson_hit \
    --seeds 0 \
    --n-cycles 3
```

Expected: `output/quick_test/results.csv` with 6 rows (1 dataset × 2 methods × 3 cycles).

---

## 3. Toy Dataset Ablation Study

### What it does

Compares 5 acquisition methods (P(Hit), Thompson Hit, Thompson Sampling, Top-K, Random) on 5 synthetic datasets (Sine 1D, Sine 2D, Branin 2D, SEM 6D, Pathways 4D) across a grid of batch sizes and hit thresholds. Uses a GP surrogate. This is the controlled setting where Theorem 4.3 applies.

### Full run (≈4–8 hours, 1 125 experiments)

```bash
python -m reproducibility.toy.ablation.run_ablation \
    --output-dir output/toy_ablation \
    --seeds 0 1 2 3 4
```

### Programmatic API

```python
from reproducibility.toy.ablation.run_ablation import run_ablation_study

results = run_ablation_study(
    output_dir='output/toy_ablation',
    datasets=['sine', 'sine2d', 'branin', 'perturbation_sem', 'perturbation_pathways'],
    methods=['probability_hit', 'thompson_hit', 'thompson_sampling', 'top_k', 'random'],
    batch_sizes=[5, 10, 20],
    thresholds=[5, 10, 20],
    seeds=[0, 1, 2, 3, 4],
    n_cycles=10,
    verbose=True
)
```

### Key parameters (from `config/ablation_config.yaml`)

| Parameter     | Default      | Notes                     |
| ------------- | ------------ | ------------------------- |
| `n_samples`   | 300          | Pool size per dataset     |
| `n_initial`   | 22           | Initial labelled set size |
| `batch_sizes` | 5, 10, 20    | Points acquired per cycle |
| `thresholds`  | 5%, 10%, 20% | Hit definition cutoff     |
| `n_cycles`    | 10           | Active learning cycles    |
| `seeds`       | 0–4          | Random seeds              |

### Output

```
output/toy_ablation/
├── results.csv          # one row per (method, dataset, batch, threshold, seed, cycle)
├── summary.json         # config + aggregate metrics
└── plots/               # CHR curves, bar charts
```

---

## 4. GeneDisco (Real Data) Ablation Study

### What it does

Same acquisition function comparison on 5 real CRISPR screen datasets from GeneDisco, using an MC Dropout neural network surrogate. This is the empirical validation track — Theorem 4.3 does not apply here (surrogate is miscalibrated), but P(Hit) is evaluated purely empirically.

### Prerequisites

GeneDisco datasets must be downloaded first:

```bash
# Requires genedisco package installed
python -c "from genedisco.datasets.screens import load_screen; load_screen('schmidt_2021_ifng')"
```

### Full run (≈12–24 hours, 750 experiments)

```bash
cd reproducibility/genedisco
bash run_ablation.sh
# or equivalently:
python -m reproducibility.genedisco.ablation.run_ablation \
    --output-dir output/genedisco_ablation \
    --seeds 0 1 2 3 4
```

### Run a single configuration (useful for HPC job arrays)

```bash
python reproducibility/genedisco/run_single_config.py \
    --dataset schmidt_2021_ifng \
    --feature_set string \
    --method probability_hit \
    --batch_size 200 \
    --seed 42 \
    --n_cycles 10 \
    --output_dir output/genedisco_ablation
```

### Compute SMAPE post-hoc (after models are saved)

```bash
python reproducibility/genedisco/compute_smape_posthoc.py \
    --results_dir output/genedisco_ablation
```

### Output

```
output/genedisco_ablation/
├── results.csv
├── summary.json
└── plots/
```

---

## 5. Complexity Analysis

### What it does

Quantifies dataset difficulty via hit clustering, local smoothness, and dimensionality, and correlates these metrics with method performance gaps.

### Toy datasets

```bash
python -m reproducibility.toy.complexity.analyze_complexity \
    --output-dir output/complexity/toy
```

### Real GeneDisco datasets

```bash
python reproducibility/genedisco/complexity/analyze_complexity_real.py \
    --output_dir output/complexity/genedisco \
    --cache_dir path/to/genedisco/cache
```

---

## 6. Generate Paper Figures

After running experiments, generate all figures used in the paper:

```bash
# Toy figures
python reproducibility/toy/ablation/run_ablation.py \
    --output-dir output/toy_ablation --paper-figures

# Or use the Makefile shortcut
make paper-outputs ABLATION_DIR=output/toy_ablation
```

---

## 7. Verify Results

Check that your run matches the expected paper values:

```bash
make verify ABLATION_DIR=output/toy_ablation
# or
python -m pytest reproducibility/tests/test_core.py -v
```

Expected key results from the paper:

| Dataset (toy, cycle 5, batch 5, threshold 10%) | P(Hit) CHR  | Next-best CHR       |
| ---------------------------------------------- | ----------- | ------------------- |
| Sine (1D)                                      | 0.81 ± 0.06 | 0.74 (Thompson Hit) |
| Sine2D (2D)                                    | 0.58 ± 0.17 | 0.54 (Thompson Hit) |
| Branin (2D)                                    | 0.30 ± 0.07 | 0.26 (Thompson Hit) |
| SEM (6D)                                       | 0.35 ± 0.10 | 0.34 (Straddle)     |
| Pathways (4D)                                  | 0.36 ± 0.16 | 0.29 (LS-BAX)       |

P(Hit) wins **44/50** dataset × cycle comparisons.

---

## 8. Relation to Root-Level Scripts

The `runners/toy_experiment.py` and `runners/genedisco_experiment.py` are **development scripts** — they call `activehit/apps/toy_experiment.py` directly and were used during active development. This `reproducibility/` package is the **publication-ready** version: self-contained, documented, and not dependent on internal app paths.

Use the `reproducibility/` package to reproduce paper results. Use `runners/` scripts if you want to quickly iterate on new methods or datasets during development.

---

## 9. Acquisition Functions Reference

| Name                | Description                          | Module                  |
| ------------------- | ------------------------------------ | ----------------------- |
| `probability_hit`   | P(y > τ) from GP/MC posterior        | `shared/acquisition.py` |
| `thompson_hit`      | Thompson sample → top predicted hits | `shared/acquisition.py` |
| `thompson_sampling` | Standard Thompson sampling (argmax)  | `shared/acquisition.py` |
| `top_k`             | Greedy top-k by posterior mean       | `shared/acquisition.py` |
| `random`            | Uniform random baseline              | `shared/acquisition.py` |
| `straddle`          | Level-set boundary uncertainty       | `activehit/methods/`    |
| `truvar`            | Variance reduction at hit region     | `activehit/methods/`    |

---

## 10. Troubleshooting

**Import errors**: make sure you run from the `ExperimentalDesign/` root, not from inside `reproducibility/`.

**GPyTorch errors**: `pip install gpytorch --upgrade`

**GeneDisco download issues**: ensure `genedisco` is installed and you have internet access for the first download; subsequent runs use the cache.

**Slow toy runs**: reduce `n_workers` argument (default 1) or use fewer seeds for testing.
