# ActiveHit Reproducibility Studies

This directory contains all code necessary to reproduce the experiments from the ActiveHit paper.

## Structure

```
reproducibility/
├── __init__.py
├── README.md
├── shared/                    # Common utilities
│   ├── __init__.py
│   ├── datasets.py           # Dataset loading and generation
│   ├── metrics.py            # Performance metrics
│   ├── acquisition.py        # Acquisition functions
│   └── experiment.py         # Experiment orchestration
├── toy/                       # Synthetic dataset experiments
│   ├── __init__.py
│   ├── ablation/             # Ablation studies
│   │   ├── __init__.py
│   │   └── run_ablation.py
│   └── complexity/           # Complexity analysis
│       ├── __init__.py
│       └── analyze_complexity.py
└── genedisco/                 # Real biological data experiments
    ├── __init__.py
    ├── ablation/             # Ablation studies on GeneDisco
    │   ├── __init__.py
    │   └── run_ablation.py
    └── complexity/           # Complexity analysis
        ├── __init__.py
        └── analyze_complexity.py
```

## Requirements

```bash
pip install -r requirements.txt
```

Core dependencies:

- numpy
- pandas
- scipy
- scikit-learn
- torch
- gpytorch
- matplotlib
- seaborn

## Running Experiments

### Toy Dataset Ablation Study

```python
from reproducibility.toy.ablation import run_ablation_study

# Run full study (all datasets, methods, configurations)
results = run_ablation_study(
    output_dir='output/toy_ablation',
    seeds=[0, 1, 2, 3, 4],
    verbose=True
)

# Run with specific configuration
results = run_ablation_study(
    output_dir='output/toy_ablation',
    datasets=['sine', 'branin'],
    methods=['probability_hit', 'thompson_hit'],
    batch_sizes=[10, 20],
    threshold_pcts=[0.05, 0.10],
    seeds=[42],
    verbose=True
)
```

Or from command line:

```bash
cd ExperimentalDesign
python -m reproducibility.toy.ablation.run_ablation \
    --output-dir output/toy_ablation \
    --datasets sine branin \
    --methods probability_hit thompson_hit \
    --seeds 0 1 2
```

### GeneDisco Ablation Study

```python
from reproducibility.genedisco.ablation import run_genedisco_ablation

results = run_genedisco_ablation(
    output_dir='output/genedisco_ablation',
    cache_dir='path/to/genedisco/cache',
    datasets=['schmidt_2021_ifng', 'zhuang_2019_nk'],
    methods=['probability_hit', 'thompson_hit', 'random'],
    seeds=[0, 1, 2, 3, 4],
    verbose=True
)
```

### Complexity Analysis

```python
# Toy datasets
from reproducibility.toy.complexity import analyze_toy_complexity

complexity_df = analyze_toy_complexity(
    output_dir='output/complexity/toy',
    datasets=['sine', 'branin', 'perturbation_sem'],
    threshold_pcts=[0.05, 0.10, 0.20],
    verbose=True
)

# GeneDisco datasets
from reproducibility.genedisco.complexity import analyze_genedisco_complexity

complexity_df = analyze_genedisco_complexity(
    cache_dir='path/to/genedisco/cache',
    output_dir='output/complexity/genedisco',
    datasets=['schmidt_2021_ifng', 'zhuang_2019_nk'],
    threshold_pcts=[0.05, 0.10, 0.20],
    verbose=True
)
```

## Study Configuration

### Default Parameters

| Parameter   | Toy          | GeneDisco |
| ----------- | ------------ | --------- |
| Batch sizes | 10, 20, 50   | 200       |
| Thresholds  | 5%, 10%, 20% | 10%       |
| Cycles      | 10           | 10        |
| Seeds       | 0-4          | 0-4       |

### Acquisition Functions

| Name                | Description                 |
| ------------------- | --------------------------- |
| `random`            | Random acquisition baseline |
| `thompson_sampling` | Standard Thompson sampling  |
| `probability_hit`   | Hit probability (discobax)  |
| `thompson_hit`      | Thompson sampling for hits  |
| `top_k`             | Greedy top-k selection      |

### Datasets

**Toy datasets:**

- `sine` - 1D sinusoidal function
- `sine2d` - 2D sinusoidal function
- `branin` - Branin test function (2D)
- `perturbation_sem` - SEM perturbation model
- `perturbation_pathways` - Pathway-based perturbation

**GeneDisco datasets:**

- `schmidt_2021_ifng` - IFN-γ response screen
- `schmidt_2021_il2` - IL-2 response screen
- `zhuang_2019_nk` - NK cell activation screen
- `sanchez_2021_tau` - Tau aggregation screen
- `zhu_2021_sarscov2` - SARS-CoV-2 infection screen

## Metrics

| Metric                          | Description                           |
| ------------------------------- | ------------------------------------- |
| `RecoveredHitsRatio_Cumulative` | Fraction of true hits discovered      |
| `SMAPE_Unseen`                  | Prediction error on unobserved points |
| `SpearmanRho`                   | Rank correlation with true values     |
| `HitClusters_RecoveryRatio`     | Diversity of discovered hits          |

## Output Format

Results are saved as:

- `results.csv` - All cycle metrics
- `summary.json` - Study configuration and summary
- `plots/` - Visualization figures
