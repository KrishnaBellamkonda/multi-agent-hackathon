# runners/

Development experiment runners. These scripts launch experiments directly against the `activehit/` codebase and are intended for iterating on methods, sweeping hyperparameters, and debugging locally or on a cluster.

> **For reproducing paper results**, use the `reproducibility/` package instead — it is self-contained, documented, and not dependent on internal app paths. See [reproducibility/TUTORIAL.md](../reproducibility/TUTORIAL.md).

---

## Files

| File                      | Purpose                                                                       |
| ------------------------- | ----------------------------------------------------------------------------- |
| `toy_experiment.py`       | Run acquisition methods on synthetic (GP) datasets                            |
| `genedisco_experiment.py` | Run acquisition methods on real GeneDisco CRISPR screens                      |
| `toy_experiment.sh`       | Cluster launcher for `activehit/apps/toy_experiment.py` (Slurm / LSF / local) |
| `genedisco_experiment.sh` | Cluster launcher for `activehit/apps/genedisco_experiment.py` (Slurm)         |

---

## toy_experiment.py

Runs one or more acquisition methods on synthetic datasets using a GP surrogate. Defaults to a single representative run (one dataset, one batch size, one seed). Pass extra flags to sweep.

```bash
# Default: probability_hit vs baselines on Sine, batch=10, seed=0, 10 cycles
python runners/toy_experiment.py

# Quick sanity check (3 cycles)
python runners/toy_experiment.py --quick

# Specific dataset and methods
python runners/toy_experiment.py --datasets sine,sine2d,branin --methods probability_hit,thompson_hit

# Paper ablation sweep: all datasets, 3 batch sizes, 5 seeds
python runners/toy_experiment.py \
    --datasets sine,sine2d,branin,perturbation_sem,perturbation_pathways \
    --methods probability_hit,thompson_hit,thompson_sampling,top_k,random \
    --batch_sizes 5,10,15 \
    --hit_thresholds 5,10,20 \
    --seeds 0,1,2,3,4

# Dry run: print commands without executing
python runners/toy_experiment.py --dry-run
```

**Output** is saved to `./ablation_toy_<timestamp>/` with per-experiment CSVs, figures, and a `summary.json`.

**Datasets available:**

| Name                    | Dim | Description                            |
| ----------------------- | --- | -------------------------------------- |
| `sine`                  | 1D  | Sinusoidal function                    |
| `sine2d`                | 2D  | 2D sinusoidal function                 |
| `branin`                | 2D  | Branin-Hoo test function (multi-modal) |
| `perturbation_sem`      | 6D  | SEM-based perturbation model           |
| `perturbation_pathways` | 4D  | Pathway-based perturbation model       |

**Methods available:** `probability_hit`, `thompson_hit`, `thompson_sampling`, `top_k`, `random`, `straddle`, `truvar`, `levelset_bax`, `eitau`, `discobax` (slow).

---

## genedisco_experiment.py

Runs acquisition methods on real CRISPR screen datasets from GeneDisco using an MC Dropout neural network surrogate. Defaults to a single representative run.

```bash
# Default: all methods on schmidt_2021_ifng, batch=200, seed=0
python runners/genedisco_experiment.py

# Specific dataset
python runners/genedisco_experiment.py --datasets zhuang_2019_nk

# Paper sweep: all 5 datasets, 5 seeds (750 experiments — allow 12–24 hours)
python runners/genedisco_experiment.py \
    --datasets schmidt_2021_ifng,schmidt_2021_il2,zhuang_2019_nk,sanchez_2021_tau,zhu_2021_sarscov2 \
    --methods probability_hit,thompson_hit,thompson_sampling,top_k,random \
    --batch_sizes 100,200,500 \
    --seeds 0,1,2,3,4

# Export plots from an existing run
python runners/genedisco_experiment.py \
    --export_only --ablation_dir ./output/run_20260415_120000
```

**Datasets available:**

| Name                | Description               | ~Size        |
| ------------------- | ------------------------- | ------------ |
| `schmidt_2021_ifng` | T-cell IFN-γ screen       | 18 000 genes |
| `schmidt_2021_il2`  | T-cell IL-2 screen        | 18 000 genes |
| `zhuang_2019_nk`    | NK cell activation screen | 18 000 genes |
| `sanchez_2021_tau`  | Tau aggregation screen    | 18 000 genes |
| `zhu_2021_sarscov2` | SARS-CoV-2 host factors   | 18 000 genes |

> GeneDisco datasets are downloaded automatically on first use via the `genedisco` package.

---

## toy_experiment.sh / genedisco_experiment.sh

Bash launchers that support running on a **Slurm cluster**, **LSF cluster**, or **locally**. They call `activehit/apps/toy_experiment.py` and `activehit/apps/genedisco_experiment.py` directly.

```bash
# Local run (detects no scheduler, runs all seeds sequentially)
bash runners/toy_experiment.sh

# Force local mode explicitly
bash runners/toy_experiment.sh --local

# Cluster submission (array job — each task runs one seed)
sbatch runners/toy_experiment.sh
```

**Configuration** is edited directly inside the shell scripts (top section):

- `datasets` array — which datasets to run
- `acquisition_function` array — which methods to run
- `seed` array — random seeds
- `acquisition_batch_size` — points per cycle
- `num_active_learning_cycles` — number of cycles

The shell scripts are primarily used for cluster submission. For most local development prefer the Python runners above.
