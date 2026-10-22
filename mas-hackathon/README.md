# MAEHD — Multi-Agent Experimental Hit Discovery

A multi-agent LLM framework for multi-target CRISPR gene-perturbation screens.
Specialist agents reason over biological phenotypes independently; a moderator integrates their nominations into a single batch selection.
Compared against Gaussian-process baselines including ParEGO (Knowles 2006) on real GeneDisco datasets.

---

## Table of Contents

1. [Background](#1-background)
2. [Project structure](#2-project-structure)
3. [Installation](#3-installation)
4. [Datasets](#4-datasets)
5. [Methods](#5-methods)
6. [Running experiments](#6-running-experiments)
   - [Smoke test](#61-smoke-test)
   - [Single setup](#62-single-setup)
   - [Distributed runs](#63-distributed-runs-split-across-people)
   - [Full comparison](#64-full-comparison)
7. [Results format](#7-results-format)
8. [Configuration reference](#8-configuration-reference)
9. [Extending the framework](#9-extending-the-framework)
10. [References](#10-references)

---

## 1. Background

### Problem

A CRISPR perturbation screen interrogates a pool of ~17,000 genes.
Each experiment **queries a batch** of genes and reveals whether each gene is a **hit** — i.e. whether perturbing it significantly affects a target phenotype (e.g. T-cell IFNγ secretion).
The goal is to recover as many hits as possible within a fixed query budget.

In the **multi-target** setting, a single query reveals responses for **all K phenotypes simultaneously**.
This creates an information-sharing opportunity: a gene that is a hit for two targets at once is worth more than two single-target hits.

### Approach

We frame this as **batch active learning** over a discrete gene pool and compare four strategies:

| Method | Model | Cross-target information |
|---|---|---|
| `random` | None | None |
| `parego` | Single GP per cycle (scalarized) | Implicit via Chebyshev weighting |
| `singleagent` | One LLM, all targets visible | Full |
| `multiagent` | K specialist LLMs + moderator | Structured decomposition |

The `multiagent` method is the main contribution.
Each specialist agent sees **only its own target's history** and nominates gene candidates.
The moderator then sees all nominations and selects the final batch, prioritising genes that serve multiple targets and correcting recall imbalances.

---

## 2. Project structure

```
mas-hackathon/
│
├── ai/                          # Agent framework (importable as acua via symlink)
│   ├── agents/
│   │   ├── biology_orchestrator.py   # BiologyOrchestratorAgent (K specialists + moderator)
│   │   ├── biology_specialist.py     # BiologySpecialistAgent (nominate_genes per target)
│   │   ├── biology_singleagent.py    # BiologySingleAgent (ablation: one LLM, all targets)
│   │   └── cua_agent.py              # Base CUAAgent (tool-calling conversation manager)
│   ├── envs/
│   │   └── biology/
│   │       ├── env.py            # BiologyEnv — active learning loop, action parsing
│   │       └── tools.py          # select_genes(), nominate_genes(), finish()
│   ├── experiments/
│   │   └── compare_real.py       # Main comparison runner
│   └── prompts/
│       └── dynamic_prompting.py  # Observation/discussion prompts for all agent types
│
├── acua -> ai/                   # Symlink: acua is the installed package name
│
├── maehd/                        # Experiment harness
│   ├── datasets/
│   │   └── genedisco.py          # load_multitarget(), MULTI_TARGET_SETUPS
│   └── ...
│
├── ExperimentalDesign/           # GP baselines and reproducibility
│   ├── data/                     # 715 MB of .h5 gene feature + screen files (git-lfs)
│   └── reproducibility/shared/
│       ├── multitarget_experiment.py  # Naive independent-GP baseline runner
│       ├── parego_experiment.py       # ParEGO runner
│       ├── experiment.py              # GP training/prediction (GPyTorch)
│       ├── acquisition.py             # probability_hit, random, etc.
│       └── metrics.py                 # hit recovery ratio
│
├── references/                   # Key papers (PDFs)
├── pyproject.toml
└── README.md
```

The `acua` directory is a symlink to `ai/`.
All imports use `from acua.xxx import` — this resolves to `ai/` on any OS that preserves symlinks (macOS, Linux; see [Windows note](#windows)).

---

## 3. Installation

### Prerequisites

- Python ≥ 3.10
- [git-lfs](https://git-lfs.com/) — required to pull the 715 MB data files
- An OpenAI API key — only needed for `singleagent` and `multiagent` methods

### Steps

```bash
# 1. Install git-lfs, then clone
git lfs install
git clone <repo-url>
cd mas-hackathon

# 2. Install PyTorch for your hardware FIRST — versions differ per platform
#    See https://pytorch.org/get-started/locally/

# CPU (universal):
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Apple Silicon (MPS):
pip install torch

# CUDA 12.x:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. Install everything else (editable so local edits take effect immediately)
pip install -e .

# 4. Create a .env file with your OpenAI key (skip if using --no-agent)
echo "OPENAI_API_KEY=sk-..." > .env
```

> **Windows note:** The `acua → ai` symlink requires `git config core.symlinks true` before cloning, or use Git Bash / WSL.

### Verify the install

```bash
# Should print "all imports OK" (KeOps warnings about CUDA/OpenMP are harmless)
python -c "
import ai.agents.biology_orchestrator
import maehd.datasets.genedisco
import ExperimentalDesign.reproducibility.shared.parego_experiment
print('all imports OK')
"
```

---

## 4. Datasets

All data comes from real published CRISPR perturbation screens via [GeneDisco](https://github.com/genedisco/genedisco).
The 5 individual screens are combined into 6 predefined **setups** covering different levels of biological relatedness:

| Setup | Screens (K targets) | Gene pool | Relationship |
|---|---|---|---|
| `t_cell` | IFNγ + IL-2 (K=2) | ~17,470 | Correlated — same cell type, same lab |
| `all_immune` | IFNγ + IL-2 + NK (K=3) | ~14,560 | Same lab, 3 immune phenotypes |
| `cross_biology` | Tau + SARS-CoV-2 (K=2) | ~16,363 | Unrelated biology, different labs |
| `neuro_immune` | IFNγ + Tau (K=2) | ~15,200 | Immune vs. neurodegeneration |
| `virus_immune` | IFNγ + SARS-CoV-2 (K=2) | ~15,800 | Immune vs. virology |
| `diverse` | IFNγ + IL-2 + Tau + SARS-CoV-2 (K=4) | ~13,900 | Maximum biological diversity |

**Screens used:**

| Screen name | Cell type | Phenotype | Source |
|---|---|---|---|
| `schmidt_2021_ifng` | CD4+ T cells | IFNγ secretion | Schmidt et al., Science 2021 |
| `schmidt_2021_il2` | CD4+ T cells | IL-2 secretion | Schmidt et al., Science 2021 |
| `zhuang_2019_nk` | NK cells | Cytotoxicity resistance | Zhuang et al. 2019 |
| `sanchez_2021_tau` | iPSC-derived neurons | Tau aggregation | Sanchez et al. 2021 |
| `zhu_2021_sarscov2` | HEK293T / Vero | SARS-CoV-2 entry | Zhu et al. 2021 |

**Gene features** (808-dimensional by default):

| Feature set | Description | Dimensions |
|---|---|---|
| `achilles` (default) | DepMap gene essentiality across ~1000 cell lines | 808 |
| `string` | STRING protein-protein interaction network embeddings | 800 |
| `ccle` | CCLE protein quantification | 420 |

**Hits** are defined as the top 5% of genes by response score per target (`--topk-percent 0.05`).
Each setup has ~820–870 hits per target at this threshold.

---

## 5. Methods

### `random` — Random baseline

Selects genes uniformly at random from the unqueried pool each cycle.
No model. No API key needed.
Serves as a lower bound.

### `parego` — ParEGO (Knowles 2006)

Multi-objective Bayesian optimisation with random Chebyshev scalarization.

Each cycle:
1. Sample a random weight vector λ uniformly from the K-simplex.
2. Scalarize all observed gene responses into a single scalar:
   `z = max_k(λ_k · y_k) + 0.05 · Σ λ_k · y_k`
3. Fit **one GP** on `(X_obs, z_obs)` using GPyTorch.
4. Acquire the next batch by `probability_hit` on the scalarized GP.

PCA(20) is applied to the 808-dim Achilles features before GP fitting.
No API key needed.

### `singleagent` — Single-agent ablation

One `gpt-4o-mini` agent that sees **all K targets' full history** in a single prompt.
Calls `select_genes([...])` directly — no specialist decomposition or discussion phase.
Tests whether the specialist/moderator structure adds value over a flat single-agent approach.

### `multiagent` — Main method

K `BiologySpecialistAgent`s (one per phenotype) + one `_ModeratorAgent`.

**Phase 1 — Nomination:**
Each specialist sees only its own target's history (confirmed hits, misses, hit rate vs. background, last batch outcome) and nominates up to 5 candidate genes using `nominate_genes(genes=[...], reasoning="...")`.
Specialists are **blind to each other's nominations** — no cross-target bias.

**Phase 2 — Moderation:**
The moderator receives a discussion transcript with all nominations and reasonings.
It selects the final batch of genes, guided by:
- Cross-target overlap (genes nominated by multiple specialists)
- Recall imbalance (lagging targets get more slots)
- Biological diversity across the batch

The moderator outputs plain text containing `select_genes([...])` — no tool calling, which avoids model-specific tool-schema bugs.

**Information shown to agents each cycle:**

```
Progress: 15 hits / 60 genes tested = 25.0% hit rate (5.0× background of 5.0%)
Screen coverage: 15 / 872 hits discovered (1.7% recall)
Last Batch: 3/20 hits = 15.0% (3.0× background)

Candidate pool sample (200 genes):
  JAK1, STAT3, IRF1, TYK2, SOCS1, EP300 ...

Task: nominate up to 5 genes for the next query batch.
MUST ONLY nominate from the pool list above.
```

---

## 6. Running experiments

All experiments are run from the repo root:

```bash
cd mas-hackathon
```

The main entry point is:

```bash
python -m ai.experiments.compare_real [OPTIONS]
```

### 6.1 Smoke test

Verify all four methods work end-to-end before committing to a full run.
Runs 1 seed × 3 cycles × `t_cell` only. Takes ~1–5 minutes.

```bash
# All 4 methods (needs OPENAI_API_KEY):
python -m ai.experiments.compare_real --quick

# GP methods only — no API key needed, runs in ~30 seconds:
python -m ai.experiments.compare_real --quick --no-agent

# Just the multiagent method:
python -m ai.experiments.compare_real --quick --methods multiagent
```

### 6.2 Single setup

```bash
# One setup, default seeds and cycles:
python -m ai.experiments.compare_real --setup t_cell

# Multiple setups in one run:
python -m ai.experiments.compare_real --setup t_cell cross_biology neuro_immune

# GP baselines only on cross_biology:
python -m ai.experiments.compare_real --setup cross_biology --no-agent

# Custom seeds and cycles:
python -m ai.experiments.compare_real --setup t_cell --seeds 0 1 2 --cycles 15

# Save results to a specific file:
python -m ai.experiments.compare_real --setup t_cell --output results_tcell.json
```

### 6.3 Distributed runs (split across people)

Since the full comparison (6 setups × 2 seeds × 4 methods × 10 cycles) takes several hours, it can be split:

**Split by setup:**

```bash
# Person A
python -m ai.experiments.compare_real \
    --setup t_cell cross_biology neuro_immune \
    --output results_A.json

# Person B
python -m ai.experiments.compare_real \
    --setup all_immune virus_immune diverse \
    --output results_B.json
```

**Split by method:**

```bash
# Person A — GP baselines (no API key needed, fast)
python -m ai.experiments.compare_real \
    --methods random parego \
    --output results_gp.json

# Person B — LLM methods (needs API key)
python -m ai.experiments.compare_real \
    --methods singleagent multiagent \
    --output results_llm.json
```

Results JSONs can be merged manually — they share the same schema under the `"results"` key.

### 6.4 Full comparison

Runs all 6 setups × 2 seeds × 4 methods × 10 cycles.
Estimated time: 2–4 hours depending on API latency for LLM methods.

```bash
python -m ai.experiments.compare_real
# Saves to ai/experiments/compare_real_results.json
```

Progress is printed cycle-by-cycle:

```
--- multiagent / t_cell / seed=42 ---
  [cycle 1/10] specialists nominating...
    [schmidt_2021_ifng] → NR4A1, EOMES, BACH2, FOXP3
    [schmidt_2021_il2]  → NR4A1, EOMES, FOXP3, GRB2
    [moderator] → select_genes(genes=['NR4A1', 'EOMES', 'FOXP3', 'BACH2', 'GRB2', ...])
    hits/tested: schmidt_2021_ifng=4/40(0.5%)  schmidt_2021_il2=3/40(0.3%)
                 mean_recall=0.004  batch=4/20
```

The `batch=4/20` shows how many of the 20 queried genes in this cycle were hits.

---

## 7. Results format

Results are saved as JSON with this structure:

```json
{
  "config": {
    "setups": ["t_cell", "cross_biology"],
    "methods": ["random", "parego", "singleagent", "multiagent"],
    "seeds": [42, 43],
    "n_cycles": 10,
    "batch_size": 20,
    "n_initial": 20,
    "topk_percent": 0.05,
    "model": "gpt-4o-mini"
  },
  "results": [
    {
      "method": "multiagent",
      "setup": "t_cell",
      "seed": 42,
      "screen_names": ["schmidt_2021_ifng", "schmidt_2021_il2"],
      "recall_curves": [[0.002, 0.005, 0.008, ...], [0.003, 0.006, ...]],
      "final_per_target_recall": [0.014, 0.016],
      "final_mean_recall": 0.015,
      "elapsed_sec": 54.7,
      "success": true,
      "error_message": ""
    },
    ...
  ]
}
```

**Key metrics:**

- `recall_curves[k][cycle]` — recall for target k after cycle c = `hits_found / total_hits_in_screen`
- `final_mean_recall` — mean recall across all K targets at the end of the run
- `final_per_target_recall` — per-target breakdown

**Preliminary results** (10 cycles, batch=20, n_initial=20, 2 seeds, `gpt-4o-mini`, 220 genes queried per run):

`t_cell` — IFNγ: 872 hits total · IL-2: 872 hits total

| Method | Mean recall | IFNγ hits found / total | IL-2 hits found / total |
|---|---|---|---|
| random | 0.010 ±0.001 | 10 / 872 (1.1%) | 8 / 872 (0.9%) |
| parego | 0.011 ±0.001 | 6 / 872 (0.7%) | 12 / 872 (1.4%) |
| singleagent | 0.013 ±0.002 | 11 / 872 (1.3%) | 12 / 872 (1.4%) |
| multiagent† | 0.013 ±0.000 | 12 / 872 (1.4%) | 10 / 872 (1.1%) |

`cross_biology` — Tau: 817 hits total · SARS-CoV-2: 817 hits total

| Method | Mean recall | Tau hits found / total | SARS-CoV-2 hits found / total |
|---|---|---|---|
| random | 0.010 ±0.001 | 8 / 817 (1.0%) | 9 / 817 (1.1%) |
| parego | 0.013 ±0.001 | 11 / 817 (1.3%) | 11 / 817 (1.3%) |
| singleagent | 0.020 ±0.003 | 14 / 817 (1.7%) | 19 / 817 (2.3%) |
| multiagent | 0.014 ±0.002 | 10 / 817 (1.2%) | 13 / 817 (1.6%) |

> † multiagent t_cell: one seed failed (moderator tool-call bug, now fixed). Re-run for clean numbers.
> All results at 5% hit threshold. Background hit rate = 5%, so 2× background = 10%.

---

## 8. Configuration reference

### CLI flags for `compare_real.py`

| Flag | Default | Description |
|---|---|---|
| `--setup` | `all` | One or more setup names, or `all` |
| `--methods` | all four | Subset of `random parego singleagent multiagent` |
| `--seeds` | `42 43` | Random seeds (space-separated) |
| `--cycles` | `10` | Active learning cycles per run |
| `--output` | `compare_real_results.json` | Output JSON path |
| `--quick` | off | Smoke test: 1 seed, 3 cycles, t_cell only |
| `--no-agent` | off | Skip LLM methods (GP only, no API key needed) |

### Constants in `compare_real.py`

Edit these at the top of [ai/experiments/compare_real.py](ai/experiments/compare_real.py) to change defaults:

| Constant | Value | Description |
|---|---|---|
| `BATCH_SIZE` | 20 | Genes queried per cycle |
| `N_INITIAL` | 20 | Random genes queried before cycle 1 |
| `N_CYCLES` | 10 | Acquisition cycles per run |
| `SEEDS` | [42, 43] | Default random seeds |
| `TOPK_PERCENT` | 0.05 | Top 5% = hits |
| `PCA_DIMS` | 20 | PCA for GP methods |
| `POOL_SAMPLE_SIZE` | 200 | Genes shown to LLM agents per cycle |
| `NOMINATIONS_PER_SPECIALIST` | 5 | Genes each specialist nominates |
| `MODEL` | `gpt-4o-mini` | LLM model for all agent types |

### Environment variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | For LLM methods | Set in `.env` at repo root or shell |

---

## 9. Extending the framework

### Adding a new method

1. Write a runner function in `ai/experiments/compare_real.py` following the pattern of `run_random` / `run_parego` / `run_singleagent`.
2. Add an entry to the `SETUPS` dict if you need a new data configuration.
3. Add the method name to `--methods` choices in the argparse block.

### Adding a new dataset setup

Add an entry to `MULTI_TARGET_SETUPS` in [maehd/datasets/genedisco.py](maehd/datasets/genedisco.py):

```python
MULTI_TARGET_SETUPS = {
    ...
    "my_setup": ["schmidt_2021_ifng", "sanchez_2021_tau"],
}
```

Then add it to `SETUPS` in [ai/experiments/compare_real.py](ai/experiments/compare_real.py):

```python
SETUPS = {
    ...
    "my_setup": {
        "screen_names": MULTI_TARGET_SETUPS["my_setup"],
        "description": "My custom setup — IFNγ + Tau",
    },
}
```

### Using a different LLM

Change the `MODEL` constant or pass a different provider:

```python
from acua.providers.openai_provider import OpenAIProvider
from acua.providers.anthropic_provider import AnthropicProvider

# In run_multiagent():
provider = AnthropicProvider(model_name="claude-sonnet-4-6", api_key=api_key)
```

All providers implement `AbstractProvider` and are interchangeable.

### Running a single episode programmatically

```python
import numpy as np
from acua.envs.biology.env import BiologyEnv, BiologyEnvConfig
from acua.agents.biology_orchestrator import BiologyOrchestratorAgent
from acua.agents.biology_specialist import SpecialistConfig
from acua.providers.openai_provider import OpenAIProvider
from maehd.datasets.genedisco import load_multitarget

# Load data
X, Y, meta = load_multitarget(setup="t_cell", feature_set="achilles")

# Build environment
env = BiologyEnv(
    X=X, Y=Y,
    thresholds=meta["thresholds"],
    gene_names=meta["gene_names"],
    target_names=meta["screen_names"],
    config=BiologyEnvConfig(use_gp=False, batch_size=20, n_cycles=10),
)

# Build agent
provider = OpenAIProvider(model_name="gpt-4o-mini", api_key="sk-...")
orchestrator = BiologyOrchestratorAgent(
    moderator_provider=provider,
    specialist_configs=[
        SpecialistConfig(name, provider, nominations=5)
        for name in meta["screen_names"]
    ],
)

# Run episode
obs = env.reset()
done = False
while not done:
    action = orchestrator.get_python_action(obs)
    obs, reward, done, info = env.step(action)

print("Final recall:", obs.target_metrics.per_target_recall)
```

---

## 10. References

**Datasets and biology:**
- Schmidt et al. (2021). *CRISPR activation and interference screens decode stimulation responses in primary human T cells.* Science.
- Zhuang et al. (2019). Genome-wide CRISPR screen, NK cytotoxicity.
- Sanchez et al. (2021). iPSC-derived neuron screen for Tau aggregation.
- Zhu et al. (2021). CRISPR screen for SARS-CoV-2 host factors.
- Mehrjou et al. (2021). [GeneDisco: A Benchmark for Experimental Design in Drug Discovery.](https://arxiv.org/abs/2110.11875) ICML 2022.
- Tsherniak et al. (2017). [Defining a Cancer Dependency Map.](https://doi.org/10.1016/j.cell.2017.06.010) Cell. *(Achilles features)*

**Multi-objective Bayesian optimisation:**
- Knowles, J. (2006). [ParEGO: A Hybrid Algorithm with On-Line Landscape Approximation for Expensive Multiobjective Optimization Problems.](https://doi.org/10.1109/TEVC.2005.851274) IEEE TEVC. *(ParEGO baseline)*
- Daulton et al. (2020). [Differentiable Expected Hypervolume Improvement.](https://arxiv.org/abs/2006.05078) NeurIPS.

**LLM agents for science:**
- Bran et al. (2023). [ChemCrow: Augmenting LLMs with chemistry tools.](https://arxiv.org/abs/2304.05376) Nature Machine Intelligence.
- Boiko et al. (2023). [Autonomous chemical research with large language models.](https://doi.org/10.1038/s41586-023-06792-0) Nature.

**Gene representations:**
- Chen & Zou (2024). [GenePT: A Simple But Effective Foundation Model for Genes.](https://doi.org/10.1101/2023.10.16.562533) bioRxiv.
- Cui et al. (2024). [scGPT: Toward building a foundation model for single-cell multi-omics.](https://doi.org/10.1038/s41592-024-02201-0) Nature Methods.

---

## Authors

Andrea Rubbi — Wellcome Sanger Institute / University of Cambridge
