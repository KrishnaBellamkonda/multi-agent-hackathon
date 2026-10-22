# ActiveHit

Code for the ActiveHit paper. Implements probability-of-hit and Thompson-hit acquisition functions for active learning in genomic hit discovery, with experiments on synthetic and real CRISPR screen datasets.

## Install

```bash
pip install -r requirements.txt
```

## Use

### Synthetic data experiments

Benchmarks acquisition methods on synthetic datasets with a GP surrogate, where the ground truth function is known.

```bash
python activehit/apps/toy_experiment.py \
            --output_directory=$output_directory \
            --save_fig_dir=$saving_dir \
            --model_name="gp" \
            --acquisition_function_name=$acquisition_function \
            --acquisition_batch_size=10 \
            --num_active_learning_cycles=10 \
            --dataset_name=$dataset_name
```

- `$acquisition_function`: one of `probability_hit`, `thompson_hit`, `thompson_sampling`, `top_k`, `random`, `straddle`, `truvar`, `eitau`, `topk_bax`, `levelset_bax`, `subsetmax_bax`, `discobax`
- `$dataset_name`: one of `sine`, `sine2d`, `branin`, `perturbation_sem`, `perturbation_pathways`

Multiple methods and seeds in one run:

```bash
--acquisition_function_name "['probability_hit','thompson_hit','random']"
--seed "[0,1,2]"
```

Per-seed results are written to `$output_directory/summary_by_seed.csv`; aggregated means/stds to `summary_agg.csv`.

### GeneDisco experiments

Runs acquisition methods on real CRISPR screen datasets using an MC Dropout neural network surrogate.

```bash
python activehit/apps/genedisco_experiment.py \
            --cache_directory=$cache_directory \
            --output_directory=$output_directory \
            --performance_file_location=$performance_file_location \
            --model_name="bayesian_mlp" \
            --acquisition_function_name=$acquisition_function \
            --acquisition_batch_size=200 \
            --num_active_learning_cycles=10 \
            --feature_set_name="achilles" \
            --dataset_name=$dataset_name \
            --seed=$seed
```

- `$cache_directory`: where GeneDisco datasets are stored/cached
- `$dataset_name`: `schmidt_2021_ifng`, `schmidt_2021_il2`, `zhuang_2019_nk`, `sanchez_2021_tau`, or `zhu_2021_sarscov2`

## Reproducibility

See [`reproducibility/TUTORIAL.md`](reproducibility/TUTORIAL.md) for the self-contained reproducibility package.

## Authors

Andrea Rubbi<sup>1,2</sup>, Arpit Merchant<sup>1</sup>, Amir Hossein Hosseini Akbarnejad<sup>1</sup>, Sattar Vakili<sup>1,3</sup>, Mohammad Lotfollahi<sup>1</sup>

1. Wellcome Sanger Institute
2. University of Cambridge
3. MediaTek Research
