#!/bin/bash
# Full ablation: batch=200, thresh=10%, 5 seeds, all datasets, all feature sets
# Methods: probability_hit, top_k, thompson_hit, random

OUTDIR="./full_ablation"
BATCH=200
THRESH=10.0
CYCLES=10

DATASETS="schmidt_2021_ifng schmidt_2021_il2 zhuang_2019_nk sanchez_2021_tau zhu_2021_sarscov2"
FEATURES="string achilles"
METHODS="probability_hit top_k thompson_hit random"
SEEDS="0 1 2 3 4"

total=0
for d in $DATASETS; do
  for f in $FEATURES; do
    for m in $METHODS; do
      for s in $SEEDS; do
        total=$((total + 1))
      done
    done
  done
done
echo "Total experiments: $total"
