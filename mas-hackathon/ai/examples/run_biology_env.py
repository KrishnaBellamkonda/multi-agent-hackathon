"""Minimal runnable example: biology environment + agent (stub provider).

Loads a synthetic multi-target dataset, instantiates the BiologyEnv and a
BiologyAgent backed by a no-op stub provider, runs 3 steps, and prints the
rendered observation prompt.

Usage:
    cd /Users/ar36/Desktop/mas-hackathon
    python -m ai.examples.run_biology_env
"""

from __future__ import annotations

import sys
import os

# Ensure the repo root is on sys.path when running as a script
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

from maehd.datasets.synthetic import pathway_multitarget, compute_thresholds
from acua.envs.biology.env import BiologyEnv, BiologyEnvConfig
from acua.prompts.dynamic_prompting import ExperimentObservationPrompt, ObsFlags


def _render_obs(obs, flags) -> str:
    prompt = ExperimentObservationPrompt(obs, flags)
    return prompt.to_user_message().content[0]["text"]


def main() -> None:
    # ── Load synthetic dataset ────────────────────────────────────────
    print("Loading pathway_multitarget dataset ...")
    X, Y, meta = pathway_multitarget(n_genes=200, n_pathways=6, K=3, seed=0)
    thresholds = compute_thresholds(Y, topk_percent=0.10)

    target_names = ["pathway_A", "pathway_B", "pathway_C"]
    gene_names = [f"GENE_{i:04d}" for i in range(len(X))]

    print(f"  Genes: {X.shape[0]}, Features: {X.shape[1]}, Targets: {Y.shape[1]}")
    for k, name in enumerate(target_names):
        n_hits = int((Y[:, k] > thresholds[k]).sum())
        print(f"  {name}: {n_hits} hits (threshold={thresholds[k]:.3f})")

    # ── Build environment ─────────────────────────────────────────────
    cfg = BiologyEnvConfig(
        batch_size=10,
        n_cycles=5,
        n_initial_random=10,
        max_candidates=20,
        seed=42,
    )
    env = BiologyEnv(
        X=X, Y=Y, thresholds=thresholds,
        gene_names=gene_names, target_names=target_names,
        config=cfg,
    )

    flags = ObsFlags(
        use_html=False,
        use_ax_tree=False,
        use_screenshot=False,
        use_experiment_state=True,
        use_candidate_pool=True,
        max_candidates=20,
        show_correlation=False,
        show_last_batch=True,
        use_error_logs=True,
    )

    # ── Reset and render initial observation ──────────────────────────
    print("\n" + "=" * 60)
    print("INITIAL OBSERVATION (after random seed)")
    print("=" * 60)
    obs = env.reset()
    print(_render_obs(obs, flags))

    # ── Step 1: select top candidates by max_poh ──────────────────────
    if obs.candidates:
        top_genes = [c.gene_name for c in obs.candidates[:cfg.batch_size]]
    else:
        top_genes = gene_names[:cfg.batch_size]

    action1 = f"select_genes(genes={top_genes!r})"
    print("=" * 60)
    print(f"ACTION 1: {action1[:80]}...")
    print("=" * 60)
    obs, reward, done, info = env.step(action1)
    print(f"Reward: {reward:.3f}  |  Done: {done}")
    print(_render_obs(obs, flags))

    if done:
        print("Episode finished.")
        return

    # ── Step 2: select next top candidates ────────────────────────────
    if obs.candidates:
        top_genes2 = [c.gene_name for c in obs.candidates[:cfg.batch_size]]
    else:
        top_genes2 = gene_names[cfg.batch_size : 2 * cfg.batch_size]

    action2 = f"select_genes(genes={top_genes2!r})"
    print("=" * 60)
    print(f"ACTION 2: {action2[:80]}...")
    print("=" * 60)
    obs, reward, done, info = env.step(action2)
    print(f"Reward: {reward:.3f}  |  Done: {done}")
    print(_render_obs(obs, flags))

    # ── Step 3: early stop ────────────────────────────────────────────
    action3 = 'finish(summary="Stopping after 2 greedy cycles for demo purposes.")'
    print("=" * 60)
    print(f"ACTION 3: {action3}")
    print("=" * 60)
    obs, reward, done, info = env.step(action3)
    print(f"Reward: {reward:.3f}  |  Done: {done}")

    tm = obs.target_metrics
    if tm:
        print("\nFinal recall:")
        for k, name in enumerate(tm.target_names):
            print(f"  {name}: {tm.per_target_recall[k]:.3f} ({tm.per_target_hits[k]} hits)")

    env.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
