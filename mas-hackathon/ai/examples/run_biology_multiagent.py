"""Multi-agent biology experiment: BiologyOrchestratorAgent on synthetic data.

Demonstrates the full discussion protocol:
  - K BiologySpecialistAgents (one per phenotype) nominate genes independently
  - BiologyOrchestratorAgent moderator resolves nominations → select_genes()
  - BiologyEnv (use_gp=False) runs the acquisition loop without a GP surrogate

Usage:
    cd /Users/ar36/Desktop/mas-hackathon
    python -m ai.examples.run_biology_multiagent

Requires OPENAI_API_KEY in .env or environment.
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Load .env if present (no python-dotenv dependency)
_env_path = os.path.join(_REPO_ROOT, ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

import numpy as np

from maehd.datasets.synthetic import pathway_multitarget, compute_thresholds
from acua.envs.biology.env import BiologyEnv, BiologyEnvConfig
from acua.agents.biology_orchestrator import BiologyOrchestratorAgent
from acua.agents.biology_specialist import SpecialistConfig
from acua.providers.openai_provider import OpenAIProvider


# ── Config ─────────────────────────────────────────────────────────────────

MODEL = "gpt-4o-mini"   # cheap dev model — replace with gpt-4o for production
N_GENES = 150
K = 2                   # two synthetic targets
BATCH_SIZE = 6
N_CYCLES = 3
N_INITIAL = 10
NOMINATIONS_PER_SPECIALIST = 4   # each specialist proposes this many genes
SEED = 42


def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set.  Create .env at repo root.")
        sys.exit(1)

    print(f"Model: {MODEL}  |  Genes: {N_GENES}  |  Targets: {K}  |  "
          f"Cycles: {N_CYCLES}  |  Batch: {BATCH_SIZE}\n")

    # ── Dataset ───────────────────────────────────────────────────────────
    X, Y, meta = pathway_multitarget(n_genes=N_GENES, n_pathways=5, K=K, seed=SEED)
    thresholds = compute_thresholds(Y, topk_percent=0.10)
    gene_names = [f"GENE_{i:04d}" for i in range(N_GENES)]
    # Use real screen names so specialists get phenotype-specific prompts
    target_names = ["schmidt_2021_ifng", "schmidt_2021_il2"]

    for k, name in enumerate(target_names):
        n_hits = int((Y[:, k] > thresholds[k]).sum())
        print(f"  {name}: {n_hits} hits (threshold={thresholds[k]:.3f})")

    # ── Environment (no GP) ───────────────────────────────────────────────
    cfg = BiologyEnvConfig(
        use_gp=False,
        batch_size=BATCH_SIZE,
        n_cycles=N_CYCLES,
        n_initial_random=N_INITIAL,
        pool_sample_size=30,   # genes shown per specialist per cycle
        seed=SEED,
    )
    env = BiologyEnv(
        X=X, Y=Y, thresholds=thresholds,
        gene_names=gene_names, target_names=target_names,
        config=cfg,
    )

    # ── Agents ────────────────────────────────────────────────────────────
    # A single shared provider is used for all agents to minimise API calls.
    # In production you may give each specialist its own provider/model.
    shared_provider = OpenAIProvider(
        model_name=MODEL,
        api_key=api_key,
        max_tokens=512,
    )

    specialist_configs = [
        SpecialistConfig(target_names[k], shared_provider, nominations=NOMINATIONS_PER_SPECIALIST)
        for k in range(K)
    ]

    orchestrator = BiologyOrchestratorAgent(
        moderator_provider=shared_provider,
        specialist_configs=specialist_configs,
        batch_size=BATCH_SIZE,
    )

    # ── Episode loop ──────────────────────────────────────────────────────
    obs = env.reset()
    print(f"\nEpisode started — {N_INITIAL} random seed observations done.")
    print("=" * 60)

    done = False
    while not done:
        exp = obs.experiment
        cycle = exp.cycle if exp else "?"
        print(f"\n--- Discussion cycle {cycle}/{N_CYCLES} ---")

        # Nomination phase: printed by orchestrator logging (set level=INFO to see)
        action = orchestrator.get_python_action(obs)

        # Show what each specialist nominated
        for target, nom in orchestrator.last_nominations.items():
            genes_str = ", ".join(nom["genes"][:5])
            reasoning_preview = nom["reasoning"][:100].replace("\n", " ")
            print(f"  [{target}] → {genes_str}")
            print(f"    reason: {reasoning_preview}...")

        print(f"  [moderator] → {action}")

        obs, reward, done, info = env.step(action)

        if obs.target_metrics:
            tm = obs.target_metrics
            recalls = "  ".join(
                f"{n}={r:.3f}"
                for n, r in zip(tm.target_names, tm.per_target_recall)
            )
            mean_r = sum(tm.per_target_recall) / len(tm.per_target_recall)
            print(f"  recall: {recalls}  mean={mean_r:.3f}  reward={reward:.3f}")

        if info.get("parse_error"):
            print(f"  [env error] {info['parse_error']}")

    # ── Final summary ─────────────────────────────────────────────────────
    if obs.target_metrics:
        tm = obs.target_metrics
        mean_r = sum(tm.per_target_recall) / len(tm.per_target_recall)
        print("\n=== Final Results ===")
        for k, name in enumerate(tm.target_names):
            n_hits = int((Y[:, k] > thresholds[k]).sum())
            recovered = int(round(tm.per_target_recall[k] * n_hits))
            print(f"  {name}: {tm.per_target_recall[k]:.3f} ({recovered}/{n_hits} hits)")
        print(f"  mean recall: {mean_r:.3f}")

    env.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
