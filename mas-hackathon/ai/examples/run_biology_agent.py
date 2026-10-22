"""End-to-end dev run: BiologyAgent (gpt-4o-mini) on synthetic multi-target data.

Usage:
    # Set your API key first (one-time):
    #   echo "OPENAI_API_KEY=sk-..." > .env   (repo root)
    # Then run:
    cd /Users/ar36/Desktop/mas-hackathon
    python -m ai.examples.run_biology_agent

The script runs 3 full acquisition cycles.  The LLM receives the observation
prompt (cycle status, per-target recall, candidate pool table) and must call
select_genes([...]) via the tool API.  Each selection is forwarded to
BiologyEnv.step(), which updates the GP and returns a new observation.
"""

from __future__ import annotations

import os
import sys

# Ensure repo root is importable
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Load .env if present (best-effort; no hard dependency on python-dotenv)
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
from acua.agents.biology_agent import BiologyAgent
from acua.providers.openai_provider import OpenAIProvider


# ── Config ────────────────────────────────────────────────────────────────────

MODEL = "gpt-4o-mini"          # cheap dev model
N_GENES = 150                   # small synthetic problem
K = 2                           # number of targets
BATCH_SIZE = 8
N_CYCLES = 3                    # short run for dev
N_INITIAL = 10
MAX_CANDIDATES = 25             # candidates shown to LLM
SEED = 42


# ── Dataset ───────────────────────────────────────────────────────────────────

def load_data():
    X, Y, meta = pathway_multitarget(n_genes=N_GENES, n_pathways=5, K=K, seed=SEED)
    thresholds = compute_thresholds(Y, topk_percent=0.10)
    gene_names = [f"GENE_{i:04d}" for i in range(len(X))]
    target_names = [f"target_{k}" for k in range(K)]
    return X, Y, thresholds, gene_names, target_names


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set.  Create a .env file at repo root:")
        print('  echo "OPENAI_API_KEY=sk-..." > .env')
        sys.exit(1)

    print(f"Using model: {MODEL}")
    print(f"Genes: {N_GENES}, Targets: {K}, Cycles: {N_CYCLES}, Batch: {BATCH_SIZE}\n")

    # Load data
    X, Y, thresholds, gene_names, target_names = load_data()
    for k, name in enumerate(target_names):
        n_hits = int((Y[:, k] > thresholds[k]).sum())
        print(f"  {name}: {n_hits} hits (threshold={thresholds[k]:.3f})")

    # Build env
    cfg = BiologyEnvConfig(
        batch_size=BATCH_SIZE,
        n_cycles=N_CYCLES,
        n_initial_random=N_INITIAL,
        max_candidates=MAX_CANDIDATES,
        seed=SEED,
    )
    env = BiologyEnv(
        X=X, Y=Y, thresholds=thresholds,
        gene_names=gene_names, target_names=target_names,
        config=cfg,
    )

    # Build agent with cheap dev model
    provider = OpenAIProvider(
        model_name=MODEL,
        api_key=api_key,
        max_tokens=512,        # small for dev; tool calls are concise
    )
    agent = BiologyAgent(provider=provider)

    # Reset env
    obs = env.reset()
    print("\n" + "=" * 60)
    print("Episode started — initial random observations done.")
    print("=" * 60)

    cycle = 0
    done = False

    while not done:
        cycle += 1
        print(f"\n--- Agent cycle {cycle}/{N_CYCLES} ---")

        # Agent produces a single LLM call → action string
        action = agent.get_python_action(obs)
        llm_output = agent.last_response

        if llm_output and llm_output.think:
            print(f"[think] {llm_output.think[:200].strip()}")
        print(f"[action] {action}")

        obs, reward, done, info = env.step(action)

        # Print recall after each cycle
        if obs.target_metrics:
            tm = obs.target_metrics
            recalls = "  ".join(
                f"{n}={r:.3f}" for n, r in zip(tm.target_names, tm.per_target_recall)
            )
            mean_r = sum(tm.per_target_recall) / len(tm.per_target_recall) if tm.per_target_recall else 0.0
            print(f"[recall] {recalls}  mean={mean_r:.3f}  reward={reward:.3f}")

        if info.get("error"):
            print(f"[env error] {info['error']}")

        if done:
            print("\nEpisode finished.")
            break

    # Final summary
    if obs.target_metrics:
        tm = obs.target_metrics
        mean_r = sum(tm.per_target_recall) / len(tm.per_target_recall) if tm.per_target_recall else 0.0
        print("\n=== Final Results ===")
        for k, name in enumerate(tm.target_names):
            n_hits = int((Y[:, k] > thresholds[k]).sum())
            recovered = int(round(tm.per_target_recall[k] * n_hits))
            print(f"  {name}: {tm.per_target_recall[k]:.3f} ({recovered}/{n_hits} hits)")
        print(f"  mean recall: {mean_r:.3f}")

    env.close()


if __name__ == "__main__":
    main()
