"""Biology gene-perturbation environment.

Manages one active-learning episode in which an agent (or a team of agents)
repeatedly selects batches of genes to query.  The environment reveals true
phenotypic responses, updates internal state, and returns a fresh observation.

Two operating modes are supported, selected via BiologyEnvConfig.use_gp:

  use_gp=False  (default — multi-agent mode)
      No surrogate model.  The observation carries per-target SpecialistViewHints
      (hit/miss history + a pool sample) so that specialist agents can reason
      from biology knowledge.  The orchestrator moderates their nominations.

  use_gp=True  (single-agent GP mode)
      A MultiOutputGP is refitted after every batch.  The observation carries
      a GP-ranked CandidateSummary pool used by BiologyAgent directly.

Interface mirrors AbstractEnv (reset / step / close) but does NOT subclass it
because this env owns its own state rather than wrapping a gym env.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass

import numpy as np

from acua.envs.hints import (
    CandidateSummary,
    ErrorHints,
    ExperimentHints,
    LastBatchHints,
    SpecialistViewHints,
    TargetMetricsHints,
)
from acua.envs.observation import Observation
from maehd import metrics as maehd_metrics
from maehd.models.gp import MultiOutputGP

logger = logging.getLogger(__name__)

# ── Action parameter map for positional-arg support ─────────────────
_POSITIONAL_PARAM_NAMES: dict[str, list[str]] = {
    "select_genes": ["genes"],
    "select_indices": ["indices"],
    "finish": ["summary"],
}


def _coerce_genes_string(s: str) -> list[str]:
    """Convert a string genes value to a list, handling all common LLM formats.

    Handles:
      '["JAK1", "STAT3"]'   → json.loads  → ["JAK1", "STAT3"]
      "JAK1, STAT3, IRF1"   → split/strip → ["JAK1", "STAT3", "IRF1"]
      'JAK1\\nSTAT3\\nIRF1' → normalise   → ["JAK1", "STAT3", "IRF1"]
    """
    import json as _json
    s = s.strip()
    # Try JSON array first (catches '["A","B"]' and "['A','B']")
    if s.startswith("["):
        try:
            result = _json.loads(s)
            if isinstance(result, list):
                return [str(g).strip() for g in result if str(g).strip()]
        except (_json.JSONDecodeError, ValueError):
            pass
        # Strip outer brackets and fall through to comma split
        s = s.strip("[]")
    # Comma-separated (with optional quotes): "JAK1, STAT3" or '"JAK1","STAT3"'
    return [g.strip().strip("\"'") for g in s.replace("\n", ",").split(",") if g.strip().strip("\"' ")]


def _normalise_action(action_str: str) -> str:
    """Normalise common LLM formatting quirks before AST parsing.

    LLMs sometimes produce multiline function calls or wrap the call in a
    code-fence.  Collapsing internal whitespace to single spaces lets the
    standard eval parser handle both single-line and multiline forms.
    """
    # Strip markdown fences (```python ... ```)
    lines = action_str.strip().splitlines()
    stripped = [l for l in lines if not l.strip().startswith("```")]
    # Collapse all whitespace runs to a single space so multiline calls
    # like select_genes(\n    genes=[...]\n) become one-liners.
    return " ".join(" ".join(stripped).split())


def _safe_parse_call(action_str: str) -> tuple[str, dict]:
    """Parse a function-call string safely via ast.

    Returns (func_name, kwargs_dict).  Raises ValueError on parse failure.
    """
    action_str = _normalise_action(action_str)
    try:
        tree = ast.parse(action_str, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Syntax error in action: {exc}") from exc

    if not isinstance(tree.body, ast.Call) or not isinstance(
        tree.body.func, ast.Name
    ):
        raise ValueError(
            "Action must be a simple function call, e.g. select_genes(genes=[...])"
        )

    call = tree.body
    func_name: str = call.func.id
    kwargs: dict = {}

    for kw in call.keywords:
        kwargs[kw.arg] = ast.literal_eval(kw.value)

    param_names = _POSITIONAL_PARAM_NAMES.get(func_name, [])
    for i, arg in enumerate(call.args):
        if i < len(param_names):
            kwargs[param_names[i]] = ast.literal_eval(arg)

    return func_name, kwargs


# ── Environment configuration ────────────────────────────────────────


@dataclass
class BiologyEnvConfig:
    """Configuration for a BiologyEnv episode.

    use_gp controls which operating mode the env runs in:

      False (default): multi-agent mode.  No GP.  Observations carry
        specialist_views (per-target hit history + pool sample).
        Intended for use with BiologyOrchestratorAgent.

      True: single-agent GP mode.  A MultiOutputGP is fitted after each
        batch and the observation carries a GP-ranked candidate pool.
        Intended for use with the original BiologyAgent.
    """
    batch_size: int = 20
    n_cycles: int = 15
    n_initial_random: int = 10
    # ── Multi-agent (no-GP) mode ──
    use_gp: bool = False          # set True to enable GP surrogate
    pool_sample_size: int = 100   # unobserved genes shown per specialist
    # ── GP mode (only used when use_gp=True) ──
    max_candidates: int = 100     # GP-ranked candidates shown to single agent
    gp_lengthscale: float = 0.3
    gp_amplitude: float = 1.0
    gp_noise: float = 0.01
    optimize_gp: bool = False
    seed: int = 42


# ── Main environment class ───────────────────────────────────────────


class BiologyEnv:
    """Active-learning environment for multi-target gene hit discovery.

    Usage
    -----
    obs = env.reset()
    while True:
        action = agent.get_python_action(obs)
        obs, reward, done, info = env.step(action)
        if done:
            break
    """

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        thresholds: np.ndarray,
        gene_names: list[str] | None = None,
        target_names: list[str] | None = None,
        config: BiologyEnvConfig | None = None,
    ):
        """
        Parameters
        ----------
        X : (n, d)  gene feature matrix
        Y : (n, K)  ground-truth multi-target responses (not revealed to agent)
        thresholds : (K,)  per-target hit thresholds
        gene_names : optional list of n gene name strings
        target_names : optional list of K target name strings
        config : experiment hyper-parameters
        """
        n, d = X.shape
        K = Y.shape[1]

        self._X = X.copy()
        self._Y = Y.copy()
        self._thresholds = thresholds.copy()
        self._n = n
        self._d = d
        self._K = K

        self._gene_names: list[str] = (
            gene_names if gene_names is not None else [f"GENE_{i:04d}" for i in range(n)]
        )
        self._target_names: list[str] = (
            target_names if target_names is not None else [f"target_{k}" for k in range(K)]
        )

        # Pre-compute true hit index sets for each target
        self._true_hits: list[np.ndarray] = [
            np.where(Y[:, k] > thresholds[k])[0] for k in range(K)
        ]

        cfg = config or BiologyEnvConfig()
        self._batch_size = cfg.batch_size
        self._n_cycles = cfg.n_cycles
        self._n_initial_random = cfg.n_initial_random
        self._seed = cfg.seed
        self._use_gp = cfg.use_gp
        self._pool_sample_size = cfg.pool_sample_size
        # GP-mode params (ignored when use_gp=False)
        self._max_candidates = cfg.max_candidates
        self._gp_lengthscale = cfg.gp_lengthscale
        self._gp_amplitude = cfg.gp_amplitude
        self._gp_noise = cfg.gp_noise
        self._optimize_gp = cfg.optimize_gp

        # Mutable episode state (initialised by reset())
        self._obs_mask: np.ndarray = np.zeros(n, dtype=bool)
        self._cycle: int = 0
        self._budget_remaining: int = 0
        self._last_cycle_hit_rate: float = 0.0
        self._gp: MultiOutputGP | None = None
        self._last_batch_hints: LastBatchHints | None = None
        self._rng = np.random.default_rng(cfg.seed)
        self._done: bool = False

    # ── Public interface ──────────────────────────────────────────────

    def reset(self) -> Observation:
        """Reset to a fresh episode.

        Seeds the GP with n_initial_random randomly observed genes, then
        returns the first observation with a ranked candidate pool.
        """
        self._obs_mask = np.zeros(self._n, dtype=bool)
        self._cycle = 0
        self._budget_remaining = self._batch_size * self._n_cycles
        self._last_cycle_hit_rate = 0.0
        self._gp = None
        self._last_batch_hints = None
        self._done = False
        self._rng = np.random.default_rng(self._seed)

        # Seed with random initial observations before the first agent cycle.
        # These give agents a baseline: some confirmed hits and misses to reason from.
        if self._n_initial_random > 0:
            pool_idx = np.arange(self._n)
            init_idx = self._rng.choice(
                pool_idx, size=min(self._n_initial_random, self._n), replace=False
            )
            self._obs_mask[init_idx] = True
            if self._use_gp:
                self._fit_gp()

        return self._build_observation()

    def step(self, action: str) -> tuple[Observation, float, bool, dict]:
        """Execute a select_genes / select_indices / finish action.

        Returns
        -------
        obs : updated Observation
        reward : fraction of new hits found this cycle (new_hits / batch_size)
        done : True when budget or cycle limit is reached
        info : dict with selected_indices, hit_flags, parse_error
        """
        if self._done:
            return self._build_observation(), 0.0, True, {"already_done": True}

        info: dict = {}
        error_msg = ""
        selected_indices: list[int] = []
        terminated_early = False

        try:
            func_name, kwargs = _safe_parse_call(action)
        except ValueError as exc:
            error_msg = str(exc)
            info["parse_error"] = error_msg
            return self._build_observation(error_msg), 0.0, False, info

        if func_name == "finish":
            terminated_early = True
            info["finish_summary"] = kwargs.get("summary", "")

        elif func_name in ("select_genes", "select_indices"):
            try:
                selected_indices = self._resolve_action(func_name, kwargs)
            except ValueError as exc:
                error_msg = str(exc)
                info["parse_error"] = error_msg
                return self._build_observation(error_msg), 0.0, False, info
        else:
            error_msg = (
                f"Unknown action '{func_name}'. "
                "Use select_genes(genes=[...]) or select_indices(indices=[...])."
            )
            info["parse_error"] = error_msg
            return self._build_observation(error_msg), 0.0, False, info

        # ── Update state ──────────────────────────────────────────────

        hits_before = int(
            sum(len(set(np.where(self._obs_mask)[0]) & set(th)) for th in self._true_hits)
        )

        for idx in selected_indices:
            self._obs_mask[idx] = True

        hits_after = int(
            sum(len(set(np.where(self._obs_mask)[0]) & set(th)) for th in self._true_hits)
        )

        new_hits = hits_after - hits_before
        reward = new_hits / max(self._batch_size, 1)
        self._last_cycle_hit_rate = new_hits / max(len(selected_indices), 1)

        self._budget_remaining = max(0, self._budget_remaining - len(selected_indices))
        self._cycle += 1

        # Refit GP if running in GP mode
        if self._use_gp and len(selected_indices) > 0:
            self._fit_gp()

        # Record last batch outcomes
        obs_set = set(np.where(self._obs_mask)[0])
        per_target_hits = [
            len(set(selected_indices) & set(th)) for th in self._true_hits
        ]
        self._last_batch_hints = LastBatchHints(
            selected_genes=[self._gene_names[i] for i in selected_indices],
            selected_indices=selected_indices,
            was_hit_any=any(h > 0 for h in per_target_hits),
            per_target_hits=per_target_hits,
        )

        info["selected_indices"] = selected_indices
        info["per_target_new_hits"] = per_target_hits

        done = (
            terminated_early
            or self._budget_remaining <= 0
            or self._cycle >= self._n_cycles
        )
        self._done = done

        return self._build_observation(), reward, done, info

    def close(self) -> None:
        self._gp = None

    # ── Internal helpers ──────────────────────────────────────────────

    def _resolve_action(self, func_name: str, kwargs: dict) -> list[int]:
        """Convert action kwargs to validated global gene indices.

        Unknown gene names are silently dropped and replaced with random
        pool genes so that one bad nomination doesn't waste a full cycle.
        A hard error is only raised if the action is structurally invalid
        (wrong type, already-observed index, etc.).
        """
        pool_arr = np.where(~self._obs_mask)[0]
        pool_set = set(pool_arr.tolist())

        if func_name == "select_genes":
            genes = kwargs.get("genes")
            if isinstance(genes, str):
                genes = _coerce_genes_string(genes)
            if not isinstance(genes, list):
                raise ValueError("select_genes requires genes=[...] (a list of strings)")

            name_to_idx = {self._gene_names[i]: i for i in pool_set}
            selected: list[int] = []
            unknown: list[str] = []
            seen: set[int] = set()
            for g in genes:
                if g in name_to_idx:
                    idx = name_to_idx[g]
                    if idx not in seen:
                        selected.append(idx)
                        seen.add(idx)
                else:
                    unknown.append(g)

            if unknown:
                logger.warning(
                    "%d gene(s) not in pool (silently dropped): %s%s",
                    len(unknown), unknown[:5], " ..." if len(unknown) > 5 else "",
                )

            # Pad to batch_size with random pool genes if we came up short
            n_needed = self._batch_size - len(selected)
            if n_needed > 0 and len(pool_arr) > len(selected):
                candidates = [i for i in pool_arr if i not in seen]
                n_pad = min(n_needed, len(candidates))
                if n_pad > 0:
                    pad = self._rng.choice(len(candidates), n_pad, replace=False)
                    for p in pad:
                        selected.append(candidates[p])

            return selected

        # select_indices
        indices = kwargs.get("indices")
        if not isinstance(indices, list):
            raise ValueError("select_indices requires indices=[...] (a list of ints)")
        selected = []
        invalid: list[int] = []
        for idx in indices:
            if not isinstance(idx, int) or idx < 0 or idx >= self._n:
                invalid.append(idx)
            elif idx not in pool_set:
                raise ValueError(f"Gene index {idx} has already been observed.")
            else:
                selected.append(idx)
        if invalid:
            raise ValueError(f"Invalid gene indices: {invalid}")
        return selected

    def _fit_gp(self) -> None:
        """Refit the multi-output GP on all observed genes."""
        obs_idx = np.where(self._obs_mask)[0]
        if len(obs_idx) < 2:
            return
        X_obs = self._X[obs_idx]
        Y_obs = self._Y[obs_idx]
        self._gp = MultiOutputGP(
            self._K,
            lengthscale=self._gp_lengthscale,
            amplitude=self._gp_amplitude,
            noise=self._gp_noise,
        )
        self._gp.fit(X_obs, Y_obs, optimize=self._optimize_gp)

    def _build_candidate_pool(self) -> list[CandidateSummary]:
        """Return top max_candidates unobserved genes ranked by max P(hit)."""
        pool_idx = np.where(~self._obs_mask)[0]
        if len(pool_idx) == 0:
            return []

        n_cand = min(self._max_candidates, len(pool_idx))

        if self._gp is None:
            # No GP yet: return a random sample with zero scores
            chosen = self._rng.choice(len(pool_idx), n_cand, replace=False)
            return [
                CandidateSummary(
                    gene_name=self._gene_names[pool_idx[i]],
                    gene_index=int(pool_idx[i]),
                    max_poh=0.0,
                    mean_sigma=0.0,
                    per_target_poh=[0.0] * self._K,
                )
                for i in chosen
            ]

        X_pool = self._X[pool_idx]
        poh = self._gp.predict_phit(X_pool, self._thresholds)   # (n_pool, K)
        _, sigma = self._gp.predict(X_pool)                      # (n_pool, K)

        max_poh = poh.max(axis=1)
        mean_sigma = sigma.mean(axis=1)

        order = np.argsort(max_poh)[::-1][:n_cand]

        return [
            CandidateSummary(
                gene_name=self._gene_names[pool_idx[i]],
                gene_index=int(pool_idx[i]),
                max_poh=float(max_poh[i]),
                mean_sigma=float(mean_sigma[i]),
                per_target_poh=[float(poh[i, k]) for k in range(self._K)],
            )
            for i in order
        ]

    def _build_observation(self, last_action_error: str = "") -> Observation:
        """Assemble a full Observation from current episode state.

        In GP mode (use_gp=True): populates candidates (GP-ranked pool),
          cross-target correlation, and mean GP sigma.
        In multi-agent mode (use_gp=False): populates specialist_views
          (per-target hit history + pool sample) instead.
        """
        obs_idx = np.where(self._obs_mask)[0]

        # ── Shared metrics ────────────────────────────────────────────
        if len(obs_idx) > 0:
            recall = maehd_metrics.per_target_recall(obs_idx, self._true_hits)
            hits = maehd_metrics.hit_count(obs_idx, self._true_hits)
        else:
            recall = np.zeros(self._K)
            hits = np.zeros(self._K, dtype=int)

        # Names of every gene observed so far — needed by the orchestrator
        # to validate specialist nominations against the true pool.
        observed_names = [self._gene_names[i] for i in obs_idx]

        experiment_hints = ExperimentHints(
            cycle=self._cycle,
            n_cycles=self._n_cycles,
            n_observed=int(self._obs_mask.sum()),
            n_total=self._n,
            budget_remaining=self._budget_remaining,
            batch_size=self._batch_size,
            last_hit_rate=self._last_cycle_hit_rate,
            observed_gene_names=observed_names,
        )

        # ── GP-mode: cross-target correlation + GP sigma ──────────────
        corr_matrix: list[list[float]] | None = None
        mean_gp_sigma = 0.0
        if self._use_gp and self._gp is not None and len(obs_idx) >= 2:
            Y_obs = self._Y[obs_idx]
            if Y_obs.shape[0] >= self._K + 1:
                C = np.corrcoef(Y_obs.T)
                C = np.nan_to_num(C, nan=0.0)
                corr_matrix = C.tolist()
            pool_idx = np.where(~self._obs_mask)[0]
            if len(pool_idx) > 0:
                _, sigma = self._gp.predict(self._X[pool_idx])
                mean_gp_sigma = float(sigma.mean())

        target_hints = TargetMetricsHints(
            target_names=self._target_names,
            per_target_recall=recall.tolist(),
            per_target_hits=hits.tolist(),
            total_hits=int(hits.sum()),
            thresholds=self._thresholds.tolist(),
            mean_gp_sigma=mean_gp_sigma,
            correlation_matrix=corr_matrix,
        )

        error_hints = (
            ErrorHints(last_action="", last_action_error=last_action_error)
            if last_action_error
            else None
        )

        # ── Mode-specific observation fields ─────────────────────────
        if self._use_gp:
            # Single-agent GP mode: ranked candidate pool, no specialist views
            candidates = self._build_candidate_pool()
            specialist_views = None
        else:
            # Multi-agent mode: per-target specialist views, no GP pool
            candidates = None
            specialist_views = self._build_specialist_views()

        return Observation(
            experiment=experiment_hints,
            target_metrics=target_hints,
            candidates=candidates,
            last_batch=self._last_batch_hints,
            error=error_hints,
            step_info=(self._cycle, self._n_cycles),
            specialist_views=specialist_views,
        )

    def _build_specialist_views(self) -> list[SpecialistViewHints]:
        """Build one SpecialistViewHints per target for multi-agent mode.

        Each view shows only that target's outcomes — specialists have no
        cross-target visibility, forcing them to reason from domain knowledge.
        A random pool sample is included so agents can reference real gene
        names; they may also nominate genes outside this sample.
        """
        obs_set = set(np.where(self._obs_mask)[0].tolist())
        pool_idx = np.where(~self._obs_mask)[0]

        # Draw a fixed random sample of unobserved genes to show specialists.
        # Using the episode RNG so the sample is reproducible per seed.
        sample_size = min(self._pool_sample_size, len(pool_idx))
        if sample_size > 0:
            sample_positions = self._rng.choice(len(pool_idx), sample_size, replace=False)
            pool_sample = [self._gene_names[pool_idx[i]] for i in sample_positions]
        else:
            pool_sample = []

        views: list[SpecialistViewHints] = []
        for k in range(self._K):
            hit_set_k = set(self._true_hits[k].tolist())

            # Genes observed so far that were hits for target k
            confirmed_hits = [
                self._gene_names[i] for i in sorted(obs_set) if i in hit_set_k
            ]

            # Hits and misses from the last batch only (for recency reasoning)
            last_hits: list[str] = []
            last_misses: list[str] = []
            if self._last_batch_hints is not None:
                for gene, idx in zip(
                    self._last_batch_hints.selected_genes,
                    self._last_batch_hints.selected_indices,
                ):
                    if idx in hit_set_k:
                        last_hits.append(gene)
                    else:
                        last_misses.append(gene)

            recall_k = len(confirmed_hits) / max(len(hit_set_k), 1)

            views.append(SpecialistViewHints(
                target_name=self._target_names[k],
                # phenotype / cell_type filled in by BiologyOrchestratorAgent
                # from SCREEN_BIOLOGY if available; left blank here so the env
                # stays dataset-agnostic.
                phenotype="",
                cell_type="",
                recall=recall_k,
                n_hits_found=len(confirmed_hits),
                n_total_hits=len(hit_set_k),
                confirmed_hit_genes=confirmed_hits,
                last_batch_hits=last_hits,
                last_batch_misses=last_misses,
                pool_sample=pool_sample,
                n_pool_total=len(pool_idx),
            ))

        return views
