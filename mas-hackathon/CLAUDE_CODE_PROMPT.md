# Claude Code Prompt: Agents-of-Biology Adaptation

You are working in the repo:
- ai/ (adapted from AgentOrchestra / ACUA, currently web/desktop oriented)
- ExperimentalDesign/ and maehd/ (gene perturbation experiments, metrics, datasets)

Goal
Adapt the ai project so agents operate on gene-perturbation experiments rather than browser/desktop tasks. Implement:
1) A new Observation schema aligned with experiment results.
2) A custom environment that runs experiment cycles given candidate selections.
3) Tool functions + a new ActionSet for experiment actions.
4) A concrete agent subclass (like CUAAgent) specialized for this domain.
5) dynamic_prompting updates to render experiment observations and drive tool calls.

Key references
- ai/envs/observation.py (current observation dataclass)
- ai/envs/hints.py (typed hint dataclasses)
- ai/envs/env.py (AbstractEnv)
- ai/envs/action_sets.py (ActionSet enum)
- ai/agents/agent.py and ai/agents/cua_agent.py (agent interface and example)
- ai/prompts/dynamic_prompting.py (ObservationPrompt, ActionPrompt)
- maehd/experiment.py (active learning loop, GP fitting, metrics)
- maehd/agents/llm_agent.py (decision format for LLM policy selection)
- maehd/agents/llm_policy.py (candidate table format for direct gene selection)
- maehd/datasets/genedisco.py and maehd/datasets/synthetic.py (data loaders)

Implementation requirements

1) Observation schema (ai/envs/observation.py)
- Replace or extend Observation to represent experiment state, not UI state.
- Add new typed hint dataclasses in ai/envs/hints.py, for example:
  - ExperimentHints: cycle, n_cycles, n_observed, n_total, budget_remaining, batch_size, last_hit_rate
  - TargetMetricsHints: target_names, per_target_recall, per_target_hits, total_hits, thresholds, mean_gp_sigma, correlation_matrix
  - CandidateSummary: gene_name, gene_index, max_poh, mean_sigma, per_target_poh (and optionally mu/sigma)
  - LastBatchHints: selected_genes, selected_indices, was_hit_any, per_target_hits
- Observation should include these new fields plus goal (MessageBuilder or text) and step_info.
- Keep error info (last_action_error) available so prompts can show invalid actions.

2) Custom experiment environment
- Put all new env-related files in a dedicated subfolder under ai/envs/, similar to ai/envs/browsergym.
  Suggested path: ai/envs/biology/ with __init__.py, env.py, tools.py (and any helpers).
- Implement the environment class in ai/envs/biology/env.py.
- It should manage one experiment episode with a gym-like interface:
  - reset() returns initial Observation
  - step(action: str) returns (Observation, reward, done, info)
- Use maehd.models.gp.MultiOutputGP to fit and predict; use maehd.metrics for recall/hits.
- Data loading:
  - Support synthetic datasets via maehd.datasets.synthetic (e.g., pathway_multitarget).
  - Optionally support real GeneDisco via maehd.datasets.genedisco.load_multitarget.
- Internal state should track obs_mask, cycle, budget_remaining, thresholds, last_cycle_hit_rate.
- Each step should:
  - Parse the action string (safe ast parse) to detect select_genes(...) or select_indices(...).
  - Validate gene names/indices against the current candidate pool.
  - Update obs_mask, recompute per-target recall and hits, last_cycle_hit_rate.
  - Build a fresh candidate pool for the next cycle (top N by max_poh, with per-target poh and sigma).
  - Return reward (e.g., new_hits / batch_size) and done when budget or cycles are exhausted.
- Add info dict with selected genes, hits, and any parsing errors.

3) Tools and ActionSet
- Add tool function stubs with docstrings + type annotations in ai/envs/biology/tools.py:
  - select_genes(genes: list[str])
  - select_indices(indices: list[int])  (optional)
  - finish(summary: str)  (optional early stop)
- Add ActionSet.BIOLOGY (or similar) in ai/envs/action_sets.py with these tools.
- Ensure to_tool_dict() returns tool schemas for the new action set.

4) New agent subclass
- Add ai/agents/biology_agent.py (or similar) as a concrete AbstractAgent.
- Use provider tool calling like CUAAgent but with the new ObservationPrompt.
- Default ObsFlags should show experiment state + candidate pool, not HTML/AX/screenshot.
- Agent should:
  - Build a system prompt with clear instructions for selecting genes from the pool.
  - Encourage exact gene name matching from the candidate list.
  - Support multi-action if you think it is useful (otherwise keep single action).

5) dynamic_prompting updates
- Extend or add a new ObservationPrompt for experiment observations:
  - Show cycle and budget status
  - Show per-target recall/hits/thresholds
  - Show candidate pool as a compact table (gene name, max_poh, mean_sigma, per-target poh)
  - Show last batch outcomes (selected genes, hit flags)
- Add ObsFlags fields to control visibility: use_experiment_state, use_candidate_pool, max_candidates, show_correlation, show_last_batch, etc.
- Keep the legacy web/desktop prompts intact if possible; otherwise clearly separate logic based on Observation fields.

Optional but helpful
- Add a small runnable example (ai/examples/run_biology_env.py) that:
  - Loads a synthetic dataset
  - Instantiates the environment and agent
  - Runs 2-3 steps and prints observations

Constraints
- Keep changes scoped to ai/ for agent framework, and import from maehd/ for experiment logic.
- Avoid adding heavy dependencies beyond what is already in maehd/ and ai/.
- Use ASCII-only text in new files.

Deliverables
- Updated Observation + hints
- New environment class
- New tools + ActionSet entry
- New agent class
- Updated dynamic_prompting
- (Optional) example script

Acceptance
- Running the example should show candidate pool prompts and valid select_genes actions.
- No browsergym/osworld dependencies required for the biology environment.
