# ACUA — Adaptive Computer Use Agents

A multi-agent orchestration system for automated computer use on desktop environments. Vision-language models coordinate specialist agents that execute tasks via browser actions or pyautogui commands, evaluated on the [OSWorld](https://os-world.github.io/) and [WebArena](https://webarena.dev/) benchmarks.

## Setup

```bash
# Core install
uv sync

# With OSWorld dependencies
uv sync --extra osworld

# With WebArena / BrowserGym dependencies
uv sync --extra webarena

# Complete install
uv sync --all-extras
```

### WebArena infrastructure

WebArena requires a running set of web services (shopping, forum, CMS, etc.). The setup scripts in `scripts/webarena/` automate this:

```bash
# On the server hosting the WebArena services
fish scripts/webarena/server_setup.fish

# On the client machine running the agent
fish scripts/webarena/client_setup.fish
```

See `scripts/webarena/` for OpenStreetMap container setup and server refresh scripts.

## Running experiments

### Single episode

Run a single agent-environment episode from the CLI:

```bash
python -m acua.experiments.loop \
    --env BrowserGymEnv \
    --gym-env-id browsergym/webarena.21 \
    --provider OpenAIProvider \
    --model gpt-4o-mini \
    --max-steps 15
```

Verbosity flags control log detail:

| Flag   | Level   | What's logged                                                  |
|--------|---------|----------------------------------------------------------------|
| _(none)_ | WARNING | Goal, actions, rewards, episode summary, errors              |
| `-v`   | INFO    | Step timing, obs metadata (URL, screenshot presence), thinking |
| `-vv`  | DEBUG   | Accessibility tree content, open tabs, focused element         |
| `-vvv` | TRACE   | Full HTML DOM                                                  |

### Sweep over multiple tasks

The sweep runner uses [Hydra](https://hydra.cc/) for configuration and runs `loop.run()` over a list of tasks, collecting results into a CSV and pandas summary table.

```bash
# Run with the default WebArena config
python -m acua.experiments.sweep --config-name webarena

# Override tasks and model from the CLI
python -m acua.experiments.sweep --config-name webarena \
    'task_ids=[0,1,2,3,4,5]' \
    provider.model=gpt-4o \
    max_steps=20

# Override output directory
python -m acua.experiments.sweep --config-name webarena \
    output_dir=results/my_experiment
```

Sweep configs live in `configs/`. Example (`configs/webarena.yaml`):

```yaml
env:
  name: BrowserGymEnv
  headless: true

provider:
  name: OpenAIProvider
  model: gpt-4o-mini
  max_tokens: 16384

agent:
  name: CUAAgent
  action_set: WEBARENA
  be_cautious: true

obs_flags:
  use_html: false
  use_ax_tree: true
  use_screenshot: true

max_steps: 15
task_prefix: browsergym/webarena
task_ids: [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
output_dir: results/sweep
```

Output structure:

```
results/sweep/
├── browsergym_webarena.21.log
├── browsergym_webarena.22.log
├── ...
└── results.csv
```

### Orchestrated multi-agent sweep

The `OrchestratorAgent` delegates to specialist sub-agents, each backed by a different provider/model. The orchestrator picks the best specialist per step and passes it an instruction.

```bash
python -m acua.experiments.sweep --config-name webarena-orchestrated
```

The orchestrated config (`configs/webarena-orchestrated.yaml`) defines the orchestrator's own provider and a list of sub-agents with descriptions:

```yaml
provider:
  name: OpenAIProvider
  model: gpt-4o              # orchestrator's routing model

agent:
  name: OrchestratorAgent
  action_set: WEBARENA
  sub_agents:
    - provider:
        name: OpenAIProvider
        model: gpt-4o-mini
      description: >-
        Fast, general-purpose web agent. Good at straightforward navigation,
        clicking buttons, filling forms, and reading page content.
    - provider:
        name: AnthropicProvider
        model: claude-sonnet-4-6
      description: >-
        Strong reasoning agent. Use for complex multi-step tasks that require
        careful planning or extracting information from dense pages.
```

Each sub-agent's `description` is exposed as the tool description in the orchestrator's tool list, guiding its routing decisions.

### OSWorld benchmark

OSWorld uses a multi-process runner driven by a YAML config:

```bash
python scripts/run.py --config configs/osworld.yaml
python scripts/run.py --config configs/osworld.yaml --domain shopping --num-envs 4
```

Results are written to `results/osworld/{domain}/{task_id}/` with logs, trajectories, screenshots, and recordings.

### Training an orchestrator with GRPO

Fine-tune a small LM (e.g. Qwen2.5-3B) to act as the orchestrator using GRPO (Group Relative Policy Optimization) on WebArena tasks. Requires a GPU with sufficient VRAM (24GB+ recommended with 4-bit quantization).

```bash
# Install training dependencies
pip install -e ".[training]"

# Run training (requires running WebArena infrastructure + W&B account)
python -m acua.training.train_orchestrator --config-name orchestrator

# Override model or generation count
python -m acua.training.train_orchestrator --config-name orchestrator \
    model.name=unsloth/Qwen2.5-7B-Instruct \
    grpo.num_generations=2

# Resume from checkpoint
python -m acua.training.train_orchestrator --config-name orchestrator \
    resume_from_checkpoint=results/train-orchestrator/checkpoint-100
```

The training pipeline:
- **Unsloth** loads the base model with 4-bit quantization and applies LoRA
- **TRL's GRPOTrainer** with `environment_factory` runs full WebArena episodes as the multi-turn tool-calling loop
- Each generation: the model picks a sub-agent tool, the sub-agent executes in WebArena, the observation is returned, and the model picks again — up to `max_steps`
- The reward signal is WebArena's binary task completion score (0 or 1)
- **W&B** logs training metrics, sample completions, and checkpoint artifacts

Config at `configs/train/orchestrator.yaml` controls the model, LoRA hyperparameters, GRPO settings, sub-agent definitions, and task set.

## Supported models

| Provider             | Models                                | Notes                              |
|----------------------|---------------------------------------|------------------------------------|
| `OpenAIProvider`     | `gpt-4o-mini`, `gpt-4o`, `gpt-5.4-*`  | Uses the Responses API             |
| `AnthropicProvider`  | `claude-sonnet-4-6`                   | Supports extended thinking budget  |
| `OpenRouterProvider` | Qwen, Kimi, Minimax, Grok variants    | Code-gen mode for non-tool models  |

All providers implement `AbstractProvider` and are interchangeable via config.

## Architecture

### Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Experiment Loop                          │
│                 (loop.py / sweep.py)                        │
│                                                             │
│  obs ──► Agent ──► Provider ──► LLM ──► action ──► Env      │
│   ▲                                                   │     │
│   └───────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

The system follows a layered design with clear abstractions at each boundary.

### Environments (`acua/envs/`)

**`AbstractEnv`** wraps a gymnasium environment with a unified interface:

- `reset()` → `Observation` — Reset and convert raw obs
- `step(action: str)` → `(Observation, float, bool, dict)` — Execute and convert
- `to_agent_observation(obs, info)` → `Observation` — _(abstract)_ Convert raw gym obs
- `python_to_env_action(action)` → env-native action — _(abstract)_ Convert action format

Two implementations:

- **`BrowserGymEnv`** — Wraps BrowserGym for web tasks (WebArena). Converts screenshots from numpy arrays to PNG bytes, flattens accessibility trees and DOM to text.
- **`OSWorldEnv`** — Wraps OSWorld's `DesktopEnv` for desktop tasks. Handles task configuration binding, sleep-after-execution timing, and evaluation.

### Observations (`acua/envs/observation.py`, `acua/envs/hints.py`)

**`Observation`** is an environment-agnostic dataclass with optional fields:

```python
@dataclass
class Observation:
    screenshot: bytes | None          # PNG screenshot
    html_dom: str | None              # HTML text
    accessibility_tree: str | None    # Accessibility tree text
    goal: MessageBuilder | None       # Task goal
    step_info: tuple[int, int] | None # (current_step, max_steps)

    # Typed hint groups (all optional)
    tabs: TabsHints | None
    error: ErrorHints | None
    focused_element: FocusedElementHints | None
    screenshot_som: ScreenshotSOMHints | None
    chat: ChatHints | None
    browser_internal: BrowserInternalHints | None
    osworld_task: OSWorldTaskHints | None
```

Hints are typed dataclasses grouped by semantic concern. Each environment populates only the hint groups it supports, making it clear which data is available for prompt construction.

### Providers (`acua/providers/`)

**`AbstractProvider`** defines the LLM interface:

```python
class AbstractProvider(ABC):
    def __init__(self, model_name, api_key, temperature, max_tokens): ...
    def __call__(self, payload: APIPayload) -> LLMOutput: ...
    def _call_api(self, payload) -> Any: ...       # abstract
    def _parse_response(self, response) -> LLMOutput: ... # abstract
```

**`APIPayload`** carries everything needed for an LLM call:

```python
@dataclass
class APIPayload:
    messages: list[MessageBuilder] | None  # Conversation history
    tools: list[dict] | None               # Tool schemas (OpenAI function format)
    tool_choice: str | None                # "none" | "auto" | "any" | "required"
    force_call_tool: str | None            # Force a specific tool
    reasoning_effort: str | None           # "low" | "medium" | "high"
```

**`LLMOutput`** is the structured response:

```python
@dataclass
class LLMOutput:
    raw_response: Any              # Provider-specific raw response
    think: str                     # Chain-of-thought / reasoning text
    action: str | None             # Pre-parsed action (if provider extracts it)
    tool_calls: ToolCalls | None   # Structured tool calls from the LLM
```

**`ToolCalls`** contains a list of **`ToolCall`** objects, each with a `name`, `arguments` dict, the `raw_call` from the API, and a `tool_response` slot for echoing results back into the conversation.

**`MessageBuilder`** is the provider-agnostic message format, supporting text, images, and tool call responses:

```python
msg = MessageBuilder.user() \
    .add_text("Click the submit button") \
    .add_image("data:image/png;base64,...")
```

### Agents (`acua/agents/`)

**`AbstractAgent`** defines the agent interface:

```python
class AbstractAgent(ABC):
    def get_python_action(self, obs: Observation) -> str: ...  # abstract
    def reset(self): ...
```

**`CUAAgent`** is the primary agent. It uses tool calling to interact with the environment:

- Manages conversation history via **`StructuredDiscussion`** (observation windowing, summarization of dropped groups)
- Builds prompts using the dynamic prompting framework (`SystemPrompt`, `ObservationPrompt`, `ActionPrompt`, `Think`, `BeCautious`)
- Converts the environment's `ActionSet` into tool schemas so the LLM calls actions as tools
- Extracts actions from tool calls, `<action>` tags, or code blocks (with fallback chain)

**`OrchestratorAgent`** delegates to specialist sub-agents:

- Has its own provider for decision-making (routing)
- Each sub-agent is defined by a **`SubAgentSpec`** (provider + description) and exposed as a tool (`call_{model_name}`)
- The orchestrator observes the screen, picks a specialist, and passes an instruction
- The chosen sub-agent (a `CUAAgent` internally) produces the final environment action
- Only one sub-agent per provider model is allowed

```python
orch = OrchestratorAgent(
    provider=router_provider,
    sub_agents=[
        SubAgentSpec(
            provider=qwen_provider,
            description="GUI specialist — clicking, typing, visual grounding.",
        ),
        SubAgentSpec(
            provider=claude_provider,
            description="Code specialist — terminal commands, scripting, text tasks.",
        ),
    ],
)
```

### Prompt framework (`acua/prompts/dynamic_prompting.py`)

Prompt elements are composable building blocks that render into `MessageBuilder` messages:

- **`ObservationPrompt`** — Renders the current observation (tabs, HTML, accessibility tree, error, focused element, screenshot, step counter). On the final step, injects urgency text encouraging the model to submit an answer.
- **`GoalInstructions`** / **`ChatInstructions`** — Render the task goal or chat history.
- **`ActionPrompt`** — Describes available actions and parses `<action>` tags from responses.
- **`Think`** — Encourages chain-of-thought reasoning via `<think>` tags.
- **`BeCautious`** — Instructs the agent to verify before submitting.
- **`Trunkater`** / **`fit_tokens`** — Progressive prompt shrinking to fit token budgets.

Observation flags (`ObsFlags`) control which components are included (HTML, accessibility tree, screenshot, SOM, tabs, errors, etc.).

### Action sets (`acua/envs/action_sets.py`)

Actions are plain Python functions with type annotations and docstrings, automatically converted to tool schemas via `function_to_tool_schema()`:

- **`WEBARENA`** — Browser actions: `click`, `fill`, `scroll`, `hover`, `goto`, `keyboard_press`, tab management, `send_msg_to_user` (stop), `report_infeasible`.
- **`OSWORLD`** — Desktop actions: `left_click`, `right_click`, `double_click`, `type_text`, `key`, `scroll`, `mouse_move`, `left_click_drag`, `wait`, `done`, `fail`.

### Episode flow

1. **Reset** — Environment resets, returns initial `Observation` with goal and screenshot.
2. **Loop** (up to `max_steps`):
   - Agent sees observation (with `step_info` indicating progress).
   - Agent's provider generates a tool call (environment action).
   - Action is executed in the environment.
   - Environment returns new observation, reward, and termination signal.
   - Observation history is managed (windowing + summarization).
3. **Termination** — Episode ends on environment signal, max steps, or agent's stop action.

## Project structure

```
acua/
├── agents/
│   ├── agent.py                 # AbstractAgent
│   ├── cua_agent.py             # CUAAgent + StructuredDiscussion
│   └── orchestrator_agent.py    # OrchestratorAgent + SubAgentSpec
├── envs/
│   ├── env.py                   # AbstractEnv
│   ├── observation.py           # Observation dataclass
│   ├── hints.py                 # Typed hint dataclasses
│   ├── action_sets.py           # ActionSet enum (WEBARENA, OSWORLD)
│   ├── utils.py                 # function_to_tool_schema
│   ├── browsergym/
│   │   ├── env.py               # BrowserGymEnv
│   │   ├── patches.py           # Runtime patches for webarena evaluation
│   │   └── tools.py             # WebArena action functions
│   └── osworld/
│       ├── env.py               # OSWorldEnv
│       └── tools.py             # OSWorld action functions
├── prompts/
│   ├── dynamic_prompting.py     # Prompt elements (ObservationPrompt, etc.)
│   └── parsing.py               # XML/code block parsing utilities
├── providers/
│   ├── provider.py              # AbstractProvider
│   ├── util.py                  # MessageBuilder, APIPayload, LLMOutput, ToolCall
│   ├── openai_provider.py       # OpenAIProvider (Responses API)
│   ├── anthropic_provider.py    # AnthropicProvider (Messages API)
│   └── openrouter_provider.py   # OpenRouterProvider
├── experiments/
│   ├── loop.py                  # Single-episode CLI runner
│   └── sweep.py                 # Hydra-powered multi-task sweep
├── training/
│   ├── train_orchestrator.py    # GRPO fine-tuning entry point
│   ├── orchestrator_env.py      # TRL environment_factory for GRPO training
│   └── callbacks.py             # W&B artifact checkpoint callback
└── runner.py                    # OSWorld multi-process runner
configs/
├── webarena.yaml                # WebArena sweep config
├── webarena-orchestrated.yaml   # Orchestrated multi-agent sweep config
├── osworld.yaml                 # OSWorld benchmark config
└── train/
    └── orchestrator.yaml        # GRPO training config
scripts/
└── webarena/                    # WebArena infrastructure setup scripts
tests/
├── test_dynamic_prompting.py    # Prompt framework tests
├── test_browsergym_agent.py     # BrowserGym integration tests
└── test_*_tool_call.py          # Provider tool-calling tests
```
