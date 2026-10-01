# EscapeRoom_v2 — MAST Failure Mode Coverage

A multi-agent escape-room benchmark for evaluating MAS pipelines under controlled
failure-mode injection. Rooms contain instruments (clock, thermometer, compass, scale),
clues, and traps; agents (`observer`, `clue_solver`, `item_manager`, `planner`) cooperate
through a fixed action sequence to escape. Injections corrupt either LLM outputs or
prompts to simulate the 7 MAST failure modes covered below.

## Directory Structure

```
EscapeRoom_v2/
├── README.md                  # this file
├── er_core.py                 # Room, PuzzleItem, Clue, Trap — environment model
├── er_action.py               # Action definitions (OBSERVE_*, SOLVE_*, APPLY_DELTA, …)
├── er_agents.py               # Agent role prompts + per-difficulty action/agent mappings
├── er_mas.py                  # Base MAS orchestrator
├── er_mas_verifier.py         # EROrchestrator — sequential MAS with verifier
├── er_mas_planner.py          # ERPlannerOrchestrator — planning variant for nightmare rooms
├── er_plan_room.py            # PlanningRoom — multi-puzzle rooms with DAG dependencies
├── er_plan_main.py            # PlanningRoom standalone entry point
├── er_injection.py            # Output/prompt injection strategies for each FM
├── er_verifier.py             # Verifier agent for failure detection
├── er_runner.py               # Main benchmark runner (CLI entry point)
├── DAG.py                     # DAG utility for planning rooms
├── silent_loud.py             # Auxiliary signal analysis helper
│
├── configs/                   # YAML configs (LLM provider, model, temperature)
│   ├── config.yaml            # default (gpt-4.1, T=0.6)
│   ├── config_claude.yaml
│   ├── config_qwen.yaml
│   ├── claude/                # per-temperature variants: config_0.0, 0.3, 0.6
│   ├── deepseek/
│   ├── gpt_41/
│   ├── gpt_54/
│   └── qwen/
│
├── scripts/                   # batch sweep shell scripts
│   ├── run_nightmare.sh       # claude + gpt_54 across temperatures
│   ├── run_nightmare_ds.sh    # deepseek variant
│   └── run_qwen_nightmare.sh
│
├── MAST_annotator/            # post-hoc failure-mode annotation
│   ├── FM_annotate.py         # rule + trace-based annotator (no LLM)
│   ├── MAST_FM.py
│   └── MAST_FM_MAPPING.md
│
├── plots/                     # paper figures and analysis
│   ├── escape.py              # escape-room schematic figure
│   ├── anomaly_propagation.py
│   ├── action_tabular_tsne.py
│   ├── execution_analysis/
│   └── exp_analysis/
│
├── data_artifacts/            # dataset bundles (MADBench-Eval, MADBench-full)
└── traces/                    # run outputs (auto-created per run)
    └── run_<ts>_<fm>_<model>_<temp>/
        ├── rooms.jsonl        # generated rooms for this run
        ├── <trace_id>.json    # one file per room
        ├── summary.csv        # aggregate metrics
        └── run.log
```

## Quickstart

### 1. Install & configure

Create a Python 3.11 conda environment and install dependencies from the repo root:

```bash
conda create -n Madbench python=3.11 -y
conda activate Madbench
pip install -r ../requirements.txt     # run from EscapeRoom_v2/, or pip install -r requirements.txt from repo root
```

Then set the API keys required by whichever config you plan to use:

```bash
export OPENAI_API_KEY=...              # for gpt_41 / gpt_54 configs
export ANTHROPIC_API_KEY=...           # for claude configs
export DEEPSEEK_API_KEY=...            # for deepseek configs
```

### 2. Run a small batch (no injection, default gpt-4.1-mini config)

```bash
cd EscapeRoom_v2
python -m EscapeRoom_v2.er_runner \
    --n-easy 5 --n-medium 3 --n-hard 2 \
    --config configs/config.yaml \
    --output-dir traces
```

Outputs land under `traces/run_<timestamp>_<model>_<temp>/`.


### 3. Replay a fixed room set (reproducibility)

```bash
python -m EscapeRoom_v2.er_runner \
    --rooms-file traces/rooms.jsonl \
    --config configs/config.yaml
```

### Common CLI flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--n-easy / --n-medium / --n-hard / --n-nightmare` | `0` | Room counts per difficulty |
| `--seed` | `42` | RNG seed for room generation |
| `--rooms-file` | — | Load rooms from JSONL instead of generating |
| `--config` | `configs/config.yaml` | YAML with `llm.provider`, `model`, `temperature`, `max_tokens` |
| `--mode` | `natural` | `natural` (message passing) or `structured` (JSON between agents) |
| `--fm-id` | — | e.g. `FM-1.1`, `FM-2.5` — see table |
| `--injection-type` | — | `output` (post-LLM corruption) or `prompt` (pre-LLM rewrite) |
| `--output-dir` | `traces` | Run-output root |
| `--dry-run` | off | Generate rooms only, skip MAS execution |
| `--verbose` | off | Verbose orchestrator logging |
─────────────────────────────────────────────────────────────────────────────┘