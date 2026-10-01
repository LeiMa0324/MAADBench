# MADBench

MADBench evaluates multi-agent systems in controlled EscapeRoom tasks, records every
agent action as a trace, and evaluates anomaly detectors over those traces. The
project supports GSM-Hard and LiveCodeBench clue domains, runs with or without
unit-conversion tools, and includes tabular, graph, sequence, and LLM-judge AD
methods.

## Quick Start

Follow this order: create the shared Python environment, prepare rooms and
traces, then run anomaly detection.

### 1. Create the Python environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Python 3.12 and this one environment are used for both trace generation and all
AD methods.

### 2. Prepare rooms and traces

**Use the given traces.** Download the published dataset, then skip directly to
step 3. No model-provider API key is needed:

```bash
hf download hww123/MAADBench-full \
  --repo-type dataset \
  --local-dir output/traces
```

**Generate your own traces.** First configure the key required by the selected
model. For example, `configs/gpt_41/config_0.0.yaml` uses `OPENAI_API_KEY`:

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=...
```

Then choose one command:

**A. Run with an existing room file.** List the supplied room files under
`output/rooms/`, select one `.jsonl` file, and pass it to `--rooms-file`:

```bash
ls output/rooms

python -m main.runner \
  --rooms-file output/rooms/<your-rooms-file>.jsonl \
  --config configs/gpt_41/config_0.0.yaml \
  --output-dir output
```

**B. Generate rooms and run traces in one command:**

```bash
python -m main.runner \
  --n-easy 5 --n-medium 5 \
  --clue-domain gsm-hard \
  --config configs/gpt_41/config_0.0.yaml \
  --output-dir output
```

Both A and B save trace JSON files in `output/traces/`; B also saves its generated
room file in `output/rooms/`. The downloaded traces use the same default
`output/traces/` location. See the
[published MAADBench-full dataset](https://huggingface.co/datasets/hww123/MAADBench-full)
for the downloadable files.

### 3. Run anomaly detection

Pass the trace root with `--path-traces`. The folder may contain trace
subdirectories for multiple models; AD searches it recursively. Use
`output/traces` for the traces generated or downloaded above.

Run a small smoke test first:

```bash
bash benchmark_AD/run_methods.sh \
  --path-traces output/traces \
  --methods random
```

To run every available AD method, omit `--methods random`:

```bash
bash benchmark_AD/run_methods.sh --path-traces output/traces
```

For traces stored elsewhere, replace `output/traces` with that folder's path.

Each run writes predictions and metrics below `output/ad/`.

## What is in this repository

```text
benchmark_core/     EscapeRoom environment, agents, verifier, injection logic
main/               Trace runner and multi-model matrix launchers
configs/            Provider/model/temperature YAML configurations
output/             Generated rooms, traces, run summaries, and AD results
benchmark_AD/       Anomaly detection package, batch launcher, method code
benchmark_analyze/  Analysis scripts for trace and room-level results
paper_resources/    Scripts and artifacts used to prepare paper figures
data_artifacts/     Dataset bundles and derived data
```

## The workflow

```text
config + room generation
          ↓
main.runner / main.run_trace_matrix
          ↓
output/traces/<run>/*.json + summary.csv
          ↓
benchmark_AD/run_methods.sh
          ↓
output/ad/<batch>/<dataset>/<tooling>/<level>/<method>/seed_<n>/predictions.csv
          ↓
benchmark_analyze/ and benchmark_AD/plots/
```

Each trace JSON includes the run configuration, room state, per-action inputs and
outputs, verification result, failure report, and final `escaped` result. It is the
source of truth for AD input and post-hoc analysis.

## Installation and credentials

Python 3.12 is the supported interpreter because one environment contains PyTorch,
TensorFlow (DevNet), graph libraries, sentence embeddings, and Gemma dependencies.
Install only the repository-root [requirements.txt](requirements.txt); the files in
`benchmark_AD/` point back to it for compatibility.

Choose a configuration under `configs/`, copy `.env.example` to `.env`, and fill
the matching provider key before starting a real model run:

```bash
cp .env.example .env
# Edit .env and set one of OPENAI_API_KEY, ANTHROPIC_API_KEY, or DEEPSEEK_API_KEY.
```

`benchmark_core/LLMs.py` loads `.env` automatically. Only set the key required by
the YAML configuration you run. For Gemma judge runs, set `HF_TOKEN` when Hugging
Face access requires it.

On NVIDIA hardware, install the PyTorch wheel matched to the CUDA version before
installing `requirements.txt`. On macOS, XGBoost may require `brew install libomp`.

## Generate traces

Run one small experiment:

```bash
python -m main.runner \
  --n-easy 5 --n-medium 3 --n-hard 2 \
  --clue-domain gsm-hard \
  --config configs/gpt_41/config_0.0.yaml \
  --unit-tools enabled \
  --output-dir output
```

Useful options:

| Option | Meaning |
|---|---|
| `--clue-domain gsm-hard\|livecodebench` | Source domain for room clues |
| `--unit-tools enabled\|disabled` | Run with or without audited conversion tools |
| `--fm-id FM-…` | Inject one configured failure mode |
| `--injection-type output\|prompt` | Corrupt model output or rewrite the prompt |
| `--rooms-file FILE.jsonl` | Replay a fixed room set |
| `--dry-run` | Generate rooms without calling a model |
| `--wandb-mode disabled` | Disable Weights & Biases tracking |

The run creates:

```text
output/
  rooms/<domain>_...jsonl
  traces/run_<timestamp>_<domain>_<tool|no_tool>_<model>_<temperature>/
    <trace_id>.json
    summary.csv
    run.log
```

For a matrix over the configured models, temperatures, and supplied room files:

```bash
python -m main.run_trace_matrix --wandb-mode disabled
```

The default room files are declared in `main/run_trace_matrix.py`. Supply
`--rooms-file` one or more times to choose another fixed set.

## Run anomaly detection

`benchmark_AD` discovers trace JSON recursively under `output/traces/`. Both the
current flat run layout and archived `<model>/<domain>/run_...` layout are supported.
Do not put summaries, room JSONL files, or old prediction CSVs in a directory that
you pass as the trace root.

Run all registered methods for the available GSM-Hard tool traces:

```bash
bash benchmark_AD/run_methods.sh \
  --datasets EscapeRoom_v3_gsm_hard \
  --trace-types tool
```

Start with a lightweight smoke run:

```bash
bash benchmark_AD/run_methods.sh --methods random
```

Useful controls:

| Option | Meaning |
|---|---|
| `--methods random random_forest ...` | Select methods; default is all registered methods |
| `--datasets ...` | Select GSM-Hard and/or LiveCodeBench traces |
| `--trace-types tool no_tool` | Select MAS tool mode |
| `--seeds 1 2 3 4 5` | Repeat experiment splits |
| `--path-traces PATH` | Use a trace root other than `output/traces` |
| `--output-dir PATH` | Put AD artifacts outside `output/ad` |
| `--epochs N` | Shorten neural training for a smoke check |
| `--dry-run` | Write the batch plan without running methods |
| `--skip-llm-inference` | Export Gemma inputs but do not download/run Gemma |

The batch launcher runs methods sequentially, keeps going when one job fails, and
returns nonzero at the end if any job failed. It requires the activated Python 3.12
environment, or `PYTHON=/path/to/.venv/bin/python`.

## Results and predictions

Every batch gets a new directory:

```text
output/ad/batch_<UTC timestamp>_<id>/
  batch_manifest.json       # complete command, status, duration, log per job
  batch_summary.csv         # final status, metrics, and predictions path per job
  EscapeRoom_v3_gsm_hard/tool/action/random_forest/seed_1/
    config.json
    train_manifest.csv
    test_manifest.csv
    metrics.csv
    predictions.csv
    run.log
    data/                    # tabular feature artifacts, when applicable
```

`predictions.csv` is the common result contract. It contains `gt_label`,
`pred_label`, and `pred_score`, plus `source_file`, `action_id`, `puzzle_id`,
`step_num`, model, temperature, room metadata, and stable IDs. Join a prediction
back to its trace by `(source_file, action_id, step_num)`.

Action-level methods predict individual actions. Trace-level methods predict whole
traces; do not compare their raw row counts or action metrics as if they were the
same task. The saved train/test manifests record the exact room-disjoint split.

The default threshold policy avoids using test labels: calibrated classifiers use
probability `>= 0.5`, while score-only detectors select the configured top
`--anomaly-fraction` (default `0.1`). `--threshold-policy oracle` retains the old
research-only protocol that uses the true test anomaly count.

## Gemma as judge

Gemma uses the same environment and is included in the AD batch. Its shared input
split is exported under `.../gemma4/seed_<n>/data/`, then each requested variant
writes its own `predictions.csv` and `metrics.csv` underneath that run directory.

```bash
bash benchmark_AD/run_methods.sh \
  --methods gemma4 \
  --datasets EscapeRoom_v3_gsm_hard \
  --trace-types tool \
  --llm-models gemma-4-E2B
```

The first inference downloads the selected model. Ensure the selected model is
available to the Hugging Face account represented by `HF_TOKEN` and that the
machine has enough memory.

## Analysis and tests

Use scripts in `benchmark_analyze/` for aggregate room, action, and failure-mode
analysis. `benchmark_AD/plots/` contains the AD paper tables and figures; its input
data and plotting instructions live beside those scripts.

Run the AD validation suite:

```bash
python -m pytest benchmark_AD/tests -q
```

The test suite covers trace discovery, room-disjoint splits, prediction-to-trace
joins, batch failure isolation, Gemma data export, and lightweight method
integration. It does not download Gemma weights or call external model APIs.

## Scope notes

The runnable AD benchmark is the registry in `benchmark_AD/registry.py`. The
`cnn/`, `tspulse/`, and `gemma3_1b.py` folders are archival experimental code that
references local modules absent from this checkout, so they cannot be made runnable
by installing packages alone and are intentionally not selected by `--methods all`.

For method-specific feature processing, labels, thresholds, and output details, see
[benchmark_AD/README.txt](benchmark_AD/README.txt). For EscapeRoom internals, see
[benchmark_core/README.md](benchmark_core/README.md).
