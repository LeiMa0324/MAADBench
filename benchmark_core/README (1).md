# benchmark_core

benchmark_core is a multi-agent escape-room benchmark for studying failures in
agent collaboration. An observer identifies clues and instruments, a clue solver
produces a numeric delta, and an item manager applies that delta to an instrument.

## Clue domains

The benchmark supports two interchangeable clue domains:

- `gsm-hard`: mathematical word problems with numeric answers.
- `livecodebench`: chained Python execution problems with numeric answers.

Both implement the same domain contract and use the same room, agent,
`APPLY_DELTA`, unit-tool, trace, and failure-analysis pipeline.

```text
benchmark_core/domains/
├── base.py
├── registry.py
├── gsm_hard/
│   ├── domain.py
│   ├── dataloader.py
│   ├── problem_source.py
│   ├── prompts.py
│   ├── distractors.py
│   ├── verifier.py
│   └── runner.py
└── livecodebench/
    ├── domain.py
    ├── problem_source.py
    ├── prompts.py
    ├── distractors.py
    ├── verifier.py
    ├── runner.py
    ├── scripts/
    │   ├── fetch_number_gt.py
    │   ├── make_chains.py
    │   └── make_distractors.py
    └── data/
        └── *.jsonl
```

## Run

From the repository root:

```bash
python -m main.runner \
    --n-easy 5 \
    --clue-domain gsm-hard \
    --config configs/config.yaml
```

Switch to LiveCodeBench through the same runner:

```bash
python -m main.runner \
    --n-easy 5 \
    --clue-domain livecodebench \
    --config configs/config.yaml
```

Domain-specific entry points are also available:

```bash
python -m benchmark_core.domains.gsm_hard.runner --n-easy 5
python -m benchmark_core.domains.livecodebench.runner --n-easy 5
```

Enable the unit-conversion tools for `APPLY_DELTA` with:

```bash
python -m main.runner \
    --n-easy 5 \
    --clue-domain livecodebench \
    --unit-tools enabled \
    --max-unit-tool-calls 3 \
    --config configs/config.yaml
```

The default is `--unit-tools disabled` unless `system.unit_tools` is enabled in
the YAML configuration.

Generated room files use
`domain_{level}_{num}_..._{timestamp}.jsonl`, for example
`livecodebench_easy_5_hard_2_20260905_210000.jsonl`.

Outputs are separated by type:

```text
output/
├── rooms/
│   └── domain_level_num_..._timestamp.jsonl
└── traces/
    └── run_timestamp_domain_tool-or-no_tool_model_temperature.../
        ├── *.json
        ├── summary.csv
        └── run.log
```

## Replay rooms

```bash
python -m main.runner \
    --rooms-file output/rooms/gsm-hard_easy_5_20260905_210000.jsonl \
    --config configs/config.yaml
```

Each serialized clue stores its `domain` and `problem_id`. When replaying rooms,
the runner selects the domain-specific prompts and verification policy from the
room itself.

## Main options

| Option | Default | Description |
|--------|---------|-------------|
| `--clue-domain` | `gsm-hard` | `gsm-hard` or `livecodebench` |
| `--unit-tools` | config/off | Enable or disable unit tools |
| `--max-unit-tool-calls` | `3` | Maximum tool executions per `APPLY_DELTA` |
| `--n-easy` | `0` | Number of easy rooms |
| `--n-medium` | `0` | Number of medium rooms |
| `--n-hard` | `0` | Number of hard rooms |
| `--n-nightmare` | `0` | Number of planning rooms |
| `--rooms-file` | — | Replay rooms from JSONL |
| `--config` | `configs/config.yaml` | LLM YAML configuration |
| `--output-dir` | `output` | Output root containing `rooms/` and `traces/` |
| `--fm-id` | — | Failure mode to inject |
| `--injection-type` | — | `output` or `prompt` injection |

## Tests

```bash
python -m unittest \
    benchmark_core.test_domains \
    benchmark_core.test_unit_tools -v
```
