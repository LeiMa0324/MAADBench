"""
main.py — Entry point for evaluating a MetaGPT-style MAS on SWE-bench Verified.

Usage:
    # From project root:
    PYTHONPATH=. python SWE-Verified/main.py

    # With options:
    PYTHONPATH=. python SWE-Verified/main.py --sample_size 10 --difficulty easy
"""

import sys
import os
from pathlib import Path
from utils import *

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ── Load .env and bootstrap ~/.metagpt/config2.yaml BEFORE metagpt imports ──
# MetaGPT calls Config.default() at module level, so the config file must
# exist before any metagpt import is triggered.
from dotenv import load_dotenv
import yaml as _yaml

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

_here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_here, "configs/config.yaml")) as _f:
    _proj_cfg = _yaml.safe_load(_f)

_llm = _proj_cfg.get("llm", {})
_metagpt_cfg = Path.home() / ".metagpt" / "config2.yaml"
_metagpt_cfg.parent.mkdir(exist_ok=True)
_metagpt_cfg.write_text(
    f"llm:\n"
    f"  api_type: \"{_llm.get('api_type', 'openai')}\"\n"
    f"  model: \"{_llm.get('model', 'gpt-4.1-mini')}\"\n"
    f"  api_key: \"{os.environ.get('OPENAI_API_KEY', '')}\"\n"
    f"  temperature: {_llm.get('temperature', 0.0)}\n"
)
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import json
import yaml
import wandb

from swe_dataloader import SWEDataLoader
from swe_mas import SWEOrchestrator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mas_framework.mas import Agent  # noqa (ensures mas_framework importable)


DIFFICULTIES = ["easy", "medium", "hard", "expert"]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MetaGPT MAS on SWE-bench Verified.")
    parser.add_argument("--dataset", type=str, default="SWE-Verified")
    parser.add_argument("--mas_arch", type=str, default="MetaGPT")
    parser.add_argument("--config_file", type=str, default="configs/config.yaml")
    parser.add_argument("--data_file", type=str, default="swe_bench_verified.jsonl")
    parser.add_argument("--sample_size", type=int, default=-1,
                        help="Total instances to sample (-1 = all).")
    parser.add_argument("--difficulty", type=str, default=None,
                        help="Filter by difficulty: easy / medium / hard / expert. "
                             "Default: run all difficulties.")
    parser.add_argument("--output_dir", type=str, default="traces")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()

    # ── Resolve paths relative to this file ────────────────────────────────────
    here = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(here, args.config_file)
    data_path   = os.path.join(here, args.data_file)
    output_dir  = os.path.join(here, args.output_dir)

    # ── Load config and MAS ────────────────────────────────────────────────────
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    mas = SWEOrchestrator.from_config(config_path)

    # ── wandb ──────────────────────────────────────────────────────────────────
    wandb.init(
        project="Lomas-SWE-Verified",
        config={
            "dataset": args.dataset,
            "mas_arch": args.mas_arch,
            "config_file": args.config_file,
            "sample_size": args.sample_size,
            "difficulty": args.difficulty,
            "seed": args.seed,
        },
        resume="allow",
    )

    # ── Load dataset (all instances; sampling is per-difficulty below) ──────────
    print(f"\nLoading dataset: {data_path}")
    loader = SWEDataLoader.from_jsonl(data_path, seed=args.seed)
    print(f"Total instances loaded: {len(loader)}")

    # ── Determine which difficulties to run ────────────────────────────────────
    if args.difficulty:
        difficulties_to_run = [args.difficulty]
    else:
        difficulties_to_run = DIFFICULTIES

    # ── Main loop ──────────────────────────────────────────────────────────────
    level_results = {}
    all_predictions = []
    predictions_dir = os.path.join(here, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    pred_path = os.path.join(predictions_dir, "predictions.jsonl")

    for difficulty in difficulties_to_run:
        instances = loader.by_difficulty(difficulty, sample_size=args.sample_size)
        if not instances:
            continue

        generated = 0
        print(f"\nRunning {len(instances)} '{difficulty}' instances...")

        for tested, item in enumerate(instances, start=1):
            print(f"\n{'━' * 60}")
            print(f"  [{difficulty}] {tested}/{len(instances)} — {item.instance_id}")
            print(f"  Repo   : {item.repo}")
            print(f"{'━' * 60}")

            context = json.dumps({
                "repo": item.repo,
                "base_commit": item.base_commit,  # ← 新增
                "hints_text": item.hints_text,
                "fail_to_pass": item.fail_to_pass,
                "pass_to_pass": item.pass_to_pass,  # ← 新增，reviewer需要
                "version": item.version,
            })

            prediction = mas.run(problem=item.problem_statement, instance_id=item.instance_id, context=context)
            all_predictions.append(prediction)
            with open(pred_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(prediction, ensure_ascii=False) + "\n")

            print(f"[{difficulty}] generation {generated}/{tested} = {generated/tested*100:.1f}%  ")

            wandb.log({
                f"{difficulty}_generated": generated,
                f"{difficulty}_rate": generated / tested * 100,
            })

            # Save trace
            trace_id = generate_trace_id(
                dataset=args.dataset,
                mas_arch=args.mas_arch,
                query_id=item.instance_id,
                temp=config["llm"].get("temperature", 0.0),
                label =""
            )
            trace_path = mas.save_execution_trace(
                config,
                question_data=item.to_dict(),
                output_dir=output_dir,
                trace_id=trace_id,
            )
            print(f"Trace saved: {trace_path}")

        level_results[difficulty] = (generated,  len(instances))

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n[Evaluation Summary]")
    total_gen = total_n = 0
    for diff, (gen,  n) in level_results.items():
        print(f"  {diff:8s}: generated {gen}/{n} ({gen/n*100:.1f}%)  ")
        total_gen += gen
        total_n   += n
    if total_n > 0:
        print(f"  {'overall':8s}: generated {total_gen}/{total_n} ({total_gen/total_n*100:.1f}%) ")

    print(f"\nPredictions saved: {pred_path}  ({len(all_predictions)} entries)")

