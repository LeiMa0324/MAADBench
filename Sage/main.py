"""
sage_runner.py — Command-line entry point for the SAGE simplification pipeline.

Usage:
    # Interactive mode (prompts for expression):
    python sage_runner.py

    # Direct expression argument:
    python sage_runner.py --expr "((x * 1) + (0 + y))"

    # Custom config and max steps:
    python sage_runner.py --expr "((x * 1) + (0 + y))" --config sage_config.yaml --max-steps 30

    # Save trace to file:
    python sage_runner.py --expr "((x * 1) + (0 + y))" --save-trace
"""

import sys
import os
# Add project root to path so local package imports work when running this file directly.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import argparse
import json
from typing import Optional
from Sage_Dataloader import *
import wandb

from sage_mas import SAGEOrchestrator
from mas_framework.mas import Agent
import yaml
from utils import *  # noqa

# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MAS on Sage datasets.")
    parser.add_argument("--dataset", type=str, default="Sage",
                        help="Dataset name used for trace_id generation (e.g.,  Sage).")
    parser.add_argument("--mas_arch", type=str, default="AutoGen",
                        help="MAS framework used for trace_id generation (e.g., AutoGen, LangGraph).")
    parser.add_argument("--config_file", type=str, default="configs/config.yaml",
                        help="Path to MAS config yaml.")
    parser.add_argument("--sample_size", type=int, default=-1,
                        help="Number of problems to sample for evaluation. -1 as no sample. ")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split to load (train/test).")
    parser.add_argument("--output_dir", type=str, default="traces",
                        help="Directory to save execution traces.")
    parser.add_argument("--seed", type=int, default=41,
                        help="Random seed for sampling.")
    parser.add_argument("--progress_file", type=str, default=None,
                        help="Path to JSONL progress file. Auto-generated if not set.")
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()

    mas = SAGEOrchestrator.from_config(args.config_file)

    wandb.init(
        project=f"Lomas-SAGE",
        config={
            "dataset": "SAGE",
            "mas_arch": "sequential",
            "config_file": args.config_file,
            "seed": args.seed,
            "split": args.split,
        },
        resume="allow",
    )

    print("\n--- Reload from CSV ---")
    loader2 = SAGEDataLoader.from_csv("tasks.csv")

    with open(args.config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    levels = ["easy", "medium", "hard"]
    level_results = {}

    for level in levels:
        problems = loader2.by_level(level)
        if not problems:
            continue

        correct = 0
        print(f"\nselecting {len(problems)} {level} problems...")
        for tested, item in enumerate(problems, start=1):
            print(f"  {item.e_id}: {item.ori_expression}")
            expr = item.ori_expression

            print(f"\n{'━' * 60}")
            print(f"  SAGE Simplification Pipeline")
            print(f"  Expression : {expr}")
            print(f"{'━' * 60}")

            result = mas.run(problem=expr)

            prediction = result["final_answer"]
            is_correct = item.simple_evaluate(result["final_answer"])

            if is_correct:
                correct += 1
                status = "✓ CORRECT"
            else:
                status = "✗ WRONG"

            print(f"Predicted: {prediction}, Ground Truth: {item.ground_truth} -> {status}")
            print(f"[{level}] Running accuracy: {correct}/{tested} = {correct/tested*100:.1f}%")
            running_accuracy = correct/tested*100
            level_percentage = tested/len(problems)
            wandb.log({
                "total": len(loader2),
                f"{level}_percent": level_percentage*100,
                f"{level}_accuracy": running_accuracy

            })
            label = "Normal" if is_correct else "Abnormal"

            # generate trace_id for file
            trace_id = generate_trace_id(
                dataset=args.dataset,
                mas_arch=args.mas_arch,
                query_id=item.e_id,
                label=label
            )
            trace_path = mas.save_execution_trace(config,
                                                  question_data=item.to_dict(), output_dir=args.output_dir,
                                                  trace_id=trace_id)
            print(f"Execution trace saved to: {trace_path}")

        level_results[level] = (correct, len(problems))

    print(f"\n[Phase 4: Evaluation Summary]")
    total_correct = 0
    total_problems = 0
    for level, (correct, n) in level_results.items():
        print(f"  {level:8s}: {correct}/{n} = {correct/n*100:.1f}%")
        total_correct += correct
        total_problems += n
    if total_problems > 0:
        print(f"  {'overall':8s}: {total_correct}/{total_problems} = {total_correct/total_problems*100:.1f}%")

