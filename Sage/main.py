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

import argparse
import json
import os
import sys
from typing import Optional
from Sage_Dataloader import *

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

    print("\n--- Reload from CSV ---")
    loader2 = SAGEDataLoader.from_csv("tasks.csv")

    correct = 0
    level = 'hard'
    problems = loader2.by_level("hard")
    print(f"selecting {len(problems)} {level} problems...")
    for item in problems:
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
        label = "Normal" if is_correct else "Abnormal"

        with open(args.config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        label = ''
        # generate trace_if for file
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

    print(f"[Phase 4: Evaluation Summary] \nTotal Problems: {len(loader2)}, correct: {correct}, Accuracy: {correct/len(problems) *100:.1f}%")

