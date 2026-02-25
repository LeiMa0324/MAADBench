import os
import random
import sys
import json
import argparse
from collections import defaultdict
from tqdm import tqdm
import wandb

# Add project root to path so local package imports work when running this file directly.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
print(PROJECT_ROOT)

from GSM_hard.GSM_dataloader import *  # noqa
from utils import *  # noqa
from mas_framework.mas import *


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MAS on GSM-hard datasets.")
    parser.add_argument("--dataset", type=str, default="GSM-hard",
                        help="Dataset name used for trace_id generation (e.g.,  GSM-hard).")
    parser.add_argument("--mas_arch", type=str, default="AutoGen",
                        help="MAS framework used for trace_id generation (e.g., AutoGen, LangGraph).")
    parser.add_argument("--config_file", type=str, default="configs/default_config.yaml",
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


def load_progress(progress_file):
    """Load completed problem records from JSONL progress file."""
    completed = {}
    if os.path.exists(progress_file):
        with open(progress_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    completed[record["problem_id"]] = record
    return completed


def append_progress(progress_file, record):
    """Append a completed problem record to progress file (fsync for crash safety)."""
    with open(progress_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def main():
    args = parse_args()

    dataset = args.dataset
    mas_arch = args.mas_arch
    config_file = args.config_file
    sample_size = args.sample_size

    # ================ Loading dataset ================
    problems = GSM_DataLoader.load_from_huggingface(
        split=args.split,
    )

    # Sample problems
    random.seed(args.seed)
    if sample_size == -1 or sample_size > len(problems):
        selected_problems = problems
    else:
        selected_problems = random.sample(problems, sample_size)

    # ================ Progress tracking ================
    progress_file = args.progress_file
    if progress_file is None:
        os.makedirs(args.output_dir, exist_ok=True)
        progress_file = os.path.join(
            args.output_dir,
            f"progress.jsonl",
        )

    hist_completed = load_progress(progress_file)

    remaining = [
        p for p in selected_problems
        if p.problem_id not in hist_completed
        or hist_completed[p.problem_id].get("has_parse_error")
    ]

    # ================ Wandb ================
    wandb.init(
        project=f"Lomas-{dataset}",
        config={
            "dataset": dataset,
            "mas_arch": mas_arch,
            "config_file": config_file,
            "sample_size": sample_size,
            "seed": args.seed,
            "split": args.split,
            "total_problems": len(selected_problems),
        },
        resume="allow",
    )

    print(f"[Phase 1: Dataset Loading] {args.dataset} dataset, total problems: {len(problems)}, selected {len(selected_problems)} problems for evaluation")
    if hist_completed:
        print(f"  Resumed from progress file: {len(hist_completed)} done, {len(remaining)} remaining")
    print()

    # ================ Create MAS ================
    if args.mas_arch =='AutoGen':
        mas = AutoGenMAS.from_config(config_file)
    elif args.mas_arch =='LangGraph':
        mas = LangGraphMAS.from_config(config_file)
    else:
        raise NotImplementedError

    print(f"[Phase 2: Creating MAS] {args.mas_arch} MAS architecture, {mas.architecture}, agents: {[role for role, agent in  mas.agents.items()]}\n")



    # ================ Evaluating ================
    print(f"[Phase 3: Evaluation] start ... \n")
    correct = 0
    parsed_errors = 0
    for i, math_problem in enumerate(tqdm(remaining, desc="problems")):
        print(f"\nProblem ID: {math_problem.problem_id}")
        print(f"Question: {math_problem.problem}")
        question_data = {
            "problem_id": math_problem.problem_id,
            "question": math_problem.problem,
            "gold_answer": math_problem.answer
        }

        # Simplify problem for agents
        simplified = GSM_DataLoader.simplify_problem(math_problem.problem)

        try:
            # MAS Solve
            result = mas.run(simplified)
            predicted = result.get("final_answer", None)
            ground_truth = GSM_DataLoader.extract_numerical_answer(math_problem.answer)


            # Simple verification
            is_correct = GSM_DataLoader.simple_evaluate(
                ground_truth=ground_truth,
                predicted=predicted
            )

            if is_correct:
                correct += 1
                status = "✓ CORRECT"
            else:
                status = "✗ WRONG"

            # Check for parse errors in trace
            has_parse_error = any(
                step.get("output", {}).get("content", {}).get("error") == "Failed to parse"
                for step in result.get("trace", [])
            )
            if has_parse_error:
                print(f"  ⚠ Parse error detected")
                parsed_errors += 1

            print(f"Predicted: {predicted}, Ground Truth: {math_problem.answer} -> {status}")
            label = "Normal" if is_correct else "Abnormal"
            label = label+"_parseError" if has_parse_error else label



            with open(config_file, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            # generate trace_if for file
            trace_id = generate_trace_id(
                dataset=dataset,
                mas_arch=mas_arch,
                query_id=math_problem.problem_id,
                temp=config["llm"].get("temperature", 0.0),
                label= label
            )
            trace_path = mas.save_execution_trace(config,
                question_data = question_data, output_dir=args.output_dir, trace_id=trace_id)
            print(f"Execution trace saved to: {trace_path}")

            # Record progress
            append_progress(progress_file, {
                "problem_id": math_problem.problem_id,
                "is_correct": is_correct,
                "predicted": predicted,
                "ground_truth": str(ground_truth),
                "trace_id": trace_id,
                "has_parse_error": has_parse_error,
            })

            wandb.log({
                "total": len(remaining),
                "progress_pct": (i+1)  / len(remaining) * 100,
                "processed": (i + 1) ,
                "correct_num": correct,
                "parse_error_num": parsed_errors,
                "accuracy": correct/(i+1)  *100
            })

        except Exception as e:
            print(f"Error: {e}")
            # Record failed problems too so they are not re-run
            append_progress(progress_file, {
                "problem_id": math_problem.problem_id,
                "is_correct": False,
                "predicted": None,
                "ground_truth": None,
                "error": str(e),
            })
            wandb.log({
                "total": len(remaining),
                "progress_pct": (i+1)  / len(remaining) * 100,
                "processed": (i + 1),
                "correct_num": correct,
                "parse_error_num": parsed_errors,
                "accuracy": correct/(i+1) *100
            })


    print(f"[Phase 4: Evaluation Summary] \nTotal Problems: {len(remaining)}, correct: {correct}, Accuracy: {correct/len(remaining) *100:.1f}%")
    wandb.finish()

    # todo: 按照temperature构造progress

if __name__ == "__main__":
    main()