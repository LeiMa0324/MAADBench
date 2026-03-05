import argparse
import csv
import json
import os
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
import yaml

# Add sage/ to path if running from project root
sys.path.insert(0, str(Path(__file__).parent))
import wandb

ENABLE_WANDB = False
from sage_generator import SAGEDataGenerator
from sage_anomaly import AnomalyInjector, AnomalyConfig
from sage_tracker import SAGETracker, TraceEvaluation, print_evaluation, FM_DESCRIPTIONS
from sage_mas import SAGEOrchestrator
from sage_prompts import get_system
from utils import *

def _count_by_level(results: list) -> dict:
    counts = {}
    for r in results:
        level = r.get("level", "unknown")
        ev    = r.get("evaluation") or {}
        overall = ev.get("overall", "error" if r.get("error") else "unknown")
        if level not in counts:
            counts[level] = {"success": 0, "partial": 0, "failure": 0, "error": 0}
        counts[level][overall] = counts[level].get(overall, 0) + 1
    return counts


_CSV_COLUMNS = [
    "trace_filename",
    "task_id", "level", "task_success",
    "step", "agent", "agent_fail",
    "fail_mode", "failure_source", "fail_desc", "tracker_detail",
]


def _append_to_anomaly_details(csv_path: str, task: dict, evaluation: dict,
                        trace_filename: str = "",
                        anomaly_record: dict = None,
                        task_id: str = "") -> None:
    """Append one row per agent step for a completed task to the results CSV."""
    task_id      = task_id or evaluation.get("task_id", task.get("task_id", ""))
    level        = task.get("level", "")
    task_success = evaluation.get("overall", "")

    write_header = not Path(csv_path).exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        if write_header:
            writer.writeheader()

        # ── Anomaly injection: single row for the injected agent ──────────
        if anomaly_record:
            writer.writerow({
                "trace_filename": trace_filename,
                "task_id":       task_id,
                "level":         level,
                "task_success":  task_success,
                "step":          anomaly_record.get("injection_step", 0) + 1,
                "agent":         anomaly_record.get("target_agent", ""),
                "agent_fail":    True,
                "fail_mode":     anomaly_record.get("anomaly_type", ""),
                "failure_source": "inject",
                "fail_desc":     anomaly_record.get("description", ""),
                "tracker_detail": "",
            })
            return

        # ── Tracker evaluation: only failed agents ─────────────────────
        steps = evaluation.get("steps", [])
        for idx, s in enumerate(steps, start=1):
            failed = s.get("status") not in ("correct", "skip")
            if not failed:
                continue
            writer.writerow({
                "trace_filename": trace_filename,
                "task_id":       task_id,
                "level":         level,
                "task_success":  task_success,
                "step":          idx,
                "agent":         s.get("agent", ""),
                "agent_fail":    True,
                "fail_mode":     s.get("failure_mode", ""),
                "failure_source": "tracker",
                "fail_desc":     s.get("failure_mode_desc", ""),
                "tracker_detail": s.get("detail", ""),
            })


def _agent_fm_breakdown(ev: "TraceEvaluation") -> dict:
    """Returns {agent: {fm_code: count}} for a single TraceEvaluation."""
    breakdown: dict[str, dict[str, int]] = {}
    for s in ev.steps:
        if s.failure_mode:
            breakdown.setdefault(s.agent, {})
            breakdown[s.agent][s.failure_mode] = breakdown[s.agent].get(s.failure_mode, 0) + 1
    return breakdown


def _aggregate_failure_modes(results: list) -> dict:
    fm_counts: dict[str, int] = {}
    for r in results:
        ev = r.get("evaluation") or {}
        for fm, n in ev.get("failure_mode_counts", {}).items():
            fm_counts[fm] = fm_counts.get(fm, 0) + n
    return dict(sorted(fm_counts.items()))


def _aggregate_agent_failures(results: list) -> dict:
    """Returns {agent: {fm_code: count}} across all tasks."""
    combined: dict[str, dict[str, int]] = {}
    for r in results:
        ev = r.get("evaluation") or {}
        for agent, fm_counts in ev.get("agent_failure_modes", {}).items():
            if agent not in combined:
                combined[agent] = {}
            for fm, n in fm_counts.items():
                combined[agent][fm] = combined[agent].get(fm, 0) + n
    return combined

def _serialise(obj):
    """Recursively make an object JSON-serialisable."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialise(v) for v in obj]
    # Protocol dataclasses (ParserOutput, etc.)
    if hasattr(obj, "__dataclass_fields__"):
        return _serialise(vars(obj))
    # Fallback
    return str(obj)


def _evaluation_to_dict(ev: TraceEvaluation) -> dict:
    return {
        "task_id": ev.task_id,
        "overall": ev.overall,
        "correct_steps": ev.correct_steps(),
        "total_steps": ev.total_steps(),
        "first_failure_step": ev.first_failure_step,
        "first_failure_agent": ev.first_failure_agent,
        "failure_propagated": ev.failure_propagated,
        "anomaly_detected": ev.anomaly_detected,
        "agent_accuracy": ev.agent_accuracy(),
        "failure_mode_counts": ev.failure_mode_counts(),
        "agent_failure_modes": _agent_fm_breakdown(ev),
        "steps": [
            {
                "step": r.step,
                "agent": r.agent,
                "status": r.status,
                "expected": r.expected,
                "actual": r.actual,
                "failure_mode": r.failure_mode,
                "failure_category": r.failure_category,
                "detail": r.detail,
            }
            for r in ev.steps
        ],
    }

def parse_args():
    p = argparse.ArgumentParser(
        description="SAGE MAS Benchmark Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset",   type=str, default="SAGE")
    p.add_argument("--mas_arch",   type=str, default="AG2")
    p.add_argument("--config_file", type=str, default="configs/config.yaml",
                        help="Path to MAS config yaml.")
    p.add_argument("--tasks",  type=str, default=None,help="Load tasks from CSV (skips generation)")
    p.add_argument("--seed",   type=int, default=42,  help="Random seed for task generation")
    p.add_argument("--max-steps", type=int, default=40, help="Max simplification loop iterations")
    p.add_argument("--verbose",    action="store_true", help="Print per-task evaluation details")
    p.add_argument("--model", type=str, default=None, help="Override LLM model ID")
    p.add_argument("--output_dir", type=str, default="traces",
                        help="Directory to save execution traces.")
    return p.parse_args()

class SageRunner:

    def __init__(self, args):
        self.args = args
        with open(self.args.config_file, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.mas = SAGEOrchestrator.from_config(self.args.config_file)

    def run_task(self,
            task: dict,
            task_id: str,
            verbose: bool,
    ) -> dict:
        """
        Run one SAGE task through the full MAS pipeline and evaluate.

        Returns a result dict containing:
          task_id, expression, ground_truth, k_steps,
          anomaly_record (or None),
          mas_trace        (list of per-agent step dicts from orchestrator),
          evaluation       (SAGETracker output),
          elapsed_seconds,
          error            (None on success, exception string on failure)
        """
        run_info = {"anomaly_record": None, "elapsed_seconds": 0.0, "error": None}

        # ── Anomaly injection from task dict ──────────────────────────────────
        anomaly_dict = task.get("anomaly")
        if anomaly_dict:
            config = AnomalyConfig.from_dict(anomaly_dict)
            anomaly_injector = AnomalyInjector(seed=self.args.seed)
            self.mas.set_anomaly(anomaly_injector, config)
        else:
            self.mas.set_anomaly(None, None)

        # ── Run MAS ───────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        try:
            self.mas._run_sequential(problem=task["expression"])
        except Exception as e:
            run_info["error"] = traceback.format_exc()
        run_info["elapsed_seconds"] = time.perf_counter() - t0

        # Retrieve anomaly record from orchestrator (even on error)
        if self.mas.anomaly_record is not None:
            run_info["anomaly_record"] = _serialise(self.mas.anomaly_record)

        return run_info

    def evaluate(self, task, task_id: str, verbose: bool, run_info: dict = None,
                 full_eval: bool = True):
        run_info = run_info or {}
        result = {
            "evaluation": None,
            "elapsed_seconds": run_info.get("elapsed_seconds", 0.0),
            "error": run_info.get("error"),
        }

        if result["error"]:
            return result

        tracker = SAGETracker(task)
        err = tracker.record_from_trace(self.mas.trace)

        if not full_eval:
            overall = tracker.evaluate_task()
            result["evaluation"] = {
                "overall": overall,
                "failure_mode_counts": {},
                "agent_failure_modes": {},
            }
            return result

        if err:
            result["error"] = err
            return result

        evaluation = tracker.agent_evaluate(task_id=task_id)
        result["evaluation"] = _evaluation_to_dict(evaluation)

        if verbose:
            print_evaluation(evaluation)

        return result


    def run(self, tasks):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # ── 4. Run tasks ──────────────────────────────────────────────────────
        all_results = []
        n_success = n_partial = n_failure = n_error = 0

        # Per-level counters for real-time wandb logging
        level_total = {}
        for t in tasks:
            lv = t.get("level", "unknown")
            level_total[lv] = level_total.get(lv, 0) + 1
        level_tested  = {lv: 0 for lv in level_total}
        level_success = {lv: 0 for lv in level_total}
        level_partial = {lv: 0 for lv in level_total}

        print(f"\nRunning {len(tasks)} tasks...\n{'=' * 60}")

        for idx, task in enumerate(tasks):
            task_id = task.get("task_id") or f"task_{idx:04d}"
            print(f"\n[{idx + 1}/{len(tasks)}] {task_id}  ({task.get('level', '?')})  "
                  f"k={task['k_steps']}  expr={task['expression'][:60]}")

            # Reset orchestrator trace between tasks
            self.mas.trace = []

            run_info = self.run_task(
                task=task,
                task_id=task_id,
                verbose=args.verbose,
            )

            full_eval = not bool(task.get("anomaly"))
            evaluation_trace = self.evaluate(task, task_id, args.verbose, run_info,
                                             full_eval=full_eval)
            all_results.append(evaluation_trace)

            # Tally counters
            level = task.get("level", "unknown")
            level_tested[level] = level_tested.get(level, 0) + 1
            overall, n_success, n_partial, n_failure, n_error = self._tally_result(
                evaluation_trace, level, level_success, level_partial,
                n_success, n_partial, n_failure, n_error,
            )

            # Per-level wandb logging
            if ENABLE_WANDB:
                self._log_wandb(level_total, level_tested, level_success, level_partial)

            # Build trace_id, attach tracker evals to trace, write CSV, save trace
            self._save_task_output(
                task=task, task_id=task_id, run_info=run_info,
                evaluation_trace=evaluation_trace,
            )

        # ── 5. Summary ────────────────────────────────────────────────────────
        self._print_summary(all_results, timestamp,
                            n_success, n_partial, n_failure, n_error)

    # -------------------------------------------------------------------
    # Helper methods extracted from run()
    # -------------------------------------------------------------------

    @staticmethod
    def _tally_result(tracker_trace, level, level_success, level_partial,
                      n_success, n_partial, n_failure, n_error):
        """Update counters and print per-task result line."""
        if tracker_trace["error"]:
            n_error += 1
            print(f"  ERROR: {tracker_trace['error'][:120]}")
            return "error", n_success, n_partial, n_failure, n_error

        overall = tracker_trace["evaluation"]["overall"]
        if overall == "success":
            n_success += 1
            level_success[level] = level_success.get(level, 0) + 1
        elif overall == "partial":
            n_partial += 1
            level_partial[level] = level_partial.get(level, 0) + 1
        else:
            n_failure += 1

        ev = tracker_trace["evaluation"]
        steps_str = (
            f"{ev['correct_steps']}/{ev['total_steps']} steps correct  "
            if ev.get("correct_steps") is not None else ""
        )
        print(f"  → {overall.upper()}  {steps_str}"
              f"({tracker_trace['elapsed_seconds']:.1f}s)")
        return overall, n_success, n_partial, n_failure, n_error

    @staticmethod
    def _log_wandb(level_total, level_tested, level_success, level_partial):
        """Log per-level accuracy to wandb."""
        wb_log = {}
        for lv in level_total:
            tested = level_tested.get(lv, 0)
            if tested > 0:
                s = level_success.get(lv, 0)
                p = level_partial.get(lv, 0)
                wb_log[f"{lv}_full_accuracy"] = s / tested
                wb_log[f"{lv}_partial_accuracy"] = (s + p) / tested
            wb_log[f"{lv}_progress"] = tested / level_total[lv]
        wandb.log(wb_log)

    def _save_task_output(self, task, task_id, run_info, evaluation_trace):
        """Generate trace_id, attach tracker evals, write CSV, save trace JSON."""
        ev_dict = evaluation_trace.get("evaluation") or {}

        # Generate trace filename
        label = ev_dict.get("overall", "unknown")

        anomaly_dict = task.get("anomaly")
        if anomaly_dict:
            fm = anomaly_dict["anomaly_type"].replace("-", "").replace(".", "")
            label = f"injected_{fm}_{label}"

        trace_id = generate_trace_id(
            dataset=args.dataset,
            mas_arch=args.mas_arch,
            query_id=task_id,
            label=label,
        )

        # Attach tracker evals to trace entries by agent name
        eval_steps = ev_dict.get("steps", [])
        _agent_name_map = {"expression_parser": "parser", "equivalence_checker": "checker"}
        eval_idx = 0
        for trace_entry in self.mas.trace:
            if eval_idx >= len(eval_steps):
                break
            trace_agent = _agent_name_map.get(trace_entry["agent"], trace_entry["agent"])
            if trace_agent == eval_steps[eval_idx]["agent"]:
                trace_entry["tracker_eval"] = eval_steps[eval_idx]
                eval_idx += 1

        # Write anomaly details
        csv_path = str(Path(args.output_dir) / "anomaly_details.csv")
        anomaly_rec = run_info.get("anomaly_record")

        if anomaly_rec or eval_steps:
            _append_to_anomaly_details(csv_path, task, ev_dict,
                                trace_filename=trace_id,
                                anomaly_record=anomaly_rec,
                                task_id=task_id)

        ev_dict.pop("steps", None)

        # Save trace JSON
        meta_data = {
            "config": self.config,
            "question": task,
            "injected_anomaly": run_info.get("anomaly_record"),
        }
        self.mas.save_execution_trace(meta_data=meta_data, output_dir=args.output_dir,
                                      trace_id=trace_id)

    def _print_summary(self, all_results, timestamp,
                        n_success, n_partial, n_failure, n_error):
        """Print final run summary to console."""
        total = len(all_results)
        summary = {
            "run_timestamp": timestamp,
            "seed": args.seed,
            "total_tasks": total,
            "n_success": n_success,
            "n_partial": n_partial,
            "n_failure": n_failure,
            "n_error": n_error,
            "success_rate": n_success / total if total else 0,
            "max_steps": args.max_steps,
            "tasks_by_level": _count_by_level(all_results),
            "failure_modes": _aggregate_failure_modes(all_results),
            "agent_failure_modes": _aggregate_agent_failures(all_results),
        }

        print(f"\n{'=' * 60}")
        print(f"  Run complete: {total} tasks")
        print(f"  Success  : {n_success}/{total} ({summary['success_rate']:.1%})")
        print(f"  Partial  : {n_partial}/{total}")
        print(f"  Failure  : {n_failure}/{total}")
        if n_error:
            print(f"  Errors   : {n_error}/{total}")
        if summary["failure_modes"]:
            print(f"\n  Failure modes (overall):")
            for fm, count in sorted(summary["failure_modes"].items()):
                desc = FM_DESCRIPTIONS.get(fm, "")
                print(f"    {fm} ×{count}  {desc}")
        if summary["agent_failure_modes"]:
            print(f"\n  Failure modes by agent:")
            for agent in ["parser", "rule_selector", "transformer", "checker"]:
                fm_counts = summary["agent_failure_modes"].get(agent)
                if not fm_counts:
                    continue
                total_agent = sum(fm_counts.values())
                fm_str = "  ".join(
                    f"{fm} ×{n}" for fm, n in sorted(fm_counts.items())
                )
                print(f"    {agent:<16} {total_agent} failures  →  {fm_str}")
        print(f"\n{'=' * 60}\n")


if __name__ == '__main__':
    args = parse_args()

    if ENABLE_WANDB:
        wandb.init(
            project=f"Lomas-{args.dataset}",
            config={
                "dataset": args.dataset,
                "mas_arch": args.mas_arch,
                "config_file": args.config_file,
                "seed": args.seed,
            },
            resume="allow",
        )

    runner = SageRunner(
        args=args
    )

    tasks = load_jsonl("tasks.jsonl")
    runner.run(tasks)

    if ENABLE_WANDB:
        wandb.finish()