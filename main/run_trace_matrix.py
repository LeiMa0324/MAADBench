"""Run the benchmark matrix with one sequential lane per model family."""

from __future__ import annotations

import argparse
from collections import deque
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOMS = (
    REPO_ROOT / "output/rooms/gsm-hard_easy_25_medium_25_hard_25_nightmare_25_20260907_130317.jsonl",
    REPO_ROOT / "output/rooms/livecodebench_easy_25_medium_25_hard_25_nightmare_25_20260907_130328.jsonl",
)
CONFIG_FOLDERS = (
    "claude_opus4.8",
    "claude_sonnet4",
    "deepseek",
    "gpt_41",
    "gpt_54",
)
TEMPERATURES = ("0.0", "0.3", "0.6")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _config_paths(model_families: tuple[str, ...] = CONFIG_FOLDERS) -> list[Path]:
    return [
        REPO_ROOT / "configs" / folder / f"config_{temperature}.yaml"
        for folder in model_families
        for temperature in TEMPERATURES
    ]


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(state, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run room sets and model configs in model-parallel lanes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--rooms-file", action="append", dest="rooms_files",
        help="Room JSONL to run; repeat for multiple files",
    )
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--unit-tools", choices=["enabled", "disabled"], default="enabled")
    parser.add_argument("--wandb-project", default="MADBench")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--no-upload-artifacts", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument(
        "--max-parallel-models", type=int, default=None,
        help="Maximum model families running concurrently; defaults to all selected models",
    )
    parser.add_argument(
        "--state-file", default=None,
        help="Local JSON progress file; defaults to output/matrix_<timestamp>_status.json",
    )
    return parser.parse_args()


def _job_command(
    job: dict[str, Any], args: argparse.Namespace, matrix_id: str, total_jobs: int
) -> list[str]:
    command = [
        sys.executable, "-u", "-m", "main.runner",
        "--rooms-file", job["rooms_file"],
        "--config", job["config"],
        "--unit-tools", args.unit_tools,
        "--output-dir", args.output_dir,
        "--wandb-project", args.wandb_project,
        "--wandb-mode", args.wandb_mode,
        "--wandb-group", matrix_id,
        "--wandb-job-index", str(job["index"]),
        "--wandb-job-total", str(total_jobs),
    ]
    if args.wandb_entity:
        command.extend(("--wandb-entity", args.wandb_entity))
    if args.no_upload_artifacts:
        command.append("--no-wandb-upload-artifacts")
    return command


def main(
    model_families: tuple[str, ...] | None = None,
    unit_tools: str | None = None,
    target_jobs: set[tuple[str, str]] | None = None,
) -> int:
    args = parse_args()
    if unit_tools is not None:
        args.unit_tools = unit_tools
    selected_models = model_families or CONFIG_FOLDERS
    rooms_files = [Path(path).resolve() for path in (args.rooms_files or DEFAULT_ROOMS)]
    configs = _config_paths(selected_models)
    missing = [str(path) for path in (*rooms_files, *configs) if not path.is_file()]
    if missing:
        print("Missing required input files:", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        return 2

    model_tag = (
        selected_models[0].replace(".", "_")
        if len(selected_models) == 1 else "all_models"
    )
    tool_tag = "tool" if args.unit_tools == "enabled" else "no_tool"
    matrix_id = (
        f"matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
        f"{model_tag}_{tool_tag}"
    )
    state_path = (
        Path(args.state_file).resolve()
        if args.state_file
        else (REPO_ROOT / args.output_dir / f"{matrix_id}_status.json").resolve()
    )
    jobs: list[dict[str, Any]] = []
    for rooms_file in rooms_files:
        for config in configs:
            domain = rooms_file.name.split("_", 1)[0]
            temperature = config.stem.removeprefix("config_")
            if target_jobs is not None and (domain, temperature) not in target_jobs:
                continue
            jobs.append({
                "index": len(jobs) + 1,
                "domain": domain,
                "model_family": config.parent.name,
                "temperature": temperature,
                "rooms_file": str(rooms_file),
                "config": str(config),
                "status": "pending",
                "started_at": None,
                "finished_at": None,
                "return_code": None,
            })

    state: dict[str, Any] = {
        "matrix_id": matrix_id,
        "status": "running",
        "created_at": _now(),
        "finished_at": None,
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
        "total_jobs": len(jobs),
        "completed_jobs": 0,
        "failed_jobs": 0,
        "max_parallel_models": max(1, min(
            len(selected_models) if args.max_parallel_models is None
            else args.max_parallel_models,
            len(selected_models),
        )),
        "jobs": jobs,
    }
    _write_state(state_path, state)
    print(f"Matrix: {matrix_id}")
    max_parallel = state["max_parallel_models"]
    print(
        f"Jobs: {len(jobs)} ({max_parallel} model lanes in parallel; "
        "each lane is sequential)"
    )
    print(f"Local status: {state_path}")

    env = os.environ.copy()
    lanes = {
        model: deque(job for job in jobs if job["model_family"] == model)
        for model in selected_models
    }
    active: dict[str, tuple[subprocess.Popen, dict[str, Any]]] = {}
    stop_launching = False

    try:
        while active or any(lanes.values()):
            if not stop_launching:
                for model in selected_models:
                    if len(active) >= max_parallel:
                        break
                    if model in active or not lanes[model]:
                        continue
                    job = lanes[model].popleft()
                    command = _job_command(job, args, matrix_id, len(jobs))
                    job["status"] = "running"
                    job["started_at"] = _now()
                    job["command"] = command
                    _write_state(state_path, state)
                    print(
                        f"\n[{job['index']}/{len(jobs)}] START "
                        f"{model} temp={job['temperature']} domain={job['domain']}",
                        flush=True,
                    )
                    try:
                        process = subprocess.Popen(
                            command, cwd=REPO_ROOT, env=env
                        )
                    except OSError as exc:
                        job["status"] = "failed"
                        job["return_code"] = 127
                        job["finished_at"] = _now()
                        job["error"] = f"{type(exc).__name__}: {exc}"
                        state["failed_jobs"] += 1
                        _write_state(state_path, state)
                        if args.stop_on_error:
                            stop_launching = True
                        continue
                    active[model] = (process, job)

            if not active:
                break

            time.sleep(0.5)
            for model, (process, job) in list(active.items()):
                return_code = process.poll()
                if return_code is None:
                    continue
                del active[model]
                job["return_code"] = return_code
                job["finished_at"] = _now()
                if return_code == 0:
                    job["status"] = "completed"
                    state["completed_jobs"] += 1
                else:
                    job["status"] = "failed"
                    state["failed_jobs"] += 1
                    if args.stop_on_error:
                        stop_launching = True
                print(
                    f"\n[{job['index']}/{len(jobs)}] {job['status'].upper()} "
                    f"{model} temp={job['temperature']} domain={job['domain']}",
                    flush=True,
                )
                _write_state(state_path, state)
    except KeyboardInterrupt:
        for process, _ in active.values():
            process.terminate()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and any(
            process.poll() is None for process, _ in active.values()
        ):
            time.sleep(0.2)
        for process, job in active.values():
            if process.poll() is None:
                process.kill()
            job["status"] = "interrupted"
            job["return_code"] = process.poll()
            job["finished_at"] = _now()
        state["status"] = "interrupted"
        state["finished_at"] = _now()
        _write_state(state_path, state)
        print(f"\nInterrupted. Status saved to {state_path}")
        return 130

    if stop_launching:
        for lane in lanes.values():
            for job in lane:
                job["status"] = "skipped"
        state["status"] = "stopped_on_error"
    else:
        state["status"] = (
            "completed" if state["failed_jobs"] == 0
            else "completed_with_failures"
        )
    state["finished_at"] = _now()
    _write_state(state_path, state)
    print(
        f"\nFinished: {state['completed_jobs']} completed, "
        f"{state['failed_jobs']} failed. Status: {state_path}"
    )
    return 1 if state["failed_jobs"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
