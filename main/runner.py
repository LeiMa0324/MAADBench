"""
runner.py — benchmark_core benchmark runner.

Generates rooms, runs MAS pipeline, saves per-room traces as individual
JSON files (same convention as sage_runner).

Output structure:
    {output_dir}/rooms/
        {domain}_{level}_{num}_..._{timestamp}.jsonl
    {output_dir}/traces/run_{timestamp}_{domain}_{tool_mode}_{model}_{temperature}.../
        {trace_id}.json                    # one file per room run
        summary.csv                        # aggregate stats

Each trace JSON contains:
    config            — MAS config (from YAML + CLI overrides)
    room              — full room state (puzzles, items, clues, traps)
    trace             — per-action step records from orchestrator
    failure_report    — structured failure analysis
    escaped           — bool
    timing_sec        — wall-clock seconds

Usage:
    python -m main.runner --n-easy 5 --n-medium 3 --n-hard 2
    python -m main.runner --n-hard 10 --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Tee stdout to log file
# ---------------------------------------------------------------------------

class _TeeStream:
    """Write to both the original stream and a log file."""

    def __init__(self, stream, log_file):
        self._stream = stream
        self._log = open(log_file, "w", encoding="utf-8")

    def write(self, data):
        self._stream.write(data)
        self._log.write(data)
        self._log.flush()

    def flush(self):
        self._stream.flush()
        self._log.flush()

    def close(self):
        self._log.close()

from benchmark_core.room_core import Room
from benchmark_core.mas import EROrchestrator
from benchmark_core.room_planner import PlanningRoom
from benchmark_core.mas_planner import ERPlannerOrchestrator
from benchmark_core.agent_prompts import get_actions, get_agent_roles, get_agent_prompt
from benchmark_core.domains import get_domain


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="benchmark_core MAS Benchmark Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Generation — per-difficulty counts
    p.add_argument("--n-easy", type=int, default=0,
                   help="Number of easy rooms")
    p.add_argument("--n-medium", type=int, default=0,
                   help="Number of medium rooms")
    p.add_argument("--n-hard", type=int, default=0,
                   help="Number of hard rooms")
    p.add_argument("--n-nightmare", type=int, default=0,
                   help="Number of nightmare rooms (PlanningRoom with DAG)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rooms-file", type=str, default=None,
                   help="Reuse rooms from a JSONL file instead of generating or copying them")

    # MAS config
    p.add_argument("--config", type=str, default="configs/config.yaml")
    p.add_argument("--mode", type=str, default="natural",
                   choices=["natural", "structured"],
                   help="Agent communication mode: 'natural' passes messages, "
                        "'structured' passes structured JSON between agents")
    p.add_argument("--clue-domain", type=str, default="gsm-hard",
                   choices=["gsm-hard", "livecodebench"],
                   help="Problem domain used to generate clue numeric answers")
    p.add_argument("--unit-tools", type=str, default=None,
                   choices=["enabled", "disabled"],
                   help="Allow APPLY_DELTA to call audited unit-conversion tools; "
                        "defaults to system.unit_tools in the YAML config")
    p.add_argument("--max-unit-tool-calls", type=int, default=None,
                   help="Maximum conversion calls per APPLY_DELTA action")

    # Injection
    p.add_argument("--fm-id", type=str, default=None,
                   help="Failure mode to inject, e.g. 'FM-1.1', 'FM-2.5'")
    p.add_argument("--injection-type", type=str, default=None,
                   choices=["output", "prompt"],
                   help="Injection type: 'output' corrupts structured_output after LLM, "
                        "'prompt' modifies prompt before LLM")

    # Output
    p.add_argument("--output-dir", type=str, default="output",
                   help="Output root containing rooms/ and traces/")
    p.add_argument("--dataset", type=str, default="EscapeRoom")
    p.add_argument("--mas-arch", type=str, default="sequential")
    p.add_argument("--wandb-project", type=str, default=None,
                   help="Enable W&B tracking and use this project name")
    p.add_argument("--wandb-entity", type=str, default=None,
                   help="Optional W&B team or username")
    p.add_argument("--wandb-group", type=str, default=None,
                   help="Group related runs, such as one matrix execution")
    p.add_argument("--wandb-mode", type=str, default="online",
                   choices=["online", "offline", "disabled"])
    p.add_argument("--wandb-job-index", type=int, default=None)
    p.add_argument("--wandb-job-total", type=int, default=None)
    p.add_argument("--no-wandb-upload-artifacts", action="store_true",
                   help="Do not upload trace JSON, summary.csv, and run.log")

    # Flags

    p.add_argument("--dry-run", action="store_true",
                   help="Generate rooms only, do not run MAS")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Serialisation helper
# ---------------------------------------------------------------------------

def _serialise(obj):
    """Recursively make an object JSON-serialisable."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialise(v) for v in obj]
    if isinstance(obj, set):
        return sorted(_serialise(v) for v in obj)
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "to_dict"):
        return _serialise(obj.to_dict())
    if hasattr(obj, "__dataclass_fields__"):
        return _serialise(vars(obj))
    return str(obj)



# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class EscapeRoomRunner:

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.clue_domain = getattr(args, "clue_domain", "gsm-hard")

        # Load config
        with open(args.config, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f) or {}

        self.domain = get_domain(self.clue_domain)

        system_config = self.config.get("system", {})
        self.unit_tools = (
            bool(system_config.get("unit_tools", False))
            if getattr(args, "unit_tools", None) is None
            else args.unit_tools == "enabled"
        )
        self.max_unit_tool_calls = (
            int(system_config.get("max_unit_tool_calls", 3))
            if getattr(args, "max_unit_tool_calls", None) is None
            else max(0, args.max_unit_tool_calls)
        )

        raw_model = self.config.get("llm", {}).get("model", "unknown")
        self.model = raw_model.split("/")[-1] if "/" in raw_model else raw_model
        self.temperature = self.config.get("llm", {}).get("temperature", "")
        self.llm_seed = self.config.get("llm", {}).get("seed", None)

        # Per-difficulty counts
        self.room_counts = {
            "easy":      args.n_easy,
            "medium":    args.n_medium,
            "hard":      args.n_hard,
            "nightmare": args.n_nightmare,
        }
        self.n_rooms_total = sum(self.room_counts.values())

        # Output directory
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.timestamp = ts
        inj_tag = f"{args.fm_id}_" if args.fm_id else ""
        if args.rooms_file:
            is_nightmare = self._file_is_nightmare(args.rooms_file)
            run_domain = self._file_clue_domain(args.rooms_file, self.clue_domain)
        else:
            is_nightmare = (args.n_nightmare > 0 and args.n_easy == 0
                            and args.n_medium == 0 and args.n_hard == 0)
            run_domain = self.domain.name
        run_domain_tag = run_domain.replace("/", "-")
        nightmare_tag = "_nightmare" if is_nightmare else ""
        seed_tag = f"_seed{self.llm_seed}" if self.llm_seed is not None else ""
        tool_mode_tag = "tool" if self.unit_tools else "no_tool"
        self.output_root = Path(args.output_dir)
        self.rooms_dir = self.output_root / "rooms"
        self.traces_dir = self.output_root / "traces"
        self.rooms_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = self.traces_dir / (
            f"run_{ts}_{run_domain_tag}_{tool_mode_tag}_"
            f"{inj_tag}{self.model}_{self.temperature}"
            f"{nightmare_tag}{seed_tag}"
        )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.rooms_path: Optional[Path] = None
        self.summary_path = self.run_dir / "summary.csv"
        self.log_path = self.run_dir / "run.log"
        self.run_domain = run_domain
        self.tool_mode_tag = tool_mode_tag

        # Tee all stdout to log file
        self._tee = _TeeStream(sys.stdout, self.log_path)
        sys.stdout = self._tee

        print(f"[ERRunner] config     : {args.config}")
        print(f"[ERRunner] model      : {self.model}")
        print(f"[ERRunner] mode       : {args.mode}")
        print(f"[ERRunner] clue_domain: {self.clue_domain}")
        print(f"[ERRunner] unit_tools : {'enabled' if self.unit_tools else 'disabled'} "
              f"(max_calls={self.max_unit_tool_calls})")
        if args.fm_id:
            print(f"[ERRunner] injection  : {args.fm_id} ({args.injection_type})")
        if args.rooms_file:
            print(f"[ERRunner] rooms_file : {args.rooms_file}")
        else:
            print(f"[ERRunner] rooms      : easy={args.n_easy}, medium={args.n_medium}, hard={args.n_hard}, nightmare={args.n_nightmare} (total={self.n_rooms_total})")
        print(f"[ERRunner] output     : {self.run_dir}")

        self.wandb_tracker = None
        if getattr(args, "wandb_project", None):
            from main.wandb_tracker import WandbTracker

            tracker_config = {
                "rooms_file": args.rooms_file,
                "config_file": args.config,
                "domain": run_domain,
                "model": raw_model,
                "temperature": self.temperature,
                "seed": self.llm_seed,
                "unit_tools": self.unit_tools,
                "max_unit_tool_calls": self.max_unit_tool_calls,
                "agent_mode": args.mode,
                "matrix_job_index": getattr(args, "wandb_job_index", None),
                "matrix_job_total": getattr(args, "wandb_job_total", None),
                "llm_config": self.config.get("llm", {}),
            }
            self.wandb_tracker = WandbTracker(
                project=args.wandb_project,
                entity=getattr(args, "wandb_entity", None),
                group=getattr(args, "wandb_group", None),
                name=self.run_dir.name,
                mode=getattr(args, "wandb_mode", "online"),
                run_dir=self.run_dir,
                config=tracker_config,
                tags=[run_domain, self.model, tool_mode_tag],
                upload_artifacts=not getattr(
                    args, "no_wandb_upload_artifacts", False
                ),
            )
            if self.wandb_tracker.url:
                print(f"[ERRunner] wandb      : {self.wandb_tracker.url}")

    # ── Generate rooms ───────────────────────────────────────────────────

    def _rooms_filename(self, rooms: list) -> str:
        """Build domain_{level}_{num}_..._{timestamp}.jsonl."""
        domain_names = {
            getattr(p.clue, "domain", self.domain.name)
            for room in rooms
            for p in room.puzzles
        }
        domain_name = next(iter(domain_names)) if len(domain_names) == 1 else "mixed"
        if not domain_names:
            domain_name = self.domain.name

        counts = {level: 0 for level in ("easy", "medium", "hard", "nightmare")}
        for room in rooms:
            level = "nightmare" if isinstance(room, PlanningRoom) else room.difficulty
            counts[level] = counts.get(level, 0) + 1

        parts = [domain_name]
        for level in ("easy", "medium", "hard", "nightmare"):
            if counts[level]:
                parts.extend((level, str(counts[level])))
        if len(parts) == 1:
            parts.extend(("empty", "0"))
        parts.append(self.timestamp)
        return "_".join(parts) + ".jsonl"

    def _generate_rooms(self) -> list:
        if self.args.rooms_file:
            return self._load_rooms(self.args.rooms_file)
        import random
        random.seed(self.args.seed)
        rooms = []
        idx = 0
        for difficulty in ("easy", "medium", "hard"):
            for _ in range(self.room_counts[difficulty]):
                rooms.append(Room.create(difficulty=difficulty,
                                         room_id=f"room_{idx:04d}",
                                         clue_domain=self.domain))
                idx += 1
        for _ in range(self.room_counts["nightmare"]):
            rooms.append(PlanningRoom.create(difficulty="nightmare",
                                             room_id=f"room_{idx:04d}",
                                             clue_domain=self.domain))
            idx += 1
        return rooms

    @staticmethod
    def _file_is_nightmare(path: str) -> bool:
        """True if a rooms file consists entirely of nightmare (planning) rooms."""
        saw_room = False
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                saw_room = True
                if json.loads(line).get("type") != "planning":
                    return False
        return saw_room

    @staticmethod
    def _file_clue_domain(path: str, fallback: str = "gsm-hard") -> str:
        """Return the single clue domain stored in a rooms JSONL file."""
        domains = set()
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                room = json.loads(line)
                for puzzle in room.get("puzzles", []):
                    domain = (puzzle.get("clue") or {}).get("domain")
                    if domain:
                        domains.add(domain)
        if not domains:
            return fallback
        return next(iter(domains)) if len(domains) == 1 else "mixed"

    @staticmethod
    def _load_rooms(path: str) -> list:
        """Load rooms from a JSONL file. Auto-detects Room vs PlanningRoom."""
        rooms = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    if d.get("type") == "planning":
                        rooms.append(PlanningRoom.from_dict(d))
                    else:
                        rooms.append(Room.from_dict(d))
        print(f"[ERRunner] Loaded {len(rooms)} rooms from {path}")
        return rooms

    # ── Run one room ─────────────────────────────────────────────────────

    def _run_one_room(self, room, room_idx: int) -> dict:
        """Run MAS on one room and return the result dict."""
        difficulty = "nightmare" if isinstance(room, PlanningRoom) else room.difficulty
        domain_names = {
            getattr(p.clue, "domain", self.clue_domain) for p in room.puzzles
        }
        if len(domain_names) > 1:
            raise ValueError(f"Room {room.room_id} mixes clue domains: {sorted(domain_names)}")
        room_domain_name = next(iter(domain_names), self.clue_domain)
        domain = get_domain(room_domain_name)
        actions = get_actions(difficulty, domain)
        agent_roles = get_agent_roles(difficulty)

        # Build agents from agents.py prompts + config LLM settings
        from benchmark_core.framework import Agent
        llm_config = self.config.get("llm", {})
        agents = {}
        for role in agent_roles:
            agents[role] = Agent(
                role=role,
                role_prompt=get_agent_prompt(role, domain),
                llm_config=llm_config,
            )

        if isinstance(room, PlanningRoom):
            orch = ERPlannerOrchestrator(
                agents=agents,
                actions=actions,
                mode=self.args.mode,
                fm_id=self.args.fm_id,
                injection_type=self.args.injection_type,
                stop_on_fail=True,
                unit_tools=self.unit_tools,
                max_unit_tool_calls=self.max_unit_tool_calls,
            )
        else:
            orch = EROrchestrator(
                agents=agents,
                actions=actions,
                mode=self.args.mode,
                fm_id=self.args.fm_id,
                injection_type=self.args.injection_type,
                unit_tools=self.unit_tools,
                max_unit_tool_calls=self.max_unit_tool_calls,
            )

        t0 = time.time()
        result = orch.run(room=room)
        elapsed = time.time() - t0

        failure_report = orch.get_failure_report()
        self._strip_tool_metadata_from_llm_actions(result["trace"])

        return {
            "escaped": result["escaped"],
            "puzzles_solved": result["puzzles_solved"],
            "puzzles_total": result["puzzles_total"],
            "trace": result["trace"],
            "failure_report": failure_report,
            "timing_sec": round(elapsed, 2),
        }

    @staticmethod
    def _strip_tool_metadata_from_llm_actions(trace: list[dict]) -> None:
        """Keep tool details on UNIT_TOOL_CALL entries, not LLM actions."""
        tool_fields = (
            "unit_tool_mode", "tool_calls", "unit_tool_diagnostics", "llm_calls",
        )
        for entry in trace:
            if entry.get("action") == "UNIT_TOOL_CALL":
                continue
            for field in tool_fields:
                entry.pop(field, None)

    # ── Save per-room trace ──────────────────────────────────────────────

    def _save_room_trace(self, room, room_idx: int, result: dict) -> str:
        """Save one room's trace as an individual JSON file."""
        label = "escaped" if result["escaped"] else "failed"
        room_id = room.room_id

        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        trace_id = f"{ts}_{self.args.dataset}_{room_id}_{label}"

        # Build config with agents info for trace reproducibility
        agent_roles = get_agent_roles(room.difficulty)
        domain_names = {
            getattr(p.clue, "domain", self.clue_domain) for p in room.puzzles
        }
        room_domain = get_domain(next(iter(domain_names), self.clue_domain))
        agents_info = [
            {"role": role, "prompt": get_agent_prompt(role, room_domain)}
            for role in agent_roles
        ]
        config_with_agents = {
            **self.config,
            "agents": agents_info,
            "benchmark": {
                "clue_domain": room_domain.name,
                "unit_tools": self.unit_tools,
                "max_unit_tool_calls": self.max_unit_tool_calls,
            },
        }

        trace_payload = {
            "config": _serialise(config_with_agents),
            "room": _serialise(room.to_dict()),
            "trace": _serialise(result["trace"]),
            "failure_report": _serialise(result["failure_report"]),
            "escaped": result["escaped"],
            "puzzles_solved": result["puzzles_solved"],
            "puzzles_total": result["puzzles_total"],
            "timing_sec": result["timing_sec"],
        }

        trace_path = self.run_dir / f"{trace_id}.json"
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(trace_payload, f, indent=2, ensure_ascii=False)
        print(f"  Trace → {trace_path}")
        return trace_id

    # ── Main entry point ─────────────────────────────────────────────────

    def _run_impl(self):
        rooms = self._generate_rooms()
        if self.args.rooms_file:
            self.rooms_path = Path(self.args.rooms_file).resolve()
            print(f"[ERRunner] Reusing {len(rooms)} rooms → {self.rooms_path}")
        else:
            self.rooms_path = self.rooms_dir / self._rooms_filename(rooms)
            # Save newly generated rooms to JSONL (using Room.to_dict for round-trip support).
            with open(self.rooms_path, "w", encoding="utf-8") as f:
                for room in rooms:
                    f.write(json.dumps(_serialise(room.to_dict()), ensure_ascii=False) + "\n")
            print(f"[ERRunner] Saved {len(rooms)} rooms → {self.rooms_path}")

        if self.args.dry_run:
            print(f"\nDry-run: {len(rooms)} rooms generated. Exiting.")
            return

        # Run rooms
        n_escaped = 0
        self._init_summary_csv()

        print(f"\n{'━' * 60}")
        print(f"  Running {len(rooms)} rooms")
        print(f"{'━' * 60}")

        for i, room in enumerate(rooms):
            print(f"\n\033[1;33m{'█' * 60}")
            print(f"  [{i + 1}/{len(rooms)}] {room.room_id} │ {room.difficulty}")
            print(f"{'█' * 60}\033[0m")

            try:
                result = self._run_one_room(room, i)
            except Exception as e:
                print(f"  ⚠  ROOM CRASHED: {type(e).__name__}: {e}")
                result = {
                    "escaped": False,
                    "puzzles_solved": 0,
                    "puzzles_total": len(room.puzzles),
                    "trace": [],
                    "failure_report": {"error": f"{type(e).__name__}: {e}"},
                    "timing_sec": 0.0,
                }

            if result["escaped"]:
                n_escaped += 1
                icon = "ESCAPED"
            else:
                icon = "FAILED"

            print(f"  {icon}  solved={result['puzzles_solved']}/{result['puzzles_total']}  "
                  f"({result['timing_sec']:.1f}s)")
            print(f"  Running escape rate: {n_escaped}/{i + 1} = {n_escaped / (i + 1) * 100:.1f}%")

            trace_id = self._save_room_trace(room, i, result)
            self._append_summary_csv(room, result, trace_id)
            if self.wandb_tracker:
                difficulty = (
                    "nightmare" if isinstance(room, PlanningRoom)
                    else room.difficulty
                )
                self.wandb_tracker.log_room(
                    difficulty=difficulty,
                    result=result,
                    room_index=i + 1,
                    total_rooms=len(rooms),
                )

        # Statistics
        self.run_summarize(str(self.summary_path))

    def run(self):
        """Run the benchmark and always finalize log and W&B state."""
        status = "failed"
        exit_code = 1
        primary_error = None
        try:
            self._run_impl()
            status = "completed"
            exit_code = 0
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            try:
                if self.wandb_tracker:
                    self.wandb_tracker.finish(status=status, exit_code=exit_code)
            except Exception as exc:
                print(f"[ERRunner] W&B finalization failed: {type(exc).__name__}: {exc}")
                if primary_error is None:
                    raise
            finally:
                self._close_log()

    def _close_log(self):
        """Restore stdout and close the log file."""
        if hasattr(self, '_tee'):
            sys.stdout = self._tee._stream
            self._tee.close()
            print(f"Log saved → {self.log_path}")

    _CSV_COLUMNS = [
        "action_type", "room_id", "difficulty", "puzzle_id", "llm_model", "temperature",
        "timestamp", "task_success", "puzzle_solved", "trace_filename",
        "action_id", "action_name", "parent_action_id", "agent",
        "action_status", "error_type", "anomaly_label",
        "item_type", "answer", "gt", "desc", "fm_id",
    ]

    @staticmethod
    def _get_item_type(action_name: str, trace_entry: dict, room, puzzle_id_short: str) -> str:
        """Return 'clue' for clue actions, puzzle item type for item actions."""
        if action_name in ("OBSERVE_CLUE", "OBSERVE_PLANNED_CLUE", "SOLVE_CLUE"):
            return "clue"
        if action_name == "SELECT_PUZZLE":
            return "planning"
        # Find the puzzle to get its item type
        for p in room.puzzles:
            if p.puzzle_id.endswith(puzzle_id_short):
                return p.item.item_type.value
        structured = trace_entry.get(
            "output_structured", trace_entry.get("structured", {})
        )
        return structured.get("item_type", "")

    def _init_summary_csv(self) -> None:
        """Write CSV header."""
        with open(self.summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._CSV_COLUMNS)
            writer.writeheader()

    def _append_summary_csv(self, room, result: dict, trace_id: str = "") -> None:
        """Append one summary row per executed action, including tool calls."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_success = "escaped" if result["escaped"] else "failed"

        base = {
            "room_id":        room.room_id,
            "difficulty":     room.difficulty,
            "llm_model":      self.model,
            "temperature":    self.temperature,
            "timestamp":      ts,
            "task_success":   task_success,
            "trace_filename": f"{trace_id}.json",
        }

        with open(self.summary_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._CSV_COLUMNS)

            # No action entries (e.g. API timeout before any action completed).
            if not result["trace"]:
                writer.writerow({**base,
                    "action_type":    "LLM_action",
                    "puzzle_id":      "",
                    "action_id":      "API_TIMEOUT",
                    "action_name":    "API_TIMEOUT",
                    "action_status":  "wrong",
                    "error_type":     "API_TIMEOUT",
                    "anomaly_label":  1,
                    "puzzle_solved":  False,
                    "agent":          "",
                    "item_type":      "",
                    "answer":         "",
                    "gt":             "",
                    "desc":           "LLM call timed out before any action completed",
                    "fm_id":          "",
                })
                return

            puzzle_solved_map = {}
            for p in room.puzzles:
                pid = p.puzzle_id.split("_")[-1] if "_P" in p.puzzle_id else p.puzzle_id
                puzzle_solved_map[pid] = p.solved

            for entry in result["trace"]:
                short_pid = entry.get("puzzle_id", "")
                if "_P" in short_pid:
                    short_pid = short_pid.split("_")[-1]
                verification = entry.get("verification", {})
                action_name = entry.get("action", "")
                is_tool = action_name == "UNIT_TOOL_CALL"
                status = verification.get("status", "unverified")
                anomaly_label = (
                    verification.get("anomaly_label") if is_tool
                    else 0 if status == "correct" else 1 if status == "wrong" else None
                )
                tool_output = (
                    entry.get("tool_execution") or entry.get("output") or {}
                )
                writer.writerow({**base,
                    "action_type":       "tool_action" if is_tool else "LLM_action",
                    "puzzle_id":         short_pid,
                    "action_id":         entry.get("action_id", ""),
                    "action_name":       action_name,
                    "parent_action_id":  entry.get("parent_action_id", ""),
                    "agent":             entry.get("agent", ""),
                    "action_status":     status,
                    "error_type":        verification.get("error_type", ""),
                    "anomaly_label":     anomaly_label,
                    "puzzle_solved":     puzzle_solved_map.get(short_pid, False),
                    "item_type":         ("unit_tool" if is_tool else
                                          self._get_item_type(action_name, entry, room, short_pid)),
                    "answer":            (tool_output.get("result") if is_tool
                                          else verification.get("answer")),
                    "gt":                (verification.get("expected_result") if is_tool
                                          else verification.get("gt")),
                    "desc":              verification.get("desc", ""),
                    "fm_id":             "" if is_tool else verification.get("fm_id", ""),
                })

    @staticmethod
    def run_summarize(csv_path: str) -> None:
        """Read summary CSV with pandas and print aggregate statistics."""
        import pandas as pd

        df = pd.read_csv(csv_path)
        if df.empty:
            print("No data in summary CSV.")
            return

        type_column = (
            "action_type" if "action_type" in df.columns
            else "record_type" if "record_type" in df.columns
            else None
        )
        tool_rows = (
            df[df[type_column] == "tool_action"].copy()
            if type_column else df.iloc[0:0].copy()
        )
        if type_column:
            df = df[df[type_column] != "tool_action"].copy()

        # Per-room success (deduplicate: one room may have multiple failure rows)
        rooms = df.drop_duplicates(subset="room_id")[["room_id", "difficulty", "task_success"]]
        total = len(rooms)
        n_escaped = (rooms["task_success"] == "escaped").sum()

        print(f"\n{'═' * 60}")
        print(f"  RUN SUMMARY")
        print(f"{'═' * 60}")
        print(f"  Total rooms : {total}")
        print(f"  Escaped     : {n_escaped}/{total} ({n_escaped / total * 100:.1f}%)")

        # Per-puzzle success rate
        puzzles = df.drop_duplicates(subset=["room_id", "puzzle_id"])[["room_id", "puzzle_id", "puzzle_solved"]]
        total_p = len(puzzles)
        n_solved = (puzzles["puzzle_solved"] == True).sum()
        print(f"  Puzzles     : {n_solved}/{total_p} ({n_solved / total_p * 100:.1f}%)")

        # Per-difficulty success rate
        print(f"\n  By difficulty:")
        for diff in ("easy", "medium", "hard", "nightmare"):
            grp = rooms[rooms["difficulty"] == diff]
            if grp.empty:
                continue
            t = len(grp)
            e = (grp["task_success"] == "escaped").sum()
            print(f"    {diff:<8s}: {e}/{t} ({e / t * 100:.1f}%)")

        if not tool_rows.empty:
            n_tool_ok = (tool_rows["action_status"] == "correct").sum()
            print(f"\n  Tool actions: {n_tool_ok}/{len(tool_rows)} correct "
                  f"({n_tool_ok / len(tool_rows) * 100:.1f}%)")
            for error_type, grp in tool_rows.groupby("error_type", dropna=False):
                print(f"    {str(error_type):<28s}: {len(grp):>3}")

        # Failure rows only. Fall back to the old failure-only schema so
        # historical summary files remain readable.
        if "action_status" in df.columns:
            failures = df[df["action_status"] == "wrong"].copy()
            action_column = "action_name"
            agent_column = "agent" if "agent" in df.columns else "failed_agent"
        else:
            failures = df[df["failed_agent"].notna() & (df["failed_agent"] != "")].copy()
            action_column = "failed_action"
            agent_column = "failed_agent"
        if failures.empty:
            print(f"{'═' * 60}")
            return

        # ── Error attribution: own vs inherited ──────────────────────
        # First failure per (room_id, puzzle_id) is "own"; rest are "inherited".
        ACTION_ORDER = ["SELECT_PUZZLE", "OBSERVE_CLUE", "OBSERVE_PLANNED_CLUE",
                        "SOLVE_CLUE", "OBSERVE_ITEM", "APPLY_DELTA", "OBSERVE_PUZZLE"]
        failures["_rank"] = failures[action_column].map(
            {a: i for i, a in enumerate(ACTION_ORDER)}
        )
        first_rank = failures.groupby(["room_id", "puzzle_id"])["_rank"].transform("min")
        failures["propagation_type"] = (failures["_rank"] == first_rank).map(
            {True: "own", False: "inherited"}
        )

        n_own = (failures["propagation_type"] == "own").sum()
        n_inh = (failures["propagation_type"] == "inherited").sum()
        print(f"\n  Failures: {len(failures)} total ({n_own} own, {n_inh} inherited)")

        # By agent
        print(f"\n  Failure by agent:")
        for agent, grp in failures.groupby(agent_column):
            own = (grp["propagation_type"] == "own").sum()
            inh = (grp["propagation_type"] == "inherited").sum()
            print(f"    {agent:<16s}: {len(grp):>3} (own={own}, inherited={inh})")

        # By action
        print(f"\n  Failure by action:")
        for action in ACTION_ORDER:
            grp = failures[failures[action_column] == action]
            if grp.empty:
                continue
            own = (grp["propagation_type"] == "own").sum()
            inh = (grp["propagation_type"] == "inherited").sum()
            print(f"    {action:<16s}: {len(grp):>3} (own={own}, inherited={inh})")

        print(f"{'═' * 60}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    runner = EscapeRoomRunner(args)
    runner.run()
