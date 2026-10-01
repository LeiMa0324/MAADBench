"""Optional Weights & Biases progress tracking for benchmark runs."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


class SuccessMetrics:
    """Accumulate room, puzzle, and recorded action success metrics."""

    def __init__(self) -> None:
        self.rooms = self.escaped = 0
        self.puzzles = self.puzzles_solved = 0
        self.actions = self.actions_correct = 0
        self.llm_actions = self.llm_actions_correct = 0
        self.tool_actions = self.tool_actions_correct = 0
        self.by_difficulty: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    @staticmethod
    def _action_counts(trace: list[dict[str, Any]]) -> dict[str, int]:
        counts = {
            "actions": 0, "actions_correct": 0,
            "llm_actions": 0, "llm_actions_correct": 0,
            "tool_actions": 0, "tool_actions_correct": 0,
        }
        # summary.csv records an API timeout as one failed LLM action when no
        # trace entry exists. Mirror that definition in the live metrics.
        if not trace:
            counts["actions"] = 1
            counts["llm_actions"] = 1
            return counts

        for entry in trace:
            status = (entry.get("verification") or {}).get("status", "unverified")
            is_tool = entry.get("action") == "UNIT_TOOL_CALL"
            prefix = "tool" if is_tool else "llm"
            counts["actions"] += 1
            counts[f"{prefix}_actions"] += 1
            if status == "correct":
                counts["actions_correct"] += 1
                counts[f"{prefix}_actions_correct"] += 1
        return counts

    def update(self, difficulty: str, result: dict[str, Any]) -> dict[str, float | int]:
        action_counts = self._action_counts(result.get("trace") or [])
        escaped = int(bool(result.get("escaped")))
        puzzle_total = int(result.get("puzzles_total") or 0)
        puzzle_solved = int(result.get("puzzles_solved") or 0)

        self.rooms += 1
        self.escaped += escaped
        self.puzzles += puzzle_total
        self.puzzles_solved += puzzle_solved
        for key, value in action_counts.items():
            setattr(self, key, getattr(self, key) + value)

        diff = self.by_difficulty[difficulty]
        diff["rooms"] += 1
        diff["escaped"] += escaped
        diff["puzzles"] += puzzle_total
        diff["puzzles_solved"] += puzzle_solved
        diff["actions"] += action_counts["actions"]
        diff["actions_correct"] += action_counts["actions_correct"]

        metrics: dict[str, float | int] = {
            "room/current_success": escaped,
            "room/completed": self.rooms,
            "room/escaped": self.escaped,
            "room/success_rate": _rate(self.escaped, self.rooms),
            "puzzle/current_solved": puzzle_solved,
            "puzzle/current_total": puzzle_total,
            "puzzle/solved": self.puzzles_solved,
            "puzzle/total": self.puzzles,
            "puzzle/success_rate": _rate(self.puzzles_solved, self.puzzles),
            "action/current_correct": action_counts["actions_correct"],
            "action/current_total": action_counts["actions"],
            "action/current_success_rate": _rate(
                action_counts["actions_correct"], action_counts["actions"]
            ),
            "action/correct": self.actions_correct,
            "action/total": self.actions,
            "action/success_rate": _rate(self.actions_correct, self.actions),
            "action/llm_correct": self.llm_actions_correct,
            "action/llm_total": self.llm_actions,
            "action/llm_success_rate": _rate(
                self.llm_actions_correct, self.llm_actions
            ),
            "action/tool_correct": self.tool_actions_correct,
            "action/tool_total": self.tool_actions,
            "action/tool_success_rate": _rate(
                self.tool_actions_correct, self.tool_actions
            ),
        }
        for level, values in self.by_difficulty.items():
            prefix = f"difficulty/{level}"
            metrics.update({
                f"{prefix}/rooms_completed": values["rooms"],
                f"{prefix}/room_success_rate": _rate(
                    values["escaped"], values["rooms"]
                ),
                f"{prefix}/puzzle_success_rate": _rate(
                    values["puzzles_solved"], values["puzzles"]
                ),
                f"{prefix}/action_success_rate": _rate(
                    values["actions_correct"], values["actions"]
                ),
            })
        return metrics


class WandbTracker:
    """Own one W&B run and upload live metrics plus the completed run files."""

    def __init__(
        self,
        *,
        project: str,
        entity: str | None,
        group: str | None,
        name: str,
        mode: str,
        run_dir: Path,
        config: dict[str, Any],
        tags: list[str],
        upload_artifacts: bool,
    ) -> None:
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "W&B tracking requested, but wandb is not installed. "
                "Run: python -m pip install -r requirements.txt"
            ) from exc

        self.wandb = wandb
        self.run_dir = run_dir
        self.upload_artifacts = upload_artifacts
        self.metrics = SuccessMetrics()
        self.run = wandb.init(
            project=project,
            entity=entity,
            group=group,
            name=name,
            job_type="benchmark",
            mode=mode,
            dir=str(run_dir),
            config=config,
            tags=tags,
        )
        self.run.log({"room/completed": 0}, step=0)

    @property
    def url(self) -> str | None:
        return getattr(self.run, "url", None)

    def log_room(
        self, *, difficulty: str, result: dict[str, Any], room_index: int,
        total_rooms: int,
    ) -> None:
        metrics = self.metrics.update(difficulty, result)
        metrics["room/total"] = total_rooms
        metrics["progress/fraction"] = _rate(room_index, total_rooms)
        self.run.log(metrics, step=room_index)

    def finish(self, status: str, exit_code: int) -> None:
        self.run.summary["run_status"] = status
        self.run.summary["exit_code"] = exit_code
        if self.upload_artifacts:
            artifact = self.wandb.Artifact(
                name=f"{self.run.name}-traces", type="benchmark-traces"
            )
            for path in self.run_dir.iterdir():
                if path.is_file() and (
                    path.suffix == ".json" or path.name in {"summary.csv", "run.log"}
                ):
                    artifact.add_file(str(path), name=path.name)
            self.run.log_artifact(artifact)
        self.run.finish(exit_code=exit_code)
