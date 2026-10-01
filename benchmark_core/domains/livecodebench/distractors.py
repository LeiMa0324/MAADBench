"""LiveCodeBench code-shaped distractor factory."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Optional

DEFAULT_DISTRACTORS = (
    Path(__file__).resolve().parent
    / "data"
    / "code_chain_distractors.jsonl"
)

_POOL: Optional[List[dict]] = None


def _load_pool(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Distractor pool not found: {path}\n"
            "Generate it with domains/livecodebench/scripts/make_distractors.py"
        )
    pool = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pool.append(json.loads(line))
    if not pool:
        raise ValueError(f"No distractors found in {path}")
    return pool


def _plausible_hint() -> str:
    """A hint that looks exactly like a real clue's (real instrument + valid unit).

    Making the fake hint plausible means it gives NOTHING away — the only thing
    that distinguishes a fake from a real clue is that the fake's code is invalid.
    """
    from benchmark_core.room_core import (
        ITEM_HINTS, UNIT_HINTS, PuzzleItemType,
        Thermometer, Compass, Clock, Scale,
    )
    unit_map = {
        PuzzleItemType.THERMOMETER: list(Thermometer._TO_BASE.keys()),
        PuzzleItemType.COMPASS: list(Compass._TO_BASE.keys()),
        PuzzleItemType.CLOCK: list(Clock._TO_BASE.keys()),
        PuzzleItemType.SCALE: list(Scale._TO_BASE.keys()),
    }
    it = random.choice(list(PuzzleItemType))
    du = random.choice(unit_map[it])
    return f"Hint: {random.choice(ITEM_HINTS[it])}, counted in {random.choice(UNIT_HINTS[du])}."


def _fake_note(rec: dict, fake_hint: str) -> str:
    """Format one distractor: a 2-chain note (same layout as real) with one broken
    program, plus a plausible hint — so only code validity distinguishes it."""
    from benchmark_core.domains.livecodebench.problem_source import format_chain_note
    note = format_chain_note(
        rec.get("code_a", ""), rec.get("call_a", ""),
        rec.get("code_b", ""), rec.get("call_b_template", ""), rec.get("slot", ""),
    )
    return f"{note}\n{fake_hint}"


def create_fake_clue(clue_class, hint: Optional[str] = None,
                     jsonl_path: Optional[str | Path] = None):
    """Create one code-shaped fake clue without changing global class methods."""
    global _POOL
    if _POOL is None or jsonl_path is not None:
        path = Path(jsonl_path) if jsonl_path else DEFAULT_DISTRACTORS
        _POOL = _load_pool(path)
    rec = random.choice(_POOL)
    fake_hint = hint if hint is not None else _plausible_hint()
    return clue_class(
        is_fake=True,
        hint=_fake_note(rec, fake_hint),
        domain="livecodebench",
    )
