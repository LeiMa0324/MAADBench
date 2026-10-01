"""
plan_main.py — Demo: PlanningRoom with puzzle_selection + existing action pipeline.

Flow per iteration:
  1. SELECT_PUZZLE  (planner)    → choose which puzzle to solve next
  2. OBSERVE_CLUE   (observer)   → identify the real clue
  3. SOLVE_CLUE     (clue_solver)→ solve the math problem
  4. OBSERVE_ITEM   (observer)   → identify the real instrument
  5. APPLY_DELTA    (item_manager)→ compute new reading, submit answer

Repeats until all puzzles solved, deadend, or wrong answer.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from benchmark_core.plan_room import PlanningRoom, _item_name
from benchmark_core.action_prompts import (
    Action, ActionResult, OutputField,
    OBSERVE_CLUE, SOLVE_CLUE, OBSERVE_ITEM, APPLY_DELTA,
)


# ---------------------------------------------------------------------------
# New action: SELECT_PUZZLE
# ---------------------------------------------------------------------------

SELECT_PUZZLE = Action(
    name       = "SELECT_PUZZLE",
    agent_role = "planner",

    context_keys     = ["planning_desc"],
    prev_output_keys = [],

    prompt_template = """You are a planner in an escape room with multiple puzzles.
Each puzzle is tied to an item. Items can be UNLOCKED or LOCKED.
You can only solve puzzles whose item is UNLOCKED.
Solving a puzzle may unlock or lock other items — choose wisely.

{planning_desc}

Your task: choose ONE puzzle to solve next.
Consider the unlock/lock effects carefully — a wrong ordering
may permanently lock required items (deadend).

Return a JSON object with exactly these two keys:

"message": "<your reasoning about which puzzle to solve and why>",
"structured": {{
  "puzzle_id": "<the puzzle_id you choose to solve next>"
}}
""",

    output_schema = [
        OutputField("puzzle_id", str, "the puzzle_id to solve next"),
    ],
)


# ---------------------------------------------------------------------------
# Action pipeline (per puzzle)
# ---------------------------------------------------------------------------

PUZZLE_ACTIONS = [OBSERVE_CLUE, SOLVE_CLUE, OBSERVE_ITEM, APPLY_DELTA]


# ---------------------------------------------------------------------------
# Simulated agent responses (deterministic, uses ground truth)
# ---------------------------------------------------------------------------

def _simulate_select_puzzle(room: PlanningRoom) -> ActionResult:
    """Simulate planner: pick the first available puzzle (from a valid solution path)."""
    avail = room.available_puzzles()
    if not avail:
        return ActionResult(
            action_name="SELECT_PUZZLE",
            agent_role="planner",
            structured_output={"puzzle_id": ""},
            message="No puzzles available — deadend!",
            schema_errors={},
        )

    # Pick first available (greedy)
    chosen = avail[0]
    return ActionResult(
        action_name="SELECT_PUZZLE",
        agent_role="planner",
        structured_output={"puzzle_id": chosen.puzzle_id},
        message=f"Choosing {chosen.puzzle_id} (item={_item_name(chosen)}, UNLOCKED).",
        schema_errors={},
    )


def _simulate_observe_clue(puzzle) -> ActionResult:
    """Simulate observer: return the real clue."""
    return ActionResult(
        action_name="OBSERVE_CLUE",
        agent_role="observer",
        structured_output={
            "item_id": puzzle.clue.item_id,
            "clue": str(puzzle.clue),
        },
        message=f"Found clue {puzzle.clue.item_id}: {puzzle.clue}",
        schema_errors={},
    )


def _simulate_solve_clue(puzzle) -> ActionResult:
    """Simulate clue_solver: return the correct answer."""
    return ActionResult(
        action_name="SOLVE_CLUE",
        agent_role="clue_solver",
        structured_output={
            "answer": puzzle.clue.answer,
            "unit": puzzle.clue.delta_unit,
        },
        message=f"Solved: answer={puzzle.clue.answer} {puzzle.clue.delta_unit}",
        schema_errors={},
    )


def _simulate_observe_item(puzzle) -> ActionResult:
    """Simulate observer: return the real item."""
    item = puzzle.item
    return ActionResult(
        action_name="OBSERVE_ITEM",
        agent_role="observer",
        structured_output={
            "item_index": 1,
            "item_type": item.item_type.value,
            "item_state": item.state,
            "item_unit": item.unit,
            "delta": puzzle.clue.answer,
            "delta_unit": puzzle.clue.delta_unit,
        },
        message=f"Found {item.item_type.value} (state={item.state} {item.unit})",
        schema_errors={},
    )


def _simulate_apply_delta(puzzle) -> ActionResult:
    """Simulate item_manager: compute ground truth answer."""
    gt = puzzle.item._groundtruth_answer(puzzle.clue.answer, puzzle.clue.delta_unit)
    return ActionResult(
        action_name="APPLY_DELTA",
        agent_role="item_manager",
        structured_output={
            "answer": gt,
            "converted_delta": puzzle.clue.answer,
            "converted_unit": puzzle.clue.delta_unit,
        },
        message=f"Applied delta → answer={gt}",
        schema_errors={},
    )


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def run_planning_room(difficulty: str = "medium", seed: int = 42):
    """Simulate a full PlanningRoom run with puzzle selection."""
    room = PlanningRoom.create(difficulty=difficulty, room_id="plan_demo", seed=seed)

    W = 64
    bar = "═" * W
    thin = "─" * W

    print(f"\n{bar}")
    print(f"  PLANNING ROOM SIMULATION")
    print(f"  difficulty={difficulty}  puzzles={len(room.puzzles)}")
    print(bar)

    iteration = 0
    max_iterations = len(room.puzzles) + 2  # safety bound

    while not room.completed and iteration < max_iterations:
        iteration += 1

        print(f"\n{thin}")
        print(f"  ITERATION {iteration}")
        print(thin)

        # Check for deadend
        if room.leads_to_deadend:
            print("  ⚠ DEADEND — no puzzles available. Room cannot be escaped.")
            break

        # ── Step 0: Planning description ──
        print("\n  [PLANNING DESC]")
        print(room.planning_desc())

        # ── Step 1: SELECT_PUZZLE ──
        select_result = _simulate_select_puzzle(room)
        chosen_id = select_result.structured_output["puzzle_id"]
        print(f"\n  [SELECT_PUZZLE] → {chosen_id}")
        print(f"  Reasoning: {select_result.message}")

        if not chosen_id:
            print("  No puzzle selected — stopping.")
            break

        puzzle = room.get_puzzle(chosen_id)
        if puzzle is None:
            print(f"  ⚠ Puzzle {chosen_id} not found!")
            break

        # ── Room desc before pipeline (what observer agents see) ──
        print(f"\n  [ROOM DESC — before pipeline]")
        print(room.desc())

        # ── Step 2-5: Standard pipeline ──
        print(f"\n  [OBSERVE_CLUE]")
        obs_clue = _simulate_observe_clue(puzzle)
        print(f"    {obs_clue.message}")

        print(f"  [SOLVE_CLUE]")
        solve = _simulate_solve_clue(puzzle)
        print(f"    {solve.message}")

        print(f"  [OBSERVE_ITEM]")
        obs_item = _simulate_observe_item(puzzle)
        print(f"    {obs_item.message}")

        print(f"  [APPLY_DELTA]")
        apply = _simulate_apply_delta(puzzle)
        print(f"    {apply.message}")

        # ── Evaluate ──
        answer = apply.structured_output["answer"]
        result = room.update_item(chosen_id, answer)
        print(f"\n  [EVAL] {chosen_id} → {result}")

        if result == "correct":
            print(f"  ✓ Puzzle solved! Progress: {len(room.solved_ids)}/{len(room.puzzles)}")
            print(f"  Item states: {[k for k, v in room.item_states.items() if v]}")

            # ── Room desc after update (what agents see next round) ──
            print(f"\n  [ROOM DESC — after update]")
            print(room.desc())
        else:
            print(f"  ✗ Wrong answer — stopping.")
            break

    # ── Final result ──
    print(f"\n{bar}")
    if room.completed:
        print(f"  ✓ ESCAPED! All {len(room.puzzles)} puzzles solved.")
    elif room.leads_to_deadend:
        print(f"  ✗ DEADEND after solving {len(room.solved_ids)}/{len(room.puzzles)} puzzles.")
    else:
        print(f"  ✗ FAILED. Solved {len(room.solved_ids)}/{len(room.puzzles)} puzzles.")
    print(bar)


if __name__ == "__main__":
    import sys
    difficulty = sys.argv[1] if len(sys.argv) > 1 else "medium"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    run_planning_room(difficulty=difficulty, seed=seed)
