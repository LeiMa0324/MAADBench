"""
DAG.py — Sequential planning DAG generator for EscapeRoom.

Framework 2: Directed graph on state space.

State  = frozenset of UNLOCKED items + frozenset of solved puzzles
Node   = State
Edge   = execute puzzle p (only if p.item is UNLOCKED and p not solved)
         → new state after applying p's unlock/lock effects

Goal   = all puzzles solved
Deadend= node with no outgoing edges and goal not reached

Guarantees (verified by construction):
  1. At least one path from S0 to goal exists
  2. At least one path from S0 leads to a deadend
  3. At least one first-step choice is a guaranteed trap (if min_trap_first_moves>0)
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

ITEMS = ["Clock", "Thermometer", "Compass", "Scale"]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PuzzleSpec:
    """Specification of one puzzle's planning-level properties."""
    puzzle_id:  str          # e.g. "P_clock"
    item_name:  str          # which item must be UNLOCKED to solve this
    unlocks:    List[str]    # items to set UNLOCKED after solving
    locks:      List[str]    # items to set LOCKED after solving

    def __hash__(self):
        return hash(self.puzzle_id)

    def __eq__(self, other):
        return self.puzzle_id == other.puzzle_id


@dataclass
class RoomDAG:
    """
    The planning graph for one room.

    Attributes
    ----------
    puzzles             : all 4 puzzle specs
    initial_state       : which items start UNLOCKED
    solution_paths      : list of valid orderings (each is a list of puzzle_ids)
    deadend_paths       : list of orderings that lead to deadend
    first_move_classes  : puzzle_id → "safe" | "trap" for first-step choices
                          "trap"  = every path starting with this puzzle → deadend
                          "safe"  = at least one solution starts with this puzzle
    """
    puzzles:             List[PuzzleSpec]
    initial_state:       FrozenSet[str]           # initially UNLOCKED items
    solution_paths:      List[List[str]]          # valid orderings
    deadend_paths:       List[List[str]]          # deadend orderings
    first_move_classes:  Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# State transition
# ---------------------------------------------------------------------------

State = Tuple[FrozenSet[str], FrozenSet[str]]
# (unlocked_items, solved_puzzles)


def apply_action(state: State, puzzle: PuzzleSpec) -> Optional[State]:
    """
    Apply puzzle action to state.
    Returns None if precondition not met.
    """
    unlocked, solved = state

    # Precondition: item must be unlocked, puzzle not yet solved
    if puzzle.item_name not in unlocked:
        return None
    if puzzle.puzzle_id in solved:
        return None

    # Apply effects
    new_unlocked = set(unlocked)
    for item in puzzle.unlocks:
        new_unlocked.add(item)
    for item in puzzle.locks:
        new_unlocked.discard(item)

    new_solved = solved | {puzzle.puzzle_id}
    return (frozenset(new_unlocked), frozenset(new_solved))


def is_goal(state: State, puzzles: List[PuzzleSpec]) -> bool:
    _, solved = state
    return len(solved) == len(puzzles)


def available_actions(state: State, puzzles: List[PuzzleSpec]) -> List[PuzzleSpec]:
    unlocked, solved = state
    return [
        p for p in puzzles
        if p.puzzle_id not in solved
        and p.item_name in unlocked
    ]


# ---------------------------------------------------------------------------
# Graph exploration
# ---------------------------------------------------------------------------

def explore(
    initial_state: State,
    puzzles: List[PuzzleSpec],
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    BFS/DFS over state space.

    Returns (solution_paths, deadend_paths).
    Each path is a list of puzzle_ids in execution order.
    """
    solution_paths: List[List[str]] = []
    deadend_paths:  List[List[str]] = []

    # DFS with path tracking
    # stack: (state, path_so_far)
    stack = [(initial_state, [])]
    visited_paths: Set[Tuple[str, ...]] = set()

    while stack:
        state, path = stack.pop()

        path_key = tuple(path)
        if path_key in visited_paths:
            continue
        visited_paths.add(path_key)

        if is_goal(state, puzzles):
            solution_paths.append(path)
            continue

        actions = available_actions(state, puzzles)

        if not actions:
            # No available actions but goal not reached → deadend
            deadend_paths.append(path)
            continue

        for action in actions:
            new_state = apply_action(state, action)
            if new_state is not None:
                stack.append((new_state, path + [action.puzzle_id]))

    return solution_paths, deadend_paths


def classify_first_moves(
    initial_state: State,
    puzzles: List[PuzzleSpec],
) -> Dict[str, str]:
    """
    For each puzzle available at the first step, classify as:
      "safe"  — at least one solution path starts with this puzzle
      "trap"  — every path starting with this puzzle leads to deadend

    A "trap" first move means: regardless of what the planner does
    after choosing this puzzle, the room cannot be completed.
    """
    result: Dict[str, str] = {}
    for first in available_actions(initial_state, puzzles):
        next_state = apply_action(initial_state, first)
        if next_state is None:
            result[first.puzzle_id] = "trap"
            continue
        sub_sols, _ = explore(next_state, puzzles)
        result[first.puzzle_id] = "safe" if sub_sols else "trap"
    return result


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def _random_puzzle_spec(item_name: str, other_items: List[str]) -> PuzzleSpec:
    """Generate random unlock/lock effects for one puzzle."""
    # unlock 1-2 other items
    n_unlock = random.randint(1, min(2, len(other_items)))
    unlocks = random.sample(other_items, k=n_unlock)

    # lock 0-1 items that are NOT being unlocked
    lockable = [i for i in other_items if i not in unlocks]
    n_lock = random.randint(0, min(1, len(lockable)))
    locks = random.sample(lockable, k=n_lock) if lockable else []

    return PuzzleSpec(
        puzzle_id = f"P_{item_name.lower()}",
        item_name = item_name,
        unlocks   = unlocks,
        locks     = locks,
    )


def generate_room_dag(
    items: List[str] = ITEMS,
    min_solutions: int = 1,
    min_deadends: int = 1,
    min_deadend_depth: int = 2,
    min_trap_first_moves: int = 1,
    max_attempts: int = 10_000,
    seed: Optional[int] = None,
) -> RoomDAG:
    """
    Generate a room DAG that satisfies:
      1. At least one solution path exists
      2. At least one deadend path exists
      3. At least one deadend path has length >= min_deadend_depth
         (planner must make min_deadend_depth correct moves before
          hitting a deadend — trivial first-step blocks are excluded)
      4. At least min_trap_first_moves first-step choices are guaranteed
         traps: every path starting with that puzzle leads to deadend,
         no matter what the planner does afterwards.
         (Always requires at least one "safe" first move too.)

    Parameters
    ----------
    min_solutions : int
        Minimum number of valid solving orderings required (default 1).
    min_deadends : int
        Minimum number of deadend orderings required (default 1).
    min_deadend_depth : int
        Minimum steps taken before hitting a deadend (default 2).
        Depth 0 = blocked at step 1 (trivial, no planning needed).
        Depth 1 = one correct step then blocked.
        Depth 2 = two correct steps then blocked (default minimum).
        Depth 3 = three correct steps then blocked (hardest).
    min_trap_first_moves : int
        Minimum first-step choices that are guaranteed traps (default 1).
        Set to 0 to disable. Higher values make the first choice harder.

    Uses rejection sampling: generate random lock/unlock relations,
    explore the full state graph, accept if all thresholds met.
    """
    if seed is not None:
        random.seed(seed)

    for attempt in range(max_attempts):
        # Random initial state: 1-(N-1) items UNLOCKED
        n_initial = random.randint(1, len(items) - 1)
        initial_unlocked = frozenset(random.sample(items, k=n_initial))

        # Generate puzzle specs
        puzzles = []
        for item in items:
            other = [i for i in items if i != item]
            spec = _random_puzzle_spec(item, other)
            puzzles.append(spec)

        initial_state: State = (initial_unlocked, frozenset())

        # Explore graph
        sol_paths, dead_paths = explore(initial_state, puzzles)

        # Filter deadend paths by minimum depth
        # len(path) = number of puzzles successfully solved before deadend
        deep_dead_paths = [p for p in dead_paths if len(p) >= min_deadend_depth]

        # Classify first moves
        first_classes = classify_first_moves(initial_state, puzzles)
        n_trap_first  = sum(1 for v in first_classes.values() if v == "trap")
        n_safe_first  = sum(1 for v in first_classes.values() if v == "safe")

        # Accept if all conditions satisfied
        if (len(sol_paths) >= min_solutions
                and len(dead_paths) >= min_deadends
                and deep_dead_paths
                and n_trap_first >= min_trap_first_moves
                and n_safe_first >= 1):
            return RoomDAG(
                puzzles            = puzzles,
                initial_state      = initial_unlocked,
                solution_paths     = sol_paths,
                deadend_paths      = dead_paths,
                first_move_classes = first_classes,
            )

    raise RuntimeError(
        f"Failed to generate valid room DAG after {max_attempts} attempts "
        f"(min_solutions={min_solutions}, min_deadends={min_deadends}, "
        f"min_deadend_depth={min_deadend_depth}, "
        f"min_trap_first_moves={min_trap_first_moves})."
    )


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def print_dag_summary(dag: RoomDAG) -> None:
    """Print a human-readable summary of the room DAG."""
    W = 64
    bar = "─" * W

    print(f"\n{bar}")
    print(f"  ROOM DAG SUMMARY")
    print(f"{bar}")

    print(f"\n  Initial UNLOCKED: {sorted(dag.initial_state)}")
    print(f"  Initial LOCKED:   {sorted(set(ITEMS) - dag.initial_state)}")

    print(f"\n  Puzzle Effects:")
    for p in dag.puzzles:
        avail = "✓" if p.item_name in dag.initial_state else "✗"
        cls   = dag.first_move_classes.get(p.puzzle_id, "-")
        tag   = f"[{cls}]" if cls != "-" else ""
        print(f"    [{avail}] {p.puzzle_id:<16} "
              f"requires={p.item_name:<14} "
              f"unlocks={p.unlocks}  locks={p.locks}  {tag}")

    print(f"\n  First move classification:")
    for pid, cls in dag.first_move_classes.items():
        icon = "✗ TRAP" if cls == "trap" else "✓ safe"
        print(f"    {pid:<20} → {icon}")

    print(f"\n  Solution paths  : {len(dag.solution_paths)}")
    for path in dag.solution_paths:
        print(f"    {' → '.join(path)}")
    if len(dag.solution_paths) > 3:
        print(f"    ... and {len(dag.solution_paths) - 3} more")

    print(f"\n  Deadend paths   : {len(dag.deadend_paths)}")
    for path in dag.deadend_paths:
        print(f"    {' → '.join(path)}  ✗")
    if len(dag.deadend_paths) > 3:
        print(f"    ... and {len(dag.deadend_paths) - 3} more")

    print(f"{bar}")


def trace_path(
    path: List[str],
    dag: RoomDAG,
) -> None:
    """Print step-by-step state trace for a given path."""
    puzzle_map = {p.puzzle_id: p for p in dag.puzzles}
    unlocked = set(dag.initial_state)
    solved   = set()

    print(f"\n  Path: {' → '.join(path)}")
    print(f"  Initial: UNLOCKED={sorted(unlocked)}")

    for pid in path:
        p = puzzle_map[pid]
        ok = p.item_name in unlocked and pid not in solved
        status = "✓" if ok else "✗ BLOCKED"
        print(f"\n  → {pid} [{status}]")
        if ok:
            for item in p.unlocks:
                unlocked.add(item)
            for item in p.locks:
                unlocked.discard(item)
            solved.add(pid)
            print(f"    UNLOCKED now: {sorted(unlocked)}")
        else:
            print(f"    Cannot solve — {p.item_name} is LOCKED")
            print(f"    DEADEND reached after {len(solved)} puzzles")
            break

    if len(solved) == len(dag.puzzles):
        print(f"\n  ✓ ALL PUZZLES SOLVED — ESCAPED!")
    elif ok:
        pass  # deadend already printed
    print()


# ---------------------------------------------------------------------------
# Room desc for Planner
# ---------------------------------------------------------------------------

def room_desc_for_planner(
    dag: RoomDAG,
    current_unlocked: Set[str],
    solved: Set[str],
) -> str:
    """
    Generate the room description shown to the Planner at each step.
    Shows item states, available puzzles (with effects), and locked puzzles.
    """
    lines = ["=== ROOM STATUS ===\n"]

    # Item states
    lines.append("ITEMS:")
    for item in ITEMS:
        state = "UNLOCKED ✓" if item in current_unlocked else "LOCKED   ✗"
        lines.append(f"  {item:<16}: {state}")

    # Available puzzles
    lines.append("\nAVAILABLE PUZZLES (you may solve these now):")
    available = [
        p for p in dag.puzzles
        if p.puzzle_id not in solved
        and p.item_name in current_unlocked
    ]
    if available:
        for p in available:
            lines.append(f"  [{p.puzzle_id}]")
            lines.append(f"    requires : {p.item_name} = UNLOCKED ✓")
            lines.append(f"    if solved: unlock {p.unlocks}"
                         + (f", lock {p.locks}" if p.locks else ""))
    else:
        lines.append("  (none — deadend!)")

    # Locked puzzles
    lines.append("\nLOCKED PUZZLES (dependencies not met):")
    locked = [
        p for p in dag.puzzles
        if p.puzzle_id not in solved
        and p.item_name not in current_unlocked
    ]
    if locked:
        for p in locked:
            lines.append(f"  [{p.puzzle_id}]  requires {p.item_name} = UNLOCKED ✗")
            lines.append(f"    if solved: unlock {p.unlocks}"
                         + (f", lock {p.locks}" if p.locks else ""))
    else:
        lines.append("  (none)")

    lines.append("\nGoal: solve ALL puzzles to escape.")
    lines.append("Warning: wrong order may permanently lock required items.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from collections import Counter

    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    W = 64
    SEP = "─" * W

    # ── 1. Generate & summary ─────────────────────────────────────────────
    print(f"\nGenerating room DAG  seed={seed} ...")
    dag = generate_room_dag(seed=seed)
    print_dag_summary(dag)

    # ── 2. First-move classification ──────────────────────────────────────
    print(f"\n{'═' * W}")
    print("  FIRST-MOVE ANALYSIS")
    print(f"{'═' * W}")
    initial_state: State = (dag.initial_state, frozenset())
    for pid, cls in dag.first_move_classes.items():
        spec = next(p for p in dag.puzzles if p.puzzle_id == pid)
        if cls == "trap":
            print(f"  {pid:<20} ✗ TRAP  — every continuation leads to deadend")
        else:
            print(f"  {pid:<20} ✓ SAFE  — at least one solution reachable")
            ns = apply_action(initial_state, spec)
            sub_sols, _ = explore(ns, dag.puzzles)
            for s in sub_sols:
                print(f"      {pid} → {' → '.join(s)}")

    n_trap = sum(1 for v in dag.first_move_classes.values() if v == "trap")
    n_safe = sum(1 for v in dag.first_move_classes.values() if v == "safe")
    n_avail = len(dag.first_move_classes)
    print(f"\n  {n_avail} available at step 1:  {n_trap} trap  |  {n_safe} safe")
    print(f"  Random-guess success rate at step 1: "
          f"{n_safe}/{n_avail} = {n_safe / n_avail * 100:.0f}%")

    # ── 3. Trace all solution paths ───────────────────────────────────────
    print(f"\n{'═' * W}")
    print(f"  ALL SOLUTION PATHS  ({len(dag.solution_paths)} total)")
    print(f"{'═' * W}")
    for path in dag.solution_paths:
        trace_path(path, dag)

    # ── 4. Trace deepest deadend path ─────────────────────────────────────
    deepest = max(dag.deadend_paths, key=len)
    print(f"{'═' * W}")
    print(f"  DEEPEST DEADEND PATH  (depth={len(deepest)})")
    print(f"{'═' * W}")
    trace_path(deepest, dag)

    # ── 5. Deadend depth distribution ────────────────────────────────────
    print(f"{'═' * W}")
    print("  DEADEND DEPTH DISTRIBUTION")
    print(f"{'═' * W}")
    depth_dist = Counter(len(p) for p in dag.deadend_paths)
    for depth in sorted(depth_dist):
        bar = "█" * depth_dist[depth]
        print(f"  depth {depth}: {bar} ({depth_dist[depth]})")

    # ── 6. Planner room description (step 0) ─────────────────────────────
    print(f"\n{'═' * W}")
    print("  PLANNER ROOM DESC  (initial state, step 0)")
    print(f"{'═' * W}")
    print(room_desc_for_planner(dag, set(dag.initial_state), set()))

    # ── 7. Planner room description after first safe move ─────────────────
    safe_pid = next(pid for pid, cls in dag.first_move_classes.items()
                    if cls == "safe")
    safe_spec = next(p for p in dag.puzzles if p.puzzle_id == safe_pid)
    ns = apply_action(initial_state, safe_spec)
    unlocked_after, solved_after = ns
    print(f"\n{'═' * W}")
    print(f"  PLANNER ROOM DESC  (after solving {safe_pid})")
    print(f"{'═' * W}")
    print(room_desc_for_planner(dag, set(unlocked_after), set(solved_after)))
