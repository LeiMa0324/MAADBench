from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from benchmark_core.room_core import (
    Room, Puzzle, PuzzleItem, PuzzleItemType, Clue,
    SceneryItem, TrapType, FakeItemTrap, FakeClueTrap,
)


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
    first_move_classes  : puzzle_id -> "safe" | "trap" for first-step choices
                          "trap" = every path starting with this puzzle -> deadend
                          "safe" = at least one solution starts with this puzzle
    """
    puzzles:            List[PuzzleSpec]
    initial_state:      FrozenSet[str]
    solution_paths:     List[List[str]]
    deadend_paths:      List[List[str]]
    first_move_classes: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# State transition
# ---------------------------------------------------------------------------

State = Tuple[FrozenSet[str], FrozenSet[str]]
# (unlocked_items, solved_puzzles)


def apply_action(state: State, puzzle: PuzzleSpec) -> Optional[State]:
    unlocked, solved = state
    if puzzle.item_name not in unlocked:
        return None
    if puzzle.puzzle_id in solved:
        return None
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
    solution_paths: List[List[str]] = []
    deadend_paths:  List[List[str]] = []
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

def _random_puzzle_spec(item_name: str, other_items: List[str],
                        hard_locks: bool = False) -> PuzzleSpec:
    n_unlock = random.randint(1, min(2, len(other_items)))
    unlocks = random.sample(other_items, k=n_unlock)
    lockable = [i for i in other_items if i not in unlocks]
    if hard_locks:
        n_lock = random.randint(1, min(2, len(lockable))) if lockable else 0
    else:
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
    max_solutions: int = 0,
    min_deadends: int = 1,
    min_deadend_depth: int = 2,
    min_trap_first_moves: int = 1,
    n_initial: int = 0,
    hard_locks: bool = False,
    max_attempts: int = 50_000,
    seed: Optional[int] = None,
) -> RoomDAG:
    """
    Generate a room DAG that satisfies:
      1. At least one solution path exists
      2. At least one deadend path exists
      3. At least one deadend path has length >= min_deadend_depth
      4. At least min_trap_first_moves first-step choices are guaranteed
         traps: every path starting with that puzzle leads to deadend,
         no matter what the planner does afterwards.
         (Always requires at least one "safe" first move too.)

    Parameters
    ----------
    min_solutions : int
        Minimum number of valid solving orderings required (default 1).
    max_solutions : int
        Maximum number of valid solving orderings allowed (0 = unlimited).
    min_deadends : int
        Minimum number of deadend orderings required (default 1).
    min_deadend_depth : int
        Minimum steps taken before hitting a deadend (default 2).
    min_trap_first_moves : int
        Minimum first-step choices that are guaranteed traps (default 1).
        Set to 0 to disable. Higher values make the first choice harder.
    n_initial : int
        Exact number of initially unlocked items (0 = random 1..len-1).
    hard_locks : bool
        If True, each puzzle locks 1-2 items instead of 0-1.
    """
    if seed is not None:
        random.seed(seed)

    for attempt in range(max_attempts):
        if n_initial > 0:
            ni = n_initial
        else:
            ni = random.randint(1, len(items) - 1)
        initial_unlocked = frozenset(random.sample(items, k=ni))

        puzzles = []
        for item in items:
            other = [i for i in items if i != item]
            spec = _random_puzzle_spec(item, other, hard_locks=hard_locks)
            puzzles.append(spec)

        initial_state: State = (initial_unlocked, frozenset())
        sol_paths, dead_paths = explore(initial_state, puzzles)
        deep_dead_paths = [p for p in dead_paths if len(p) >= min_deadend_depth]

        # Classify first moves
        first_classes = classify_first_moves(initial_state, puzzles)
        n_trap_first  = sum(1 for v in first_classes.values() if v == "trap")
        n_safe_first  = sum(1 for v in first_classes.values() if v == "safe")

        if (len(sol_paths) >= min_solutions
                and (max_solutions == 0 or len(sol_paths) <= max_solutions)
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
        f"(min_solutions={min_solutions}, max_solutions={max_solutions}, "
        f"min_deadends={min_deadends}, "
        f"min_deadend_depth={min_deadend_depth}, "
        f"min_trap_first_moves={min_trap_first_moves})."
    )


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def print_dag_summary(dag: RoomDAG) -> None:
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
    for path in dag.solution_paths[:3]:
        print(f"    {' → '.join(path)}")
    if len(dag.solution_paths) > 3:
        print(f"    ... and {len(dag.solution_paths) - 3} more")
    print(f"\n  Deadend paths   : {len(dag.deadend_paths)}")
    for path in dag.deadend_paths[:3]:
        print(f"    {' → '.join(path)}  ✗")
    if len(dag.deadend_paths) > 3:
        print(f"    ... and {len(dag.deadend_paths) - 3} more")
    print(f"{bar}")


def trace_path(path: List[str], dag: RoomDAG) -> None:
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
    print()


def room_desc_for_planner(
    dag: RoomDAG,
    current_unlocked: Set[str],
    solved: Set[str],
) -> str:
    lines = ["=== ROOM STATUS ===\n"]
    lines.append("ITEMS:")
    for item in ITEMS:
        state = "UNLOCKED ✓" if item in current_unlocked else "LOCKED   ✗"
        lines.append(f"  {item:<16}: {state}")
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
# Item name <-> PuzzleItemType mapping
# ---------------------------------------------------------------------------

_TYPE_TO_NAME: Dict[PuzzleItemType, str] = {
    PuzzleItemType.CLOCK:        "Clock",
    PuzzleItemType.THERMOMETER:  "Thermometer",
    PuzzleItemType.COMPASS:      "Compass",
    PuzzleItemType.SCALE:        "Scale",
}

_NAME_TO_TYPE: Dict[str, PuzzleItemType] = {v: k for k, v in _TYPE_TO_NAME.items()}


def _item_name(puzzle: Puzzle) -> str:
    return _TYPE_TO_NAME[puzzle.item.item_type]


# ---------------------------------------------------------------------------
# PlanningRoom
# ---------------------------------------------------------------------------

class PlanningRoom(Room):
    """
    A room where puzzles must be solved in a DAG-constrained order.
    """

    _DIFFICULTY = {
        "nightmare": {"n_puzzles": 4, "n_scenery": 5,
                      "n_fake_items": 3, "n_fake_clues": 3,
                      "trap_item_desc": False,
                      "min_deadends": 2,
                      "min_deadend_depth": 2,
                      "min_trap_first_moves": 2,
                      "max_solutions": 1,
                      "n_initial": 3},
    }

    def __init__(
        self,
        puzzles:     List[Puzzle],
        dag:         RoomDAG,
        scenery:     List[SceneryItem],
        difficulty:  str = "nightmare",
        room_id:     str = "",
    ):
        self.dag = dag
        super().__init__(puzzles=[], scenery=scenery,
                         difficulty=difficulty, room_id=room_id)
        self.puzzles = puzzles
        self._exit_shown: bool = False
        self.item_states: Dict[str, bool] = dict(
            {name: False for name in ITEMS},
            **{name: True for name in dag.initial_state},
        )
        self._spec_map: Dict[str, PuzzleSpec] = {
            p.puzzle_id: spec
            for p in puzzles
            for spec in dag.puzzles
            if spec.puzzle_id == f"P_{_item_name(p).lower()}"
        }
        self._item_puzzle_map: Dict[str, Puzzle] = {
            _item_name(p): p for p in puzzles
        }

    @classmethod
    def create(
        cls,
        difficulty: str = "nightmare",
        room_id:    str = "",
        seed:       Optional[int] = None,
        clue_domain = None,
    ) -> "PlanningRoom":
        import itertools as _it
        import benchmark_core.room_core as _core
        _core._id_counter = _it.count(1)

        cfg = cls._DIFFICULTY[difficulty]
        item_types = [
            PuzzleItemType.CLOCK,
            PuzzleItemType.THERMOMETER,
            PuzzleItemType.COMPASS,
            PuzzleItemType.SCALE,
        ]
        puzzles = []
        for i, itype in enumerate(item_types):
            puz = Puzzle.create(
                item_type      = itype,
                level          = 1,
                n_fake_items   = cfg["n_fake_items"],
                n_fake_clues   = cfg["n_fake_clues"],
                trap_item_desc = cfg["trap_item_desc"],
                puzzle_id      = f"P{i+1}",
                clue_domain    = clue_domain,
            )
            puzzles.append(puz)

        dag = generate_room_dag(
            items                = ITEMS,
            min_deadends         = cfg.get("min_deadends", 1),
            min_deadend_depth    = cfg["min_deadend_depth"],
            min_trap_first_moves = cfg["min_trap_first_moves"],
            max_solutions        = cfg.get("max_solutions", 0),
            n_initial            = cfg.get("n_initial", 0),
            hard_locks           = cfg.get("hard_locks", False),
            seed                 = seed,
        )
        scenery = [SceneryItem.create() for _ in range(cfg["n_scenery"])]
        room = cls(
            puzzles    = puzzles,
            dag        = dag,
            scenery    = scenery,
            difficulty = difficulty,
            room_id    = room_id,
        )
        room.print_meta()
        return room

    @property
    def solved_ids(self) -> Set[str]:
        return {p.puzzle_id for p in self.puzzles if p.solved}

    @property
    def completed(self) -> bool:
        return len(self.solved_ids) == len(self.puzzles)

    @property
    def is_deadend(self) -> bool:
        if self.completed:
            return False
        return len(self.available_puzzles()) == 0

    @property
    def leads_to_deadend(self) -> bool:
        if self.completed:
            return False
        if self.is_deadend:
            return True
        current_unlocked = frozenset(k for k, v in self.item_states.items() if v)
        # Convert solved puzzle_ids (P1,P2...) to DAG spec ids (P_clock,P_thermometer...)
        dag_solved = frozenset(
            spec.puzzle_id for pid in self.solved_ids
            for spec in self.dag.puzzles
            if self._spec_map.get(pid) and self._spec_map[pid].puzzle_id == spec.puzzle_id
        )
        current_state: State = (current_unlocked, dag_solved)
        sol_paths, _ = explore(current_state, self.dag.puzzles)
        return len(sol_paths) == 0

    def available_puzzles(self) -> List[Puzzle]:
        return [
            p for p in self.puzzles
            if p.puzzle_id not in self.solved_ids
            and self.item_states.get(_item_name(p), False)
        ]

    def locked_puzzles(self) -> List[Puzzle]:
        return [
            p for p in self.puzzles
            if p.puzzle_id not in self.solved_ids
            and not self.item_states.get(_item_name(p), False)
        ]

    def get_puzzle(self, puzzle_id: str) -> Optional[Puzzle]:
        for p in self.puzzles:
            if p.puzzle_id == puzzle_id:
                return p
        return None

    def get_spec(self, puzzle_id: str) -> Optional[PuzzleSpec]:
        return self._spec_map.get(puzzle_id)

    def update_item(self, puzzle_id: str, answer) -> str:
        puzzle = self.get_puzzle(puzzle_id)
        if puzzle is None:
            return "unavailable"
        if puzzle.solved:
            return "unavailable"
        if not self.item_states.get(_item_name(puzzle), False):
            return "unavailable"

        result = puzzle.evaluate(answer)
        if result == "correct":
            self._apply_dag_effects(puzzle_id)
            self.prefix_desc = "You sense something changed in the room. "
            if self.completed:
                self._exit_shown = True
                self.prefix_desc = "All items suddenly disappear. A door swings open. "
        else:
            self._rotate_fakes(puzzle)
            self.prefix_desc = "You sense something changed in the room. "
        return result

    def _apply_dag_effects(self, puzzle_id: str) -> None:
        spec = self.get_spec(puzzle_id)
        if spec is None:
            return
        for item_name in spec.unlocks:
            self.item_states[item_name] = True
        for item_name in spec.locks:
            self.item_states[item_name] = False

    def _rotate_fakes(self, puzzle: Puzzle) -> None:
        if TrapType.FAKE_ITEM in puzzle.traps:
            old_trap = puzzle.traps[TrapType.FAKE_ITEM]
            new_trap = FakeItemTrap.create(puzzle.item, len(old_trap.fake_items))
            puzzle.traps[TrapType.FAKE_ITEM] = new_trap
            puzzle.all_items = new_trap.all_items
        if TrapType.FAKE_CLUE in puzzle.traps:
            old_trap = puzzle.traps[TrapType.FAKE_CLUE]
            new_trap = FakeClueTrap.create(puzzle.clue, len(old_trap.fake_clues))
            puzzle.traps[TrapType.FAKE_CLUE] = new_trap
            puzzle.all_clues = new_trap.all_clues

    def desc(self) -> str:
        lines = []
        if self.prefix_desc:
            lines.append(self.prefix_desc)
            self.prefix_desc = ""
        if self._exit_shown:
            lines.append(SceneryItem.EXIT_DESC)
            return "\n".join(lines)
        lines.append("You see the following in the room:\n")
        for s in self.scenery:
            lines.append(f"  [Item {s.item_id}] {s.desc}")
        lines.append("")
        for p in self.puzzles:
            item_name = _item_name(p)
            if p.puzzle_id in self.solved_ids:
                tag = " [SOLVED]"
            elif not self.item_states.get(item_name, False):
                tag = " [LOCKED — cannot interact]"
            else:
                tag = ""
            for obj in p.all_items:
                lines.append(f"  [Item {obj.item_id}]{tag} {obj}")
            for obj in p.all_clues:
                lines.append(f"  [Item {obj.item_id}]{tag} {obj}")
        return "\n".join(lines)

    def planning_desc(self) -> str:
        lines = ["=== PLANNING ROOM STATUS ===\n"]
        lines.append("ITEM STATES:")
        for name in ITEMS:
            state = "UNLOCKED ✓" if self.item_states.get(name, False) else "LOCKED   ✗"
            lines.append(f"  {name:<16}: {state}")
        lines.append("")
        avail = self.available_puzzles()
        lines.append(f"AVAILABLE PUZZLES ({len(avail)}) — you may solve these now:")
        if avail:
            for p in avail:
                spec = self.get_spec(p.puzzle_id)
                lines.append(f"\n  [{p.puzzle_id}]  item={_item_name(p)}  UNLOCKED ✓")
                if spec:
                    lines.append(f"    if solved → unlock {spec.unlocks}"
                                 + (f", lock {spec.locks}" if spec.locks else ""))
        else:
            lines.append("  ⚠ NONE — DEADEND.")
        lines.append("")
        locked = self.locked_puzzles()
        lines.append(f"LOCKED PUZZLES ({len(locked)}) — item is locked:")
        if locked:
            for p in locked:
                spec = self.get_spec(p.puzzle_id)
                lines.append(f"  [{p.puzzle_id}]  item={_item_name(p)}  LOCKED ✗")
                if spec:
                    lines.append(f"    if solved → unlock {spec.unlocks}"
                                 + (f", lock {spec.locks}" if spec.locks else ""))
        else:
            lines.append("  (none)")
        solved = [p for p in self.puzzles if p.puzzle_id in self.solved_ids]
        if solved:
            lines.append("")
            lines.append(f"SOLVED PUZZLES ({len(solved)}):")
            for p in solved:
                lines.append(f"  [{p.puzzle_id}]  item={_item_name(p)}  ✓")
        lines.append("")
        lines.append(f"Progress: {len(self.solved_ids)}/{len(self.puzzles)} puzzles solved")
        lines.append("Goal: solve ALL puzzles to escape.")
        lines.append("⚠ Warning: wrong ordering may permanently lock required items.")
        return "\n".join(lines)

    def print_meta(self) -> None:
        W = 64
        bar = "═" * W
        thin = "─" * W
        print(f"\n{bar}")
        print(f"  PLANNING ROOM  │  difficulty={self.difficulty}  "
              f"puzzles={len(self.puzzles)}  scenery={len(self.scenery)}")
        print(bar)
        print(f"  DAG  │  initial_unlocked={sorted(self.dag.initial_state)}")
        print(f"        │  solution_paths={len(self.dag.solution_paths)}  "
              f"deadend_paths={len(self.dag.deadend_paths)}")
        print(thin)
        for p in self.puzzles:
            item_name = _item_name(p)
            unlocked  = "✓" if self.item_states.get(item_name, False) else "✗"
            spec      = self.get_spec(p.puzzle_id)
            effect    = (f"unlock {spec.unlocks}, lock {spec.locks}" if spec else "?")
            cls       = self.dag.first_move_classes.get(
                f"P_{item_name.lower()}", "-")
            print(f"  [{unlocked}] {p.puzzle_id:<6} "
                  f"item={item_name:<14} "
                  f"effect: {effect:<40} "
                  f"first_move={cls}")
        print(thin)
        print("  Solution orderings:")
        for path in self.dag.solution_paths[:3]:
            print(f"    {' → '.join(path)}")
        if len(self.dag.solution_paths) > 3:
            print(f"    ... and {len(self.dag.solution_paths)-3} more")
        print("  Deadend orderings (deepest first):")
        deep = sorted(self.dag.deadend_paths, key=len, reverse=True)
        for path in deep[:3]:
            print(f"    {' → '.join(path)}  ✗")
        n_trap = sum(1 for v in self.dag.first_move_classes.values() if v == "trap")
        n_safe = sum(1 for v in self.dag.first_move_classes.values() if v == "safe")
        print(f"  First move: {n_trap} trap | {n_safe} safe  "
              f"(random success rate: {n_safe}/{n_trap+n_safe} = "
              f"{n_safe/(n_trap+n_safe)*100:.0f}%)")
        print(bar)

    def to_dict(self) -> dict:
        return {
            "room_id":      self.room_id,
            "difficulty":   self.difficulty,
            "type":         "planning",
            "item_states":  self.item_states,
            "solved_ids":   list(self.solved_ids),
            "puzzles":      [p.to_dict() for p in self.puzzles],
            "scenery":      [s.to_dict() for s in self.scenery],
            "dag": {
                "initial_state":      sorted(self.dag.initial_state),
                "solution_paths":     self.dag.solution_paths,
                "deadend_paths":      self.dag.deadend_paths,
                "first_move_classes": self.dag.first_move_classes,
                "puzzles": [
                    {"puzzle_id": sp.puzzle_id, "item_name": sp.item_name,
                     "unlocks": sp.unlocks, "locks": sp.locks}
                    for sp in self.dag.puzzles
                ],
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PlanningRoom":
        import itertools as _it
        import benchmark_core.room_core as _core
        max_id = 0
        for p in d["puzzles"]:
            max_id = max(max_id, p.get("item", {}).get("item_id", 0))
            for fi in p.get("fake_items", []):
                max_id = max(max_id, fi.get("item_id", 0))
            max_id = max(max_id, p.get("clue", {}).get("item_id", 0))
            for fc in p.get("fake_clues", []):
                max_id = max(max_id, fc.get("item_id", 0))
        for s in d["scenery"]:
            max_id = max(max_id, s.get("item_id", 0))
        _core._id_counter = _it.count(max_id + 1)

        puzzles = [Puzzle.from_dict(p) for p in d["puzzles"]]
        scenery = [SceneryItem.from_dict(s) for s in d["scenery"]]
        dag_d   = d["dag"]
        specs   = [
            PuzzleSpec(puzzle_id=sp["puzzle_id"], item_name=sp["item_name"],
                       unlocks=sp["unlocks"], locks=sp["locks"])
            for sp in dag_d["puzzles"]
        ]
        dag = RoomDAG(
            puzzles            = specs,
            initial_state      = frozenset(dag_d["initial_state"]),
            solution_paths     = dag_d["solution_paths"],
            deadend_paths      = dag_d["deadend_paths"],
            first_move_classes = dag_d.get("first_move_classes", {}),
        )
        room = cls(puzzles=puzzles, dag=dag, scenery=scenery,
                   difficulty=d.get("difficulty", "nightmare"),
                   room_id=d.get("room_id", ""))
        room.item_states = d["item_states"]
        room._exit_shown = room.completed
        return room


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    difficulty = sys.argv[1] if len(sys.argv) > 1 else "nightmare"
    seed       = int(sys.argv[2]) if len(sys.argv) > 2 else 42

    print(f"\nCreating PlanningRoom (difficulty={difficulty}, seed={seed})...")
    room = PlanningRoom.create(difficulty=difficulty, seed=seed)

    print("\n" + "─"*64)
    print("PLANNING DESC (what Planner sees at step 0):")
    print("─"*64)
    print(room.planning_desc())

    # Simulate correct solution
    sol = room.dag.solution_paths[0]
    print("\n" + "─"*64)
    print(f"Simulating correct path: {' → '.join(sol)}")
    print("─"*64)

    for pid in sol:
        # find the puzzle matching this DAG spec id
        spec = next(s for s in room.dag.puzzles if s.puzzle_id == pid)
        puzzle = room._item_puzzle_map.get(spec.item_name)
        if puzzle is None:
            print(f"  [{pid}] puzzle not found for item {spec.item_name}!")
            break

        avail = [p.puzzle_id for p in room.available_puzzles()]
        if puzzle.puzzle_id not in avail:
            print(f"  [{pid}] DEADEND — not available! "
                  f"Available puzzle_ids: {avail}")
            break

        gt = puzzle.item._groundtruth_answer(
            puzzle.clue.answer, puzzle.clue.delta_unit
        )
        result = room.update_item(puzzle.puzzle_id, gt)
        unlocked_now = [k for k, v in room.item_states.items() if v]
        print(f"  [{pid}] ({puzzle.puzzle_id}) → {result}  "
              f"item_states_unlocked={unlocked_now}")

    print(f"\nCompleted: {room.completed}  |  leads_to_deadend: {room.leads_to_deadend}")
