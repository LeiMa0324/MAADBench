"""
sage_generator.py — Phase 1 + Phase 2 pipeline.

Responsibilities:
  - Phase 1: build_k_steps (backward generation)
  - Phase 2: inject_simplifiable, inject_trap, enforce_nesting
  - Produce clean TaskInstance dicts ready for Phase 3 anomaly injection

Phase 3 (anomaly injection) is intentionally separate: see sage_anomaly.py.
"""

import random
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

from sage_core import (
    Expr, leaf, is_zero, is_one, is_numeric, is_leaf,
    all_nodes, set_node, full_simplify, count_forward_steps
)
from sage_difficulty import (
    DifficultyConfig, DifficultyLevel,
    inject_simplifiable, inject_trap, enforce_nesting,
)
from sage_anomaly import AnomalyInjector
from utils import *

# ---------------------------------------------------------------------------
# Backward rules
# ---------------------------------------------------------------------------

@dataclass
class BackwardRule:
    name: str
    forward_rule: str
    can_apply: callable
    apply: callable


def _get_factors(n: int) -> list:
    return [i for i in range(2, n) if n % i == 0]

def _is_composite(e: Expr) -> bool:
    try:
        n = int(float(e.value))
        return n >= 4 and len(_get_factors(n)) > 0
    except:
        return False


BACKWARD_RULES = [
    BackwardRule("insert_add_zero", "add_zero_r",
        can_apply=lambda e, vs: True,
        apply=lambda e, vs, rng: Expr("+", e.clone(), leaf("0"), None)),
    BackwardRule("insert_zero_add", "add_zero_l",
        can_apply=lambda e, vs: True,
        apply=lambda e, vs, rng: Expr("+", leaf("0"), e.clone(), None)),
    BackwardRule("insert_mul_one", "mul_one_r",
        can_apply=lambda e, vs: True,
        apply=lambda e, vs, rng: Expr("*", e.clone(), leaf("1"), None)),
    BackwardRule("insert_one_mul", "mul_one_l",
        can_apply=lambda e, vs: True,
        apply=lambda e, vs, rng: Expr("*", leaf("1"), e.clone(), None)),
    BackwardRule("expand_zero_sub", "sub_self",
        can_apply=lambda e, vs: is_zero(e) and len(vs) > 0,
        apply=lambda e, vs, rng: (
            lambda v: Expr("-", leaf(v), leaf(v), None))(rng.choice(vs))),
    BackwardRule("expand_one_div", "div_self",
        can_apply=lambda e, vs: is_one(e) and len(vs) > 0,
        apply=lambda e, vs, rng: (
            lambda v: Expr("/", leaf(v), leaf(v), None))(rng.choice(vs))),
    BackwardRule("split_numeric_add", "numeric_fold",
        can_apply=lambda e, vs: is_numeric(e) and 2 <= int(float(e.value)) <= 20,
        apply=lambda e, vs, rng: (
            lambda n, a: Expr("+", leaf(str(a)), leaf(str(n - a)), None)
        )(int(float(e.value)), rng.randint(1, int(float(e.value)) - 1))),
    BackwardRule("split_numeric_mul", "numeric_fold",
        can_apply=lambda e, vs: is_numeric(e) and _is_composite(e),
        apply=lambda e, vs, rng: (
            lambda fs, a: Expr("*", leaf(str(a)),
                               leaf(str(int(float(e.value)) // a)), None)
        )(_get_factors(int(float(e.value))),
          rng.choice(_get_factors(int(float(e.value)))))),
]


# ---------------------------------------------------------------------------
# Phase 1: backward generation
# ---------------------------------------------------------------------------

def _random_terminal(variables: list, rng: random.Random) -> Expr:
    c = rng.random()
    if c < 0.4 or not variables:
        return leaf(rng.choice(variables) if variables else "x")
    elif c < 0.7 and len(variables) >= 2:
        v1, v2 = rng.sample(variables, 2)
        return Expr(rng.choice(["+", "-", "*"]), leaf(v1), leaf(v2), None)
    else:
        return leaf(str(rng.choice([2, 3, 5, 7, 11, 13])))


def _try_build_k_steps(
    k: int, variables: list, rng: random.Random, rule_diversity: int
) -> Optional[tuple]:
    gt   = _random_terminal(variables, rng)
    expr = gt.clone()
    applied = []

    for step in range(k):
        candidates = [
            (path, node, br)
            for path, node in all_nodes(expr)
            for br in BACKWARD_RULES
            if br.can_apply(node, variables)
        ]
        if not candidates:
            return None
        if step >= 1:
            used = {b.forward_rule for b in applied}
            diverse = [(p, n, br) for p, n, br in candidates
                       if br.forward_rule not in used]
            if diverse and rng.random() < 0.7:
                candidates = diverse
        path, node, brule = rng.choice(candidates)
        expr = set_node(expr, path, brule.apply(node, variables, rng))
        applied.append(brule)

    if count_forward_steps(expr) != k:
        return None
    if len({b.forward_rule for b in applied}) < rule_diversity:
        return None
    return expr, gt, full_simplify(expr)


def build_k_steps(
    k: int, variables: list, rng: random.Random,
    rule_diversity: int = 1, max_retries: int = 50
) -> tuple:
    for _ in range(max_retries):
        result = _try_build_k_steps(k, variables, rng, rule_diversity)
        if result is not None:
            return result
    raise RuntimeError(
        f"Could not build expression with exactly {k} steps "
        f"after {max_retries} attempts."
    )


# ---------------------------------------------------------------------------
# SAGEDataGenerator — Phase 1 + Phase 2
# ---------------------------------------------------------------------------

class SAGEDataGenerator:
    """
    Generates clean SAGE task instances across difficulty levels.
    Each instance is a dict with:
      - expression, ground_truth, simplification_trace, k_steps, rules_used
      - trap_log: list of TrapRecord dicts (empty if trap_density=0)
      - anomaly: None  (reserved for Phase 3)
      - all DifficultyConfig fields
    """

    _LEVEL_MAP = {
        "easy":   DifficultyLevel.EASY,
        "medium": DifficultyLevel.MEDIUM,
        "hard":   DifficultyLevel.HARD,
    }

    def __init__(self, seed: int = 42):
        self.seed = seed

    def generate(
        self,
        easy: int = 0,
        medium: int = 0,
        hard: int = 0,
        config: Optional[DifficultyConfig] = None,
        n: Optional[int] = None,
        anomaly_modes: Optional[dict] = None,
        anomaly_levels: Optional[list] = None,
    ) -> list:
        """
        Generate tasks at specified difficulty levels.
        Pass config + n to use a custom DifficultyConfig instead of level presets.

        easy/medium/hard counts produce *clean* tasks only.
        anomaly_modes: e.g. {"FM-2.6": 5, "FM-3.1": 3} — adds 5+3 new anomaly tasks
                       (cloned from eligible clean tasks, originals stay clean)
        anomaly_levels: which difficulty levels to clone from (default: ["easy"])
        """
        if config is not None and n is not None:
            return self._generate_from_config(config, n, anomaly_modes, anomaly_levels)

        tasks = []
        for level_str, count in [("easy", easy), ("medium", medium), ("hard", hard)]:
            if count <= 0:
                continue
            cfg  = DifficultyConfig.from_level(self._LEVEL_MAP[level_str])
            pool = []
            for i in range(count):
                rng  = random.Random(self.seed * 10000 + i)
                task = self._generate_one(cfg, rng, pool, level_str)
                pool.append(task)
                tasks.append(task)
            print(f"Generated {count} {level_str} tasks.")

        if anomaly_modes:
            self._assign_anomalies(tasks, anomaly_modes, anomaly_levels)

        return tasks

    def _generate_from_config(self, cfg: DifficultyConfig, n: int,
                              anomaly_modes: Optional[dict] = None,
                              anomaly_levels: Optional[list] = None) -> list:
        tasks, pool = [], []
        level_str = "custom"
        for i in range(n):
            rng  = random.Random(self.seed * 10000 + i)
            task = self._generate_one(cfg, rng, pool, level_str)
            pool.append(task)
            tasks.append(task)
        print(f"Generated {n} custom tasks.")

        if anomaly_modes:
            self._assign_anomalies(tasks, anomaly_modes, anomaly_levels)

        return tasks

    def _generate_one(
        self,
        cfg: DifficultyConfig,
        rng: random.Random,
        pool: list,
        level_str: str,
    ) -> dict:

        # --- Phase 1: backward generation ---
        expr, gt, trace = build_k_steps(
            cfg.k_steps, cfg.variables, rng, cfg.rule_diversity
        )
        gt_str = str(gt)

        # --- Phase 2a: inject additional simplifiable patterns ---
        extra = rng.randint(cfg.min_inject,
                            cfg.min_inject + max(1, cfg.min_inject - 1))
        expr  = inject_simplifiable(expr, rng, count=extra)
        trace = full_simplify(expr)
        if str(expr) == trace[-1]["expr"]:
            expr  = inject_simplifiable(expr, rng, count=extra + 2)
            trace = full_simplify(expr)

        # --- Phase 2b: inject trap nodes ---
        trap_log = []
        if cfg.trap_density > 0:
            expr, trap_records = inject_trap(
                expr, cfg.variables, rng,
                trap_density=cfg.trap_density,
                ground_truth=gt_str,
            )
            # Re-simplify after trap injection (traps should not change GT)
            trace = full_simplify(expr)
            trap_log = [
                {
                    "trap_name": r.trap_name,
                    "mimics_rule": r.mimics_rule,
                    "why_invalid": r.why_invalid,
                    "path": r.path,
                }
                for r in trap_records
            ]

        # --- Phase 2c: enforce minimum nesting depth ---
        if cfg.min_depth > 0:
            expr  = enforce_nesting(expr, rng, min_depth=cfg.min_depth)
            trace = full_simplify(expr)

        # --- Context chain ---
        context = []
        if cfg.context_depth > 0 and pool:
            context = rng.sample(pool, min(cfg.context_depth, len(pool)))

        # --- Assemble task dict ---
        task = {
            **cfg.to_dict(),          # config defaults first (incl. target k_steps)
            "level": level_str,
            "expression": str(expr),
            "ground_truth": gt_str,
            "simplification_trace": trace,
            "k_steps": len(trace) - 1,   # actual steps after all Phase 2 injections
            "rules_used": [s["rule"] for s in trace[1:]],
            "context_chain": context,
            "trap_log": trap_log,
            "anomaly": None,
        }

        return task

    def _assign_anomalies(self, tasks, anomaly_modes, anomaly_levels):
        """Create new anomaly tasks by cloning eligible clean tasks. Originals stay clean."""
        if anomaly_levels is None:
            anomaly_levels = ["easy"]

        import copy
        anomaly_inj = AnomalyInjector(seed=self.seed)
        eligible = [t for t in tasks if t["level"] in anomaly_levels]
        rng = random.Random(self.seed + 999)
        rng.shuffle(eligible)

        idx = 0
        for fm_code, count in anomaly_modes.items():
            for _ in range(count):
                if idx >= len(eligible):
                    print(f"Warning: not enough eligible tasks for all anomalies")
                    break
                source = eligible[idx]
                new_task = copy.deepcopy(source)
                config = anomaly_inj.pick_random_config(new_task, anomaly_type=fm_code)
                new_task["anomaly"] = {
                    "anomaly_type": config.anomaly_type,
                    "target_agent": config.target_agent,
                    "injection_step": config.injection_step,
                }
                tasks.append(new_task)
                idx += 1

        # Shuffle so anomaly tasks are interleaved with clean tasks
        rng.shuffle(tasks)

        summary = ", ".join(f"{fm} x{c}" for fm, c in anomaly_modes.items())
        levels_str = ", ".join(anomaly_levels)
        print(f"Injected anomalies: {summary} (on {levels_str} tasks, {idx} added)")


# ---------------------------------------------------------------------------
# CLI / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    gen = SAGEDataGenerator(seed=42)
    tasks = gen.generate(easy=30, medium=30, hard=40, anomaly_modes={
        "FM-2.4": 5,"FM-2.5": 5,"FM-2.6": 5,"FM-3.1": 5,"FM-3.2": 5,"FM-3.3": 5})
    save_jsonl(tasks, "tasks.jsonl")

    # "FM-2.4": "transformer_withhold",
    # "FM-2.5": "ignored_input",
    # "FM-2.6": "rule_action_mismatch",
    # "FM-3.1": "premature_termination",
    # "FM-3.2": "checker_incomplete",
    # "FM-3.3": "checker_wrong_verdict",