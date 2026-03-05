"""
sage_difficulty.py — Phase 2: Difficulty injection.

All injectors in this module modify the expression BEFORE evaluation.
They are part of the task instance definition and are applied during generation.
The resulting task is still clean (no agent-level anomalies).

Injectors:
  - inject_simplifiable : add more legitimate reducible patterns (existing)
  - inject_trap         : add misleading but invalid patterns
  - enforce_nesting     : force deep nesting structure
"""

import random
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from sage_core import (
    Expr, leaf, is_zero, is_one, is_numeric, is_leaf,
    all_nodes, set_node, full_simplify, count_forward_steps
)


# ---------------------------------------------------------------------------
# Phase 2a: inject_simplifiable (existing logic, cleaned up)
# ---------------------------------------------------------------------------

INFLATORS = [
    lambda e, rng: Expr("+", e, leaf("0"), None),
    lambda e, rng: Expr("+", leaf("0"), e, None),
    lambda e, rng: Expr("*", e, leaf("1"), None),
    lambda e, rng: Expr("*", leaf("1"), e, None),
]

def _get_factors(n: int) -> list:
    return [i for i in range(2, n) if n % i == 0]

def inject_simplifiable(expr: Expr, rng: random.Random, count: int = 1) -> Expr:
    """Wrap random leaf nodes with identity-preserving simplifiable patterns."""
    expr = expr.clone()
    for _ in range(count):
        leaves = [(p, n) for p, n in all_nodes(expr) if is_leaf(n)]
        if not leaves:
            break
        path, chosen = rng.choice(leaves)
        if is_numeric(chosen) and rng.random() < 0.3:
            n = int(float(chosen.value))
            if 2 <= n <= 20:
                factors = _get_factors(n)
                if rng.random() < 0.5:
                    a = rng.randint(1, n - 1)
                    expr = set_node(expr, path,
                                    Expr("+", leaf(str(a)), leaf(str(n - a)), None))
                elif factors:
                    a = rng.choice(factors)
                    expr = set_node(expr, path,
                                    Expr("*", leaf(str(a)), leaf(str(n // a)), None))
                continue
        expr = set_node(expr, path, rng.choice(INFLATORS)(chosen.clone(), rng))
    return expr


# ---------------------------------------------------------------------------
# Phase 2b: inject_trap
# ---------------------------------------------------------------------------

@dataclass
class TrapTemplate:
    name: str
    mimics_rule: str       # which rule it looks like
    why_invalid: str       # why the rule cannot actually apply
    can_generate: callable # (variables) -> bool
    generate: callable     # (variables, rng) -> Expr


TRAP_TEMPLATES = [

    # x / y  looks like div_self but x != y
    TrapTemplate(
        name="false_div_self_vars",
        mimics_rule="div_self",
        why_invalid="left operand != right operand",
        can_generate=lambda vs: len(vs) >= 2,
        generate=lambda vs, rng: (
            lambda pair: Expr("/", leaf(pair[0]), leaf(pair[1]), None)
        )(rng.sample([v for v in vs], 2)),
    ),

    # x / 0  looks like div_self but denominator is zero
    TrapTemplate(
        name="false_div_zero",
        mimics_rule="div_self",
        why_invalid="division by zero",
        can_generate=lambda vs: len(vs) >= 1,
        generate=lambda vs, rng: Expr("/", leaf(rng.choice(vs)), leaf("0"), None),
    ),

    # x - y  looks like sub_self but x != y
    TrapTemplate(
        name="false_sub_self",
        mimics_rule="sub_self",
        why_invalid="left operand != right operand",
        can_generate=lambda vs: len(vs) >= 2,
        generate=lambda vs, rng: (
            lambda pair: Expr("-", leaf(pair[0]), leaf(pair[1]), None)
        )(rng.sample([v for v in vs], 2)),
    ),

    # x * (y - y)  looks like mul_zero but zero is hidden inside sub_self
    TrapTemplate(
        name="hidden_zero_mul",
        mimics_rule="mul_zero_r",
        why_invalid="zero not yet reduced; requires sub_self first",
        can_generate=lambda vs: len(vs) >= 1,
        generate=lambda vs, rng: (
            lambda v, x: Expr("*", leaf(x), Expr("-", leaf(v), leaf(v), None), None)
        )(rng.choice(vs), rng.choice(vs)),
    ),

    # x + 3  looks like numeric_fold but contains a variable
    TrapTemplate(
        name="false_numeric_fold",
        mimics_rule="numeric_fold",
        why_invalid="contains variable; not purely numeric",
        can_generate=lambda vs: len(vs) >= 1,
        generate=lambda vs, rng: Expr(
            "+", leaf(rng.choice(vs)), leaf(str(rng.randint(2, 9))), None
        ),
    ),

    # (x + 1) / (x + 2)  structurally similar to div_self but unequal
    TrapTemplate(
        name="false_div_self_complex",
        mimics_rule="div_self",
        why_invalid="sub-expressions are structurally similar but not equal",
        can_generate=lambda vs: len(vs) >= 1,
        generate=lambda vs, rng: (
            lambda v: Expr(
                "/",
                Expr("+", leaf(v), leaf(str(rng.randint(1, 4))), None),
                Expr("+", leaf(v), leaf(str(rng.randint(5, 9))), None),
                None,
            )
        )(rng.choice(vs)),
    ),
]

TRAP_BY_NAME = {t.name: t for t in TRAP_TEMPLATES}


@dataclass
class TrapRecord:
    path: tuple
    trap_name: str
    mimics_rule: str
    why_invalid: str


def inject_trap(
    expr: Expr,
    variables: list,
    rng: random.Random,
    trap_density: int = 1,
    ground_truth: Optional[str] = None,
) -> tuple:
    """
    Inject trap_density trap nodes into the expression.

    Traps are placed at leaf positions so they do not alter the
    existing simplification structure. After injection, the forward
    simplification trace is re-verified to ensure ground_truth is unchanged.

    Returns:
        (modified_expr, list[TrapRecord])
    """
    expr = expr.clone()
    records: list[TrapRecord] = []

    viable_templates = [t for t in TRAP_TEMPLATES if t.can_generate(variables)]
    if not viable_templates:
        return expr, records

    for _ in range(trap_density):
        # Only replace leaf nodes to avoid disrupting existing structure
        leaf_positions = [(p, n) for p, n in all_nodes(expr) if is_leaf(n)]
        if not leaf_positions:
            break

        template = rng.choice(viable_templates)
        trap_node = template.generate(variables, rng)

        # Try each leaf position; accept the first that preserves ground truth
        rng.shuffle(leaf_positions)
        placed = False
        for path, _ in leaf_positions:
            candidate = set_node(expr, path, trap_node)
            # Ground truth must not change after trap injection
            if ground_truth is not None:
                new_trace = full_simplify(candidate)
                if new_trace[-1]["expr"] != ground_truth:
                    continue
            expr = candidate
            records.append(TrapRecord(
                path=path,
                trap_name=template.name,
                mimics_rule=template.mimics_rule,
                why_invalid=template.why_invalid,
            ))
            placed = True
            break

        if not placed:
            # Could not place this trap without corrupting ground truth; skip
            continue

    return expr, records


# ---------------------------------------------------------------------------
# Phase 2c: enforce_nesting
# ---------------------------------------------------------------------------

def enforce_nesting(
    expr: Expr,
    rng: random.Random,
    min_depth: int = 3,
) -> Expr:
    """
    If the expression tree depth is below min_depth, wrap the root in
    identity-preserving layers until the target depth is reached.
    Does not change the simplification trace length.
    """
    while expr.depth() < min_depth:
        wrapper = rng.choice(INFLATORS)
        expr = wrapper(expr.clone(), rng)
    return expr


# ---------------------------------------------------------------------------
# DifficultyConfig — single source of truth for all Phase 2 parameters
# ---------------------------------------------------------------------------

class DifficultyLevel(Enum):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"


@dataclass
class DifficultyConfig:
    # Core axes (existing)
    k_steps: int        = 3
    variables: list     = field(default_factory=lambda: ["x", "y", "z"])
    rule_diversity: int = 1
    context_depth: int  = 0
    min_inject: int     = 1

    # New difficulty axes (Phase 2)
    trap_density: int   = 0    # number of trap nodes to inject
    min_depth: int      = 0    # minimum tree depth (0 = no enforcement)

    @classmethod
    def from_level(cls, level: DifficultyLevel) -> "DifficultyConfig":
        return {
            DifficultyLevel.EASY: cls(
                k_steps=2, variables=["x", "y"],
                rule_diversity=1, context_depth=0, min_inject=1,
                trap_density=0, min_depth=0,
            ),
            DifficultyLevel.MEDIUM: cls(
                k_steps=5, variables=["x", "y", "z"],
                rule_diversity=2, context_depth=0, min_inject=2,
                trap_density=1, min_depth=3,
            ),
            DifficultyLevel.HARD: cls(
                k_steps=9, variables=["x", "y", "z", "w"],
                rule_diversity=3, context_depth=2, min_inject=3,
                trap_density=2, min_depth=5,
            ),
        }[level]

    def to_dict(self) -> dict:
        return {
            "k_steps": self.k_steps,
            "variables": self.variables,
            "rule_diversity": self.rule_diversity,
            "context_depth": self.context_depth,
            "min_inject": self.min_inject,
            "trap_density": self.trap_density,
            "min_depth": self.min_depth,
        }
