"""
sage_benchmark.py — SAGE Benchmark Task Generator
Imports all simplification rules from sage_rules.py.
"""

import random
from dataclasses import dataclass, field
from typing import Optional
from copy import deepcopy
from enum import Enum
import pandas as pd


"""
sage_rules.py — Shared simplification rule definitions for SAGE benchmark.
Imported by both the task generator and the MAS agent prompt builder.
"""

from dataclasses import dataclass
from typing import Optional, Callable



# ---------------------------------------------------------------------------
# Minimal Expr dependency (forward declaration for type hints)
# ---------------------------------------------------------------------------
# sage_rules.py is intentionally self-contained so it can be imported
# without pulling in the full benchmark module.

def _is_zero(e) -> bool:
    return e.op is None and e.value == "0"

def _is_one(e) -> bool:
    return e.op is None and e.value == "1"

def _is_numeric(e) -> bool:
    if e.op is not None:
        return False
    try:
        float(e.value)
        return True
    except ValueError:
        return False

def _leaf(val: str):
    # Deferred import to avoid circular dependency

    return leaf(val)

def _numeric_fold(e):

    a, b = float(e.left.value), float(e.right.value)
    if e.op == "/" and b == 0:
        return e
    result = {"+": a + b, "-": a - b, "*": a * b, "/": a / b}[e.op]
    if result == int(result):
        return leaf(str(int(result)))
    return leaf(f"{result:.4f}")


# ---------------------------------------------------------------------------
# Rule dataclass
# ---------------------------------------------------------------------------

@dataclass
class Rule:
    id: str           # e.g. "r1"
    name: str         # e.g. "add_zero_r"
    pattern: str      # human-readable: "x + 0 → x"
    condition: str    # human-readable condition, "---" if unconditional
    match: Callable   # (Expr) -> bool
    apply: Callable   # (Expr) -> Expr


# ---------------------------------------------------------------------------
# Forward simplification rules
# ---------------------------------------------------------------------------

FORWARD_RULES = [
    Rule(
        id="r1", name="add_zero_r",
        pattern="x + 0  →  x", condition="---",
        match=lambda e: e.op == "+" and _is_zero(e.right),
        apply=lambda e: e.left,
    ),
    Rule(
        id="r2", name="add_zero_l",
        pattern="0 + x  →  x", condition="---",
        match=lambda e: e.op == "+" and _is_zero(e.left),
        apply=lambda e: e.right,
    ),
    Rule(
        id="r3", name="mul_one_r",
        pattern="x * 1  →  x", condition="---",
        match=lambda e: e.op == "*" and _is_one(e.right),
        apply=lambda e: e.left,
    ),
    Rule(
        id="r4", name="mul_one_l",
        pattern="1 * x  →  x", condition="---",
        match=lambda e: e.op == "*" and _is_one(e.left),
        apply=lambda e: e.right,
    ),
    Rule(
        id="r5", name="mul_zero_r",
        pattern="x * 0  →  0", condition="---",
        match=lambda e: e.op == "*" and _is_zero(e.right),
        apply=lambda e: _leaf("0"),
    ),
    Rule(
        id="r6", name="mul_zero_l",
        pattern="0 * x  →  0", condition="---",
        match=lambda e: e.op == "*" and _is_zero(e.left),
        apply=lambda e: _leaf("0"),
    ),
    Rule(
        id="r7", name="sub_self",
        pattern="x - x  →  0", condition="---",
        match=lambda e: e.op == "-" and str(e.left) == str(e.right),
        apply=lambda e: _leaf("0"),
    ),
    Rule(
        id="r8", name="div_self",
        pattern="x / x  →  1", condition="x ≠ 0",
        match=lambda e: (
            e.op == "/" and
            str(e.left) == str(e.right) and
            not _is_zero(e.left)
        ),
        apply=lambda e: _leaf("1"),
    ),
    Rule(
        id="r9", name="numeric_fold",
        pattern="a ⊙ b  →  c", condition="a, b ∈ ℤ;  ⊙ ∈ {+,−,×,÷};  b ≠ 0 if ⊙ = ÷",
        match=lambda e: (
            e.op in ("+", "-", "*", "/") and
            _is_numeric(e.left) and _is_numeric(e.right) and
            not (e.op == "/" and float(e.right.value) == 0)
        ),
        apply=_numeric_fold,
    ),
]

# Fast lookup by name and by id
RULE_BY_NAME = {r.name: r for r in FORWARD_RULES}
RULE_BY_ID   = {r.id:   r for r in FORWARD_RULES}


# ---------------------------------------------------------------------------
# Prompt-facing rule reference (used by agent prompts)
# ---------------------------------------------------------------------------

def rules_as_prompt_text() -> str:
    """Return a formatted rule reference block for injection into agent prompts."""
    lines = ["Available simplification rules:"]
    for r in FORWARD_RULES:
        cond = f"   (condition: {r.condition})" if r.condition != "---" else ""
        lines.append(f"  {r.id}: {r.pattern}{cond}")
    return "\n".join(lines)



# ---------------------------------------------------------------------------
# Expression Tree
# ---------------------------------------------------------------------------

@dataclass
class Expr:
    op: Optional[str]
    left: Optional["Expr"]
    right: Optional["Expr"]
    value: Optional[str]

    def __str__(self) -> str:
        if self.op is None:
            return self.value
        if self.op in UNARY_OPS:
            return f"{self.op}({self.left})"
        return f"({self.left} {self.op} {self.right})"

    def depth(self) -> int:
        if self.op is None:
            return 0
        if self.op in UNARY_OPS:
            return 1 + self.left.depth()
        return 1 + max(self.left.depth(), self.right.depth())

    def variables(self) -> set:
        if self.op is None:
            try:
                float(self.value)
                return set()
            except ValueError:
                return {self.value}
        if self.op in UNARY_OPS:
            return self.left.variables()
        return self.left.variables() | self.right.variables()

    def clone(self) -> "Expr":
        return deepcopy(self)


UNARY_OPS = {"log", "sqrt", "sin", "cos"}


def leaf(val: str) -> Expr:
    return Expr(None, None, None, val)
def is_zero(e: Expr) -> bool:
    return e.op is None and e.value == "0"
def is_one(e: Expr) -> bool:
    return e.op is None and e.value == "1"
def is_numeric(e: Expr) -> bool:
    if e.op is not None:
        return False
    try:
        float(e.value)
        return True
    except ValueError:
        return False
def is_leaf(e: Expr) -> bool:
    return e.op is None

# ---------------------------------------------------------------------------
# Tree traversal helpers
# ---------------------------------------------------------------------------

def all_nodes(expr: Expr) -> list:
    result = []
    def _walk(node, path):
        result.append((path, node))
        if node.op is None:
            return
        _walk(node.left, path + ("left",))
        if node.op not in UNARY_OPS:
            _walk(node.right, path + ("right",))
    _walk(expr, ())
    return result

def get_node(expr: Expr, path: tuple) -> Expr:
    node = expr
    for direction in path:
        node = node.left if direction == "left" else node.right
    return node

def set_node(expr: Expr, path: tuple, new_node: Expr) -> Expr:
    expr = expr.clone()
    if not path:
        return new_node
    parent = get_node(expr, path[:-1])
    if path[-1] == "left":
        parent.left = new_node
    else:
        parent.right = new_node
    return expr

# ---------------------------------------------------------------------------
# Forward simplification engine (uses rules from sage_rules)
# ---------------------------------------------------------------------------
def apply_forward_once(expr: Expr) -> tuple:
    for path, node in all_nodes(expr):
        for rule in FORWARD_RULES:
            if rule.match(node):
                simplified = rule.apply(node)
                new_expr = set_node(expr, path, simplified)
                return new_expr, rule.name, path
    return expr, None, None

def full_simplify(expr: Expr) -> list:
    trace = [{"expr": str(expr), "rule": None}]
    current = expr
    for _ in range(200):
        new_expr, rule_name, _ = apply_forward_once(current)
        if rule_name is None:
            break
        current = new_expr
        trace.append({"expr": str(current), "rule": rule_name})
    return trace

def count_forward_steps(expr: Expr) -> int:
    return len(full_simplify(expr)) - 1

# ---------------------------------------------------------------------------
# Backward rules
# ---------------------------------------------------------------------------

@dataclass
class BackwardRule:
    name: str
    forward_rule: str
    can_apply: callable
    apply: callable

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
            lambda fs, a: Expr("*", leaf(str(a)), leaf(str(int(float(e.value))//a)), None)
        )(_get_factors(int(float(e.value))), rng.choice(_get_factors(int(float(e.value)))))),
]


def _is_composite(e):
    try:
        n = int(float(e.value))
        return n >= 4 and len(_get_factors(n)) > 0
    except:
        return False

def _get_factors(n):
    return [i for i in range(2, n) if n % i == 0]


# ---------------------------------------------------------------------------
# Backward generation
# ---------------------------------------------------------------------------

def build_k_steps(k, variables, rng, rule_diversity=1, max_retries=50):
    for _ in range(max_retries):
        result = _try_build_k_steps(k, variables, rng, rule_diversity)
        if result is not None:
            return result
    raise RuntimeError(f"Could not build expression with exactly {k} steps "
                       f"after {max_retries} attempts.")


def _try_build_k_steps(k, variables, rng, rule_diversity):
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


def _random_terminal(variables, rng):
    c = rng.random()
    if c < 0.4 or not variables:
        return leaf(rng.choice(variables) if variables else "x")
    elif c < 0.7 and len(variables) >= 2:
        v1, v2 = rng.sample(variables, 2)
        return Expr(rng.choice(["+", "-", "*"]), leaf(v1), leaf(v2), None)
    else:
        return leaf(str(rng.choice([2, 3, 5, 7, 11, 13])))

# ---------------------------------------------------------------------------
# Inject simplifiable patterns
# ---------------------------------------------------------------------------

INFLATORS = [
    lambda e, rng: Expr("+", e, leaf("0"), None),
    lambda e, rng: Expr("+", leaf("0"), e, None),
    lambda e, rng: Expr("*", e, leaf("1"), None),
    lambda e, rng: Expr("*", leaf("1"), e, None),
]


def inject_simplifiable(expr, rng, count=1):
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
                                    Expr("+", leaf(str(a)), leaf(str(n-a)), None))
                elif factors:
                    a = rng.choice(factors)
                    expr = set_node(expr, path,
                                    Expr("*", leaf(str(a)), leaf(str(n//a)), None))
                continue
        expr = set_node(expr, path,
                        rng.choice(INFLATORS)(chosen.clone(), rng))
    return expr

# ---------------------------------------------------------------------------
# Difficulty Config
# ---------------------------------------------------------------------------

class DifficultyLevel(Enum):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"


@dataclass
class DifficultyConfig:
    k_steps: int        = 3
    variables: list     = field(default_factory=lambda: ["x", "y", "z"])
    rule_diversity: int = 1
    context_depth: int  = 0
    min_inject: int     = 1

    @classmethod
    def from_level(cls, level: DifficultyLevel) -> "DifficultyConfig":
        return {
            DifficultyLevel.EASY:   cls( k_steps=2,  variables=["x","y"],
                                        rule_diversity=1, context_depth=0, min_inject=1),
            DifficultyLevel.MEDIUM: cls(k_steps=5,  variables=["x","y","z"],
                                        rule_diversity=2, context_depth=0, min_inject=2),
            DifficultyLevel.HARD:   cls( k_steps=9,  variables=["x","y","z","w"],
                                        rule_diversity=3, context_depth=2, min_inject=3),
        }[level]

    def to_dict(self):
        return {
            "k_steps": self.k_steps,
            "variables": self.variables,
            "rule_diversity": self.rule_diversity,
            "context_depth": self.context_depth,
            "min_inject": self.min_inject,
        }

# ---------------------------------------------------------------------------
# Task Instance
# ---------------------------------------------------------------------------


class SAGEDataGenerator:
    """
    Generates SAGEItems at multiple difficulty levels.

    Args:
        seed: Random seed for reproducibility.
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
    ) :
        tasks = []
        for level_str, n in [("easy", easy), ("medium", medium), ("hard", hard)]:
            if n <= 0:
                continue

            cfg = DifficultyConfig.from_level(self._LEVEL_MAP[level_str])
            pool = []

            for i in range(n):
                rng = random.Random(self.seed * 10000 + i)
                task = self._generate_one(cfg, rng, pool,level_str)
                pool.append(task)
                tasks.append(task)

            print(f"Generated {n} {level_str} expressions.")

        return tasks

    def _generate_one(self, cfg: DifficultyConfig, rng, pool: list,level_str:str):
        expr, gt, trace = build_k_steps(cfg.k_steps, cfg.variables, rng, cfg.rule_diversity)
        extra = rng.randint(cfg.min_inject, cfg.min_inject + max(1, cfg.min_inject - 1))
        expr  = inject_simplifiable(expr, rng, count=extra)
        trace = full_simplify(expr)
        if str(expr) == trace[-1]["expr"]:
            expr  = inject_simplifiable(expr, rng, count=extra + 2)
            trace = full_simplify(expr)

        context = []
        if cfg.context_depth > 0 and pool:
            context = rng.sample(pool, min(cfg.context_depth, len(pool)))

        config_dict = cfg.to_dict()
        task = {
            "level": level_str,
            "expression": str(expr),
            "ground_truth": str(gt),
            "simplification_trace": trace,
            "k_steps": len(trace) - 1,
            "rules_used": [s['rule'] for s in trace[1:]],
            "context_chain":context,
        }
        for k, v in config_dict.items():
            task[k] = v

        return task

# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    generator = SAGEDataGenerator(seed=42)
    tasks = generator.generate(easy=30, medium=30, hard=40)
    task_df = pd.DataFrame.from_dict(tasks)
    task_df["e_id"] = task_df.index
    task_df.to_csv("tasks.csv")

