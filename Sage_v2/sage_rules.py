"""
sage_rules.py — Shared simplification rule definitions for SAGE benchmark.
Imported by both the task generator and the MAS agent prompt builder.
"""

from dataclasses import dataclass
from typing import Callable


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
    from sage_core import leaf
    return leaf(val)

def _numeric_fold(e):
    from sage_core import leaf
    a, b = float(e.left.value), float(e.right.value)
    if e.op == "/" and b == 0:
        return e
    result = {"+": a + b, "-": a - b, "*": a * b, "/": a / b}[e.op]
    if result == int(result):
        return leaf(str(int(result)))
    return leaf(f"{result:.4f}")


@dataclass
class Rule:
    id: str
    name: str
    pattern: str
    condition: str
    match: Callable
    apply: Callable


FORWARD_RULES = [
    Rule(id="r1", name="add_zero_r",
         pattern="x + 0  →  x", condition="---",
         match=lambda e: e.op == "+" and _is_zero(e.right),
         apply=lambda e: e.left),
    Rule(id="r2", name="add_zero_l",
         pattern="0 + x  →  x", condition="---",
         match=lambda e: e.op == "+" and _is_zero(e.left),
         apply=lambda e: e.right),
    Rule(id="r3", name="mul_one_r",
         pattern="x * 1  →  x", condition="---",
         match=lambda e: e.op == "*" and _is_one(e.right),
         apply=lambda e: e.left),
    Rule(id="r4", name="mul_one_l",
         pattern="1 * x  →  x", condition="---",
         match=lambda e: e.op == "*" and _is_one(e.left),
         apply=lambda e: e.right),
    Rule(id="r5", name="mul_zero_r",
         pattern="x * 0  →  0", condition="---",
         match=lambda e: e.op == "*" and _is_zero(e.right),
         apply=lambda e: _leaf("0")),
    Rule(id="r6", name="mul_zero_l",
         pattern="0 * x  →  0", condition="---",
         match=lambda e: e.op == "*" and _is_zero(e.left),
         apply=lambda e: _leaf("0")),
    Rule(id="r7", name="sub_self",
         pattern="x - x  →  0", condition="---",
         match=lambda e: e.op == "-" and str(e.left) == str(e.right),
         apply=lambda e: _leaf("0")),
    Rule(id="r8", name="div_self",
         pattern="x / x  →  1", condition="x ≠ 0",
         match=lambda e: (
             e.op == "/" and
             str(e.left) == str(e.right) and
             not _is_zero(e.left)
         ),
         apply=lambda e: _leaf("1")),
    Rule(id="r9", name="numeric_fold",
         pattern="a ⊙ b  →  c",
         condition="a, b ∈ ℤ;  ⊙ ∈ {+,−,×,÷};  b ≠ 0 if ⊙ = ÷",
         match=lambda e: (
             e.op in ("+", "-", "*", "/") and
             _is_numeric(e.left) and _is_numeric(e.right) and
             not (e.op == "/" and float(e.right.value) == 0)
         ),
         apply=_numeric_fold),
]

RULE_BY_NAME = {r.name: r for r in FORWARD_RULES}
RULE_BY_ID   = {r.id:   r for r in FORWARD_RULES}


def rules_as_prompt_text() -> str:
    lines = ["Available simplification rules:"]
    for r in FORWARD_RULES:
        cond = f"   (condition: {r.condition})" if r.condition != "---" else ""
        lines.append(f"  {r.id}: {r.pattern}{cond}")
    return "\n".join(lines)
