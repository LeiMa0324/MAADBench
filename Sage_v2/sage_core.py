"""
sage_core.py — Expression tree, traversal helpers, and forward simplification engine.
"""

from dataclasses import dataclass
from typing import Optional
from copy import deepcopy
from sage_rules import FORWARD_RULES

UNARY_OPS = {"log", "sqrt", "sin", "cos"}

_PREC = {"+": 1, "-": 1, "*": 2, "/": 2}


def _needs_parens(child: "Expr", parent_op: str, is_right: bool) -> bool:
    """Return True if child expression needs parentheses inside parent_op."""
    if child.op is None or child.op in UNARY_OPS:
        return False
    child_prec  = _PREC.get(child.op,  1)
    parent_prec = _PREC.get(parent_op, 1)
    if child_prec < parent_prec:
        return True
    # Right operand of a non-commutative op at the same precedence level:
    # e.g.  a - (b + c)  or  a / (b * c)
    if child_prec == parent_prec and is_right and parent_op in ("-", "/"):
        return True
    return False


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
        left_s  = f"({self.left})"  if _needs_parens(self.left,  self.op, False) else str(self.left)
        right_s = f"({self.right})" if _needs_parens(self.right, self.op, True)  else str(self.right)
        return f"{left_s} {self.op} {right_s}"

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
# Forward simplification engine
# ---------------------------------------------------------------------------

def apply_forward_once(expr: Expr) -> tuple:
    """Apply first matching forward rule (BFS order).
    Returns (new_expr, rule_name, path) or (expr, None, None)."""
    for path, node in all_nodes(expr):
        for rule in FORWARD_RULES:
            if rule.match(node):
                simplified = rule.apply(node)
                new_expr = set_node(expr, path, simplified)
                return new_expr, rule.name, path
    return expr, None, None

def full_simplify(expr: Expr) -> list:
    """Returns list of {expr, rule} dicts representing the full trace."""
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
