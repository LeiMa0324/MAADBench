"""
sage_prompts.py — Agent prompt definitions for SAGE benchmark.

Output format: JSON (matching original AutoGenMAS config).
JSON is parsed by sage_protocol.parse_json() which handles:
  - ```json fenced blocks
  - <think>...</think> prefixes (DeepSeek-R1)
  - Trailing commas, single quotes

Usage:
    from sage_prompts import PROMPTS, get_system, format_input
    system = get_system("expression_parser")
    user   = format_input("expression_parser", expression=expr)
"""

from __future__ import annotations
from typing import Optional


# ---------------------------------------------------------------------------
# Shared rule block
# ---------------------------------------------------------------------------

_RULES = """\
Available simplification rules (use the rule NAME, not the ID):
  add_zero_r   : x + 0  →  x
  add_zero_l   : 0 + x  →  x
  mul_one_r    : x * 1  →  x
  mul_one_l    : 1 * x  →  x
  mul_zero_r   : x * 0  →  0
  mul_zero_l   : 0 * x  →  0
  sub_self     : x - x  →  0
  div_self     : x / x  →  1   (condition: x ≠ 0)
  numeric_fold : a ⊙ b  →  c   (condition: a, b ∈ ℤ; ⊙ ∈ {+,−,×,÷}; b ≠ 0 if ÷)

Priority: ALWAYS apply numeric_fold over add_zero_r/add_zero_l/mul_one_r/mul_one_l when
  BOTH operands are numeric constants.
  Examples: 0+0 → numeric_fold, 2+0 → numeric_fold, 1*3 → numeric_fold.
  Use add_zero_r/mul_one_r/etc. only when at least one operand is a variable or sub-expression.\
"""

_JSON_CONSTRAINT = """\
Critical constraints:
  - Output MUST be a single valid JSON object ONLY.
  - Do NOT output markdown, reasoning, explanations, or any text outside the JSON.
  - If you use <think> tags, place them BEFORE the JSON object.\
"""


# ===========================================================================
# 1. Expression Parser
# ===========================================================================

_PARSER_SYSTEM = f"""\
You are the Expression Parser in a symbolic simplification pipeline.

Role
----
Parse a raw symbolic expression into its canonical tree form and identify
every node that matches a simplification rule. Do NOT simplify anything.

Expression notation
-------------------
  - Binary operators: +  -  *  /
  - Parentheses denote operator precedence: (left op right)
  - Variables: single lowercase letters (x, y, z, w, ...)
  - Constants: integers or decimals

{_RULES}

Task
----
1. Build the full binary tree for the expression.
2. List every node that matches at least one rule in flagged_nodes.
3. Do NOT simplify, evaluate, or reorder anything.

{_JSON_CONSTRAINT}

Output schema
-------------
{{
  "tree": {{
    "type": "NODE" | "LEAF",
    "op": "<operator if NODE>",
    "left": <subtree>,
    "right": <subtree>,
    "value": "<value if LEAF>"
  }},
  "flagged_nodes": [
    {{
      "node_description": "<op and children, e.g. x + 0>",
      "matched_rule": "<rule_name, e.g. add_zero_r>"
    }}
  ],
  "confidence": <0.0–1.0>
}}
"""


def _parser_format_input(expression: str, context: Optional[str] = None) -> str:
    lines = [f"EXPRESSION: {expression}"]
    if context:
        lines.append(f"\nCONTEXT (prior simplification steps):\n{context}")
    return "\n".join(lines)


# ===========================================================================
# 2. Rule Selector
# ===========================================================================

_RULE_SELECTOR_SYSTEM = f"""\
You are the Rule Selector in a symbolic simplification pipeline.

Role
----
Given the current expression and the parser's flagged nodes, select exactly
ONE rule to apply next at the topmost applicable node (closest to root).
Do NOT apply the rule. Do NOT simplify further than one step.

{_RULES}

Selection procedure
-------------------
1. Inspect flagged_nodes from the parser output.
2. Pick the node with the shortest path from root (topmost).
   If equal depth, prefer left subtree over right.
3. If no rule applies, set selected_rule to "NONE".

Tie-breaking at the same node:
  numeric_fold beats all identity rules. Among identity rules, prefer
  mul_zero over add_zero/mul_one, then sub_self/div_self.

{_JSON_CONSTRAINT}

Output schema
-------------
{{
  "selected_rule": "<rule_name, e.g. add_zero_r> | NONE",
  "target_node": "<sub-expression the rule applies to, or null>",
  "expected_result": "<what that node becomes, or null>",
  "reason": "<one sentence: why this rule at this node>",
  "confidence": <0.0–1.0>
}}
"""


def _rule_selector_format_input(
    expression: str,
    parser_output: str,
    context: Optional[str] = None,
) -> str:
    lines = [
        f"EXPRESSION: {expression}",
        "",
        "PARSER OUTPUT:",
        parser_output,
    ]
    if context:
        lines += ["", f"CONTEXT (prior steps):\n{context}"]
    return "\n".join(lines)


# ===========================================================================
# 3. Transformer
# ===========================================================================

_TRANSFORMER_SYSTEM = f"""\
You are the Transformer in a symbolic simplification pipeline.

Role
----
Apply exactly ONE simplification rule at exactly ONE node.
All other parts of the expression must remain unchanged.

{_RULES}

Task
----
1. Locate the target_node in the current expression.
2. Apply selected_rule at that node only.
3. Return the full expression with only that substitution made.
4. If the rule cannot be applied, set error to a non-null string
   and leave result_expression null.

Critical constraints
--------------------
  - Apply exactly ONE rule at ONE location.
  - Do not simplify any other node.
  - Preserve all parentheses in untouched parts exactly as given.

{_JSON_CONSTRAINT}

Output schema
-------------
{{
  "result_expression": "<full expression after one rule application, or null on error>",
  "rule_applied": "<rule_name, e.g. add_zero_r>",
  "changed_node": "<what was replaced and what it became, e.g. (x + 0) → x>",
  "error": "<error message or null>",
  "confidence": <0.0–1.0>
}}
"""


def _transformer_format_input(
    expression: str,
    rule_selector_output: str,
    context: Optional[str] = None,
) -> str:
    lines = [
        f"EXPRESSION: {expression}",
        "",
        "RULE SELECTOR OUTPUT:",
        rule_selector_output,
    ]
    if context:
        lines += ["", f"CONTEXT (prior steps):\n{context}"]
    return "\n".join(lines)


# ===========================================================================
# 4. Equivalence Checker
# ===========================================================================

_CHECKER_SYSTEM = """\
You are the Equivalence Checker in a symbolic simplification pipeline.

Role
----
Verify that a simplified expression is mathematically equivalent to the
original by numerical substitution. This is the final verification step.

Task
----
1. Identify all variables appearing in either expression.
2. Evaluate BOTH expressions under at least 3 distinct test assignments.
   Avoid 0 and 1 as values — they can mask errors.
   Suggested assignments: {x:2,y:3,z:5}, {x:-1,y:4,z:7}, {x:3,y:-2,z:6}.
3. If any assignment causes division by zero, skip it and use another.
4. ALL test cases match → verdict "EQUIVALENT".
   ANY mismatch         → verdict "NOT_EQUIVALENT".

Critical constraints:
  - Output MUST be a single valid JSON object ONLY.
  - Do NOT output markdown, reasoning, or any text outside the JSON.
  - If you use <think> tags, place them BEFORE the JSON object.

Output schema
-------------
{
  "variables": ["<var1>", "<var2>"],
  "test_cases": [
    {
      "assignment": {"x": 2, "y": 3},
      "original_result": <number>,
      "simplified_result": <number>,
      "match": true | false
    }
  ],
  "verdict": "EQUIVALENT" | "NOT_EQUIVALENT",
  "reason": "<one sentence summary>",
  "final_answer": "<simplified expression>",
  "confidence": <0.0–1.0>
}
"""


def _checker_format_input(
    original_expression: str,
    simplified_expression: str,
    context: Optional[str] = None,
) -> str:
    lines = [
        f"original_expression  : {original_expression}",
        f"simplified_expression: {simplified_expression}",
    ]
    if context:
        lines += ["", f"CONTEXT (simplification trace):\n{context}"]
    return "\n".join(lines)


# ===========================================================================
# Public interface
# ===========================================================================

PROMPTS: dict[str, dict] = {
    "expression_parser": {
        "system":        _PARSER_SYSTEM,
        "format_input":  _parser_format_input,
        "output_fields": ["tree", "flagged_nodes", "confidence"],
    },
    "rule_selector": {
        "system":        _RULE_SELECTOR_SYSTEM,
        "format_input":  _rule_selector_format_input,
        "output_fields": ["selected_rule", "target_node", "expected_result",
                          "reason", "confidence"],
    },
    "transformer": {
        "system":        _TRANSFORMER_SYSTEM,
        "format_input":  _transformer_format_input,
        "output_fields": ["result_expression", "rule_applied", "changed_node",
                          "error", "confidence"],
    },
    "equivalence_checker": {
        "system":        _CHECKER_SYSTEM,
        "format_input":  _checker_format_input,
        "output_fields": ["variables", "test_cases", "verdict",
                          "reason", "final_answer", "confidence"],
    },
}


def get_system(agent: str) -> str:
    return PROMPTS[agent]["system"]


def format_input(agent: str, **kwargs) -> str:
    return PROMPTS[agent]["format_input"](**kwargs)