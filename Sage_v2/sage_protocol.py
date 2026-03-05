"""
sage_protocol.py — Inter-agent message protocol for SAGE.

Defines the structured message types passed between the four agents:
  Parser → RuleSelectorInput
  RuleSelector → TransformerInput
  Transformer → CheckerInput
  Checker → (final output)

Each message type has:
  - A dataclass with typed fields
  - format() → str   (for injection into agent prompts)
  - parse(str) → T   (for parsing agent output back to structured form)
  - validate()       (for tracker / anomaly injector to check field presence)

Parse is lenient: missing optional fields return None rather than raising.
Validate is strict: raises ProtocolError for missing required fields.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, fields
from typing import Optional


# ---------------------------------------------------------------------------
# Base helpers
# ---------------------------------------------------------------------------

class ProtocolError(ValueError):
    """Raised when a required protocol field is missing or malformed."""


def _extract(text: str, field_name: str) -> Optional[str]:
    """
    Extract 'FIELD_NAME: value' from text.
    Returns empty string if field present but empty, None if field absent.
    """
    pattern = "^" + re.escape(field_name) + r":[ \t]*([^\r\n]*)"
    m = re.search(pattern, text, re.MULTILINE)
    if m is None:
        return None
    return m.group(1).strip()


def _extract_block(text: str, field_name: str) -> Optional[str]:
    """
    Extract a multi-line block started by FIELD_NAME: and ended by the
    next all-caps field or end of string.
    """
    pattern = (
        r"^" + re.escape(field_name) + r":[ \t]*\n"
        r"((?:(?!^[A-Z_]+:)[\s\S])*)"
    )
    m = re.search(pattern, text, re.MULTILINE)
    if m is None:
        return None
    return m.group(1).strip()


# ---------------------------------------------------------------------------
# JSON parser helper
# ---------------------------------------------------------------------------

def _parse_json(text: str) -> Optional[dict]:
    """
    Extract and parse a JSON object from LLM output.
    Handles:
      - Raw JSON
      - ```json ... ``` fenced blocks
      - <think>...</think> prefix (DeepSeek-R1 style)
      - Trailing commas, single quotes (best-effort)
    Returns None if parsing fails.
    """
    import json as _json
    import re as _re

    # Strip <think>...</think> blocks
    text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL).strip()

    # Try fenced block first
    fenced = _re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, _re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        # Find first { ... } span
        start = text.find("{")
        end   = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start:end+1]

    # Best-effort fixes
    text = _re.sub(r",\s*([}\]])", r"", text)  # trailing commas
    # Try parsing as-is first (handles well-formed JSON that may
    # contain single quotes inside string values).
    try:
        return _json.loads(text)
    except Exception:
        pass

    # Fallback: replace single quotes with double quotes (for LLM output
    # that used single-quoted keys/values like {'key': 'val'}).
    try:
        return _json.loads(text.replace("'", '"'))
    except Exception:
        return None


def _is_json(text: str) -> bool:
    """Return True if text appears to be JSON output."""
    stripped = text.strip()
    # Check after stripping think tags
    import re as _re
    stripped = _re.sub(r"<think>.*?</think>", "", stripped, flags=_re.DOTALL).strip()
    return stripped.startswith("{") or "```json" in stripped or "```\n{" in stripped


# ---------------------------------------------------------------------------
# Status codes
# ---------------------------------------------------------------------------

class Status:
    OK    = "OK"
    ERROR = "ERROR"


# ---------------------------------------------------------------------------
# 1. ParserOutput
#    Parser receives: raw expression string
#    Parser produces: ParserOutput
# ---------------------------------------------------------------------------

@dataclass
class ParserOutput:
    """
    Output from the Expression Parser agent.

    Fields
    ------
    parsed_tree : str
        The canonical string representation of the parsed expression tree.
        Must be semantically equivalent to the input expression.
    simplifiable_nodes : list[str]
        Each entry describes one reducible node as "path | node | rule".
        e.g. "root.left | (x + 0) | add_zero_r"
    status : str
        "OK" or "ERROR".
    error_message : str | None
        Human-readable error description (only when status == ERROR).
    """
    parsed_tree:          str
    simplifiable_nodes:   list[str]      = field(default_factory=list)
    status:               str            = Status.OK
    error_message:        Optional[str]  = None

    # ------------------------------------------------------------------
    # Formatting (agent sees this as its output template)
    # ------------------------------------------------------------------

    @staticmethod
    def template() -> str:
        return (
            "PARSED_TREE: <canonical expression string>\n"
            "SIMPLIFIABLE_NODES:\n"
            "  - <path> | <node> | <matching_rule>\n"
            "  - ...\n"
            "STATUS: OK"
        )

    def format(self) -> str:
        lines = [f"PARSED_TREE: {self.parsed_tree}"]
        if self.simplifiable_nodes:
            lines.append("SIMPLIFIABLE_NODES:")
            for node in self.simplifiable_nodes:
                lines.append(f"  - {node}")
        else:
            lines.append("SIMPLIFIABLE_NODES: none")
        lines.append(f"STATUS: {self.status}")
        if self.error_message:
            lines.append(f"ERROR_MESSAGE: {self.error_message}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Parsing (tracker reads agent output)
    # ------------------------------------------------------------------

    @classmethod
    def parse_json(cls, d: dict) -> "ParserOutput":
        """Construct from a parsed JSON dict (original AutoGenMAS format)."""
        # Flatten tree to canonical string if needed
        tree = d.get("tree", {})
        if isinstance(tree, dict):
            tree_str = cls._tree_to_str(tree)
        else:
            tree_str = str(tree)
        flagged = d.get("flagged_nodes", [])
        nodes = [
            f"{n.get('node_description', '')} | {n.get('matched_rule', '')}"
            for n in flagged
        ] if isinstance(flagged, list) else []
        return cls(
            parsed_tree=tree_str,
            simplifiable_nodes=nodes,
            status=Status.OK if not d.get("error") else Status.ERROR,
            error_message=d.get("error"),
        )

    @staticmethod
    def _tree_to_str(node: dict) -> str:
        """Recursively convert a tree dict back to a parenthesized string."""
        if not node:
            return ""
        if node.get("type") == "LEAF":
            return str(node.get("value", "?"))
        op    = node.get("op", "?")
        left  = ParserOutput._tree_to_str(node.get("left",  {}))
        right = ParserOutput._tree_to_str(node.get("right", {}))
        return f"({left} {op} {right})"

    @classmethod
    def parse(cls, text: str) -> "ParserOutput":
        """Auto-detect JSON or key:value format and parse accordingly."""
        if _is_json(text):
            d = _parse_json(text)
            if d is not None:
                return cls.parse_json(d)
        # Fallback: key:value
        parsed_tree = _extract(text, "PARSED_TREE") or ""
        status      = _extract(text, "STATUS") or Status.OK
        error_msg   = _extract(text, "ERROR_MESSAGE")
        nodes: list[str] = []
        block = _extract_block(text, "SIMPLIFIABLE_NODES")
        if block:
            for line in block.splitlines():
                line = line.strip().lstrip("- ").strip()
                if line and line.lower() != "none":
                    nodes.append(line)
        else:
            single = _extract(text, "SIMPLIFIABLE_NODES")
            if single and single.lower() != "none":
                nodes = [single]
        return cls(
            parsed_tree=parsed_tree,
            simplifiable_nodes=nodes,
            status=status,
            error_message=error_msg,
        )

    def validate(self) -> None:
        if not self.parsed_tree:
            raise ProtocolError("ParserOutput: PARSED_TREE is required")
        if self.status not in (Status.OK, Status.ERROR):
            raise ProtocolError(f"ParserOutput: unknown STATUS '{self.status}'")

    def is_ok(self) -> bool:
        return self.status == Status.OK and bool(self.parsed_tree)


# ---------------------------------------------------------------------------
# 2. RuleSelectorOutput
#    RuleSelector receives: ParserOutput + current expression
#    RuleSelector produces: RuleSelectorOutput
# ---------------------------------------------------------------------------

@dataclass
class RuleSelectorOutput:
    """
    Output from the Rule Selector agent.

    Fields
    ------
    selected_rule : str
        Rule id or name, e.g. "add_zero_r" or "r1".
    target_node : str
        The sub-expression to which the rule will be applied,
        e.g. "(x + 0)".
    target_path : str | None
        Dot-separated path to the target node, e.g. "root.left".
        Optional but recommended for precise targeting.
    expected_result : str
        The expected sub-expression after applying the rule,
        e.g. "x".
    reasoning : str | None
        Brief explanation of why this rule was selected.
    status : str
    error_message : str | None
    """
    selected_rule:   str
    target_node:     str
    expected_result: str
    target_path:     Optional[str]  = None
    reasoning:       Optional[str]  = None
    status:          str            = Status.OK
    error_message:   Optional[str]  = None

    @staticmethod
    def template() -> str:
        return (
            "SELECTED_RULE: <rule_name>\n"
            "TARGET_NODE: <sub-expression>\n"
            "TARGET_PATH: <dot.path.to.node>  (optional)\n"
            "EXPECTED_RESULT: <simplified sub-expression>\n"
            "REASONING: <brief explanation>\n"
            "STATUS: OK"
        )

    def format(self) -> str:
        lines = [
            f"SELECTED_RULE: {self.selected_rule}",
            f"TARGET_NODE: {self.target_node}",
        ]
        if self.target_path:
            lines.append(f"TARGET_PATH: {self.target_path}")
        lines.append(f"EXPECTED_RESULT: {self.expected_result}")
        if self.reasoning:
            lines.append(f"REASONING: {self.reasoning}")
        lines.append(f"STATUS: {self.status}")
        if self.error_message:
            lines.append(f"ERROR_MESSAGE: {self.error_message}")
        return "\n".join(lines)

    @classmethod
    def parse_json(cls, d: dict) -> "RuleSelectorOutput":
        return cls(
            selected_rule   = d.get("selected_rule", ""),
            target_node     = d.get("target_node")   or "",
            expected_result = d.get("expected_result") or "",
            target_path     = None,
            reasoning       = d.get("reason"),
            status          = Status.OK if not d.get("error") else Status.ERROR,
            error_message   = d.get("error"),
        )

    @classmethod
    def parse(cls, text: str) -> "RuleSelectorOutput":
        if _is_json(text):
            d = _parse_json(text)
            if d is not None:
                return cls.parse_json(d)
        return cls(
            selected_rule   = _extract(text, "SELECTED_RULE")   or "",
            target_node     = _extract(text, "TARGET_NODE")     or "",
            expected_result = _extract(text, "EXPECTED_RESULT") or "",
            target_path     = _extract(text, "TARGET_PATH"),
            reasoning       = _extract(text, "REASONING"),
            status          = _extract(text, "STATUS") or Status.OK,
            error_message   = _extract(text, "ERROR_MESSAGE"),
        )

    def validate(self) -> None:
        missing = []
        if not self.selected_rule:
            missing.append("SELECTED_RULE")
        if not self.target_node:
            missing.append("TARGET_NODE")
        if not self.expected_result:
            missing.append("EXPECTED_RESULT")
        if missing:
            raise ProtocolError(
                f"RuleSelectorOutput: missing required fields: {missing}"
            )

    def is_ok(self) -> bool:
        return (self.status == Status.OK
                and bool(self.selected_rule)
                and bool(self.target_node))


# ---------------------------------------------------------------------------
# 3. TransformerOutput
#    Transformer receives: RuleSelectorOutput + current expression
#    Transformer produces: TransformerOutput
# ---------------------------------------------------------------------------

@dataclass
class TransformerOutput:
    """
    Output from the Transformer agent.

    Fields
    ------
    result_expression : str
        The full expression after applying exactly one rule to one node.
    rule_applied : str
        The rule that was actually applied (should match selected_rule).
    changed_node_before : str | None
        The sub-expression before transformation (for audit).
    changed_node_after : str | None
        The sub-expression after transformation (for audit).
    status : str
    error_message : str | None
        Set when the rule could not be applied (e.g. node not found).
    """
    result_expression:    str
    rule_applied:         str
    changed_node_before:  Optional[str]  = None
    changed_node_after:   Optional[str]  = None
    status:               str            = Status.OK
    error_message:        Optional[str]  = None

    @staticmethod
    def template() -> str:
        return (
            "RESULT_EXPRESSION: <full expression after one rule application>\n"
            "RULE_APPLIED: <rule_name>\n"
            "CHANGED_NODE_BEFORE: <sub-expression before>  (optional)\n"
            "CHANGED_NODE_AFTER: <sub-expression after>   (optional)\n"
            "STATUS: OK"
        )

    def format(self) -> str:
        lines = [
            f"RESULT_EXPRESSION: {self.result_expression}",
            f"RULE_APPLIED: {self.rule_applied}",
        ]
        if self.changed_node_before is not None:
            lines.append(f"CHANGED_NODE_BEFORE: {self.changed_node_before}")
        if self.changed_node_after is not None:
            lines.append(f"CHANGED_NODE_AFTER: {self.changed_node_after}")
        lines.append(f"STATUS: {self.status}")
        if self.error_message:
            lines.append(f"ERROR_MESSAGE: {self.error_message}")
        return "\n".join(lines)

    @classmethod
    def parse_json(cls, d: dict) -> "TransformerOutput":
        changed = d.get("changed_node", "") or ""
        # "changed_node" in original config: "(x+0) → x" style string
        before, after = None, None
        if "→" in changed:
            parts = changed.split("→", 1)
            before = parts[0].strip()
            after  = parts[1].strip()
        elif "->" in changed:
            parts = changed.split("->", 1)
            before = parts[0].strip()
            after  = parts[1].strip()
        return cls(
            result_expression   = d.get("result_expression") or "",
            rule_applied        = d.get("rule_applied") or "",
            changed_node_before = before,
            changed_node_after  = after,
            status              = Status.OK if not d.get("error") else Status.ERROR,
            error_message       = d.get("error"),
        )

    @classmethod
    def parse(cls, text: str) -> "TransformerOutput":
        if _is_json(text):
            d = _parse_json(text)
            if d is not None:
                return cls.parse_json(d)
        return cls(
            result_expression   = _extract(text, "RESULT_EXPRESSION")    or "",
            rule_applied        = _extract(text, "RULE_APPLIED")         or "",
            changed_node_before = _extract(text, "CHANGED_NODE_BEFORE"),
            changed_node_after  = _extract(text, "CHANGED_NODE_AFTER"),
            status              = _extract(text, "STATUS") or Status.OK,
            error_message       = _extract(text, "ERROR_MESSAGE"),
        )

    def validate(self) -> None:
        missing = []
        if not self.result_expression:
            missing.append("RESULT_EXPRESSION")
        if not self.rule_applied:
            missing.append("RULE_APPLIED")
        if missing:
            raise ProtocolError(
                f"TransformerOutput: missing required fields: {missing}"
            )

    def is_ok(self) -> bool:
        return (self.status == Status.OK
                and bool(self.result_expression))


# ---------------------------------------------------------------------------
# 4. CheckerOutput
#    Checker receives: original expression + TransformerOutput (final result)
#    Checker produces: CheckerOutput
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    assignment: dict          # e.g. {"x": 2, "y": 3}
    original_value: float
    simplified_value: float
    match: bool

    def format(self) -> str:
        assign_str = ", ".join(f"{k}={v}" for k, v in self.assignment.items())
        status = "match" if self.match else "MISMATCH"
        return (f"  assignment: {{{assign_str}}}, "
                f"original={self.original_value}, "
                f"simplified={self.simplified_value}, "
                f"status={status}")


@dataclass
class CheckerOutput:
    """
    Output from the Equivalence Checker agent.

    Fields
    ------
    verdict : str
        "EQUIVALENT" or "NOT_EQUIVALENT".
    test_cases : list[str]
        Each entry is a formatted test case string (raw, for logging).
    num_tests : int
        Number of test cases evaluated.
    reason : str | None
        Brief explanation of the verdict.
    status : str
    error_message : str | None
    """
    verdict:       str
    test_cases:    list[str]     = field(default_factory=list)
    num_tests:     int           = 0
    reason:        Optional[str] = None
    status:        str           = Status.OK
    error_message: Optional[str] = None

    EQUIVALENT     = "EQUIVALENT"
    NOT_EQUIVALENT = "NOT_EQUIVALENT"

    @staticmethod
    def template() -> str:
        return (
            "TEST_CASES:\n"
            "  assignment: {x: 2, y: 3}, original=<val>, simplified=<val>, status=match\n"
            "  assignment: {x: -1, y: 4}, original=<val>, simplified=<val>, status=match\n"
            "  assignment: {x: 3, y: 2}, original=<val>, simplified=<val>, status=match\n"
            "NUM_TESTS: 3\n"
            "VERDICT: EQUIVALENT\n"
            "REASON: all 3 test cases matched\n"
            "STATUS: OK"
        )

    def format(self) -> str:
        lines = ["TEST_CASES:"]
        for tc in self.test_cases:
            lines.append(f"  {tc}")
        lines.append(f"NUM_TESTS: {self.num_tests or len(self.test_cases)}")
        lines.append(f"VERDICT: {self.verdict}")
        if self.reason:
            lines.append(f"REASON: {self.reason}")
        lines.append(f"STATUS: {self.status}")
        if self.error_message:
            lines.append(f"ERROR_MESSAGE: {self.error_message}")
        return "\n".join(lines)

    @classmethod
    def parse_json(cls, d: dict) -> "CheckerOutput":
        raw_cases = d.get("test_cases", [])
        test_cases = []
        for tc in raw_cases:
            if isinstance(tc, dict):
                assign = tc.get("assignment", {})
                test_cases.append(
                    f"assignment: {assign}, "
                    f"original={tc.get('original_result')}, "
                    f"simplified={tc.get('simplified_result')}, "
                    f"match={tc.get('match')}"
                )
            else:
                test_cases.append(str(tc))
        return cls(
            verdict     = d.get("verdict", ""),
            test_cases  = test_cases,
            num_tests   = len(test_cases),
            reason      = d.get("reason"),
            status      = Status.OK if not d.get("error") else Status.ERROR,
            error_message = d.get("error"),
        )

    @classmethod
    def parse(cls, text: str) -> "CheckerOutput":
        if _is_json(text):
            d = _parse_json(text)
            if d is not None:
                return cls.parse_json(d)
        # key:value fallback
        verdict   = _extract(text, "VERDICT") or ""
        num_tests = _extract(text, "NUM_TESTS")
        reason    = _extract(text, "REASON")
        status    = _extract(text, "STATUS") or Status.OK
        error_msg = _extract(text, "ERROR_MESSAGE")
        test_cases: list[str] = []
        block = _extract_block(text, "TEST_CASES")
        if block:
            for line in block.splitlines():
                line = line.strip().lstrip("- ").strip()
                if line:
                    test_cases.append(line)
        return cls(
            verdict     = verdict,
            test_cases  = test_cases,
            num_tests   = int(num_tests) if num_tests and num_tests.isdigit() else len(test_cases),
            reason      = reason,
            status      = status,
            error_message = error_msg,
        )

    def validate(self) -> None:
        if self.verdict not in (self.EQUIVALENT, self.NOT_EQUIVALENT):
            raise ProtocolError(
                f"CheckerOutput: VERDICT must be EQUIVALENT or NOT_EQUIVALENT, "
                f"got '{self.verdict}'"
            )

    def is_ok(self) -> bool:
        return (self.status == Status.OK
                and self.verdict in (self.EQUIVALENT, self.NOT_EQUIVALENT))

    def is_equivalent(self) -> bool:
        return self.verdict == self.EQUIVALENT


# ---------------------------------------------------------------------------
# Protocol registry — maps agent name to its output class
# ---------------------------------------------------------------------------

PROTOCOL_CLASSES = {
    "parser":        ParserOutput,
    "rule_selector": RuleSelectorOutput,
    "transformer":   TransformerOutput,
    "checker":       CheckerOutput,
}


def parse_agent_output(agent: str, text: str):
    """Parse agent output text into the appropriate protocol dataclass."""
    cls = PROTOCOL_CLASSES.get(agent)
    if cls is None:
        raise ValueError(f"Unknown agent '{agent}'. "
                         f"Valid agents: {list(PROTOCOL_CLASSES.keys())}")
    return cls.parse(text)


# ---------------------------------------------------------------------------
# Suggested test values for Checker
# ---------------------------------------------------------------------------

CHECKER_TEST_ASSIGNMENTS = [
    {"x": 2,  "y": 3,  "z": 5,  "w": 7},
    {"x": -1, "y": 4,  "z": 2,  "w": 3},
    {"x": 3,  "y": 2,  "z": 4,  "w": 5},
]


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Protocol format examples ===\n")

    print("[Parser template]")
    print(ParserOutput.template())
    print()

    print("[Parser format example]")
    po = ParserOutput(
        parsed_tree="((x + 0) - y)",
        simplifiable_nodes=["root.left | (x + 0) | add_zero_r"],
    )
    print(po.format())
    print()

    print("[Parser parse round-trip]")
    po2 = ParserOutput.parse(po.format())
    assert po2.parsed_tree == po.parsed_tree
    assert po2.simplifiable_nodes == po.simplifiable_nodes
    print("round-trip OK")
    print()

    print("[RuleSelector format example]")
    rs = RuleSelectorOutput(
        selected_rule="add_zero_r",
        target_node="(x + 0)",
        target_path="root.left",
        expected_result="x",
        reasoning="node (x + 0) matches pattern x+0→x",
    )
    print(rs.format())
    print()

    print("[Transformer format example]")
    tr = TransformerOutput(
        result_expression="(x - y)",
        rule_applied="add_zero_r",
        changed_node_before="(x + 0)",
        changed_node_after="x",
    )
    print(tr.format())
    print()

    print("[Checker format example]")
    ck = CheckerOutput(
        verdict=CheckerOutput.EQUIVALENT,
        test_cases=[
            "assignment: {x=2, y=3}, original=-1, simplified=-1, status=match",
            "assignment: {x=-1, y=4}, original=-5, simplified=-5, status=match",
            "assignment: {x=3, y=2}, original=1, simplified=1, status=match",
        ],
        num_tests=3,
        reason="all 3 test cases matched",
    )
    print(ck.format())
    print()

    print("[Checker parse]")
    ck2 = CheckerOutput.parse(ck.format())
    assert ck2.verdict == CheckerOutput.EQUIVALENT
    assert ck2.num_tests == 3
    print("parse OK, verdict:", ck2.verdict, "num_tests:", ck2.num_tests)
