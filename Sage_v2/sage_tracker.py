"""
sage_tracker.py — Step-level agent evaluation tracker for SAGE.

For each agent at each simplification step, the tracker verifies the
agent's output against the deterministic ground truth derived from the
task's simplification_trace.

Verification logic per agent:
  Parser      — parsed tree must be semantically equivalent to the input expression
  RuleSelector— selected rule must match trace[step].rule; target node must
                actually match that rule in the current expression
  Transformer — result expression must match trace[step+1].expr exactly
                (with numeric equivalence fallback)
  Checker     — verdict must be EQUIVALENT when expressions are equivalent,
                NOT_EQUIVALENT otherwise

Failure modes are automatically inferred from the nature of the mismatch,
mapped to MAST taxonomy codes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional
from copy import deepcopy

from sage_core import (
    Expr, leaf, is_numeric, all_nodes, full_simplify,
)
from sage_rules import RULE_BY_NAME, RULE_BY_ID, FORWARD_RULES
from sage_protocol import (
    ParserOutput, RuleSelectorOutput, TransformerOutput, CheckerOutput,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
FM_DESCRIPTIONS = {
    # FC1: System Design Issues
    "FM-1.1": "Disobey task specification — Failed to follow task constraints or requirements.",
    "FM-1.2": "Disobey role specification — Failed to adhere to assigned role responsibilities.",
    "FM-1.3": "Step repetition — Unnecessarily repeated a previously completed step.",
    "FM-1.4": "Loss of conversation history — Context truncated; reverted to earlier state.",
    "FM-1.5": "Unaware of termination conditions — Failed to recognize when to stop.",

    # FC2: Inter-Agent Misalignment
    "FM-2.1": "Conversation reset — Dialogue restarted unexpectedly, losing prior context.",
    "FM-2.2": "Fail to ask for clarification — Proceeded without seeking needed information.",
    "FM-2.3": "Task derailment — Deviated from the intended task objective.",
    "FM-2.4": "Information withholding — Failed to share information relevant to other agents.",
    "FM-2.5": "Ignored other agent's input — Disregarded input from another agent.",
    "FM-2.6": "Reasoning-action mismatch — Actions taken did not match the stated reasoning.",

    # FC3: Task Verification
    "FM-3.1": "Premature termination — Ended task before objectives were fully met.",
    "FM-3.2": "No or incomplete verification — Omitted or partial checking of task outputs.",
    "FM-3.3": "Incorrect verification — Validated incorrectly, allowing errors to persist.",
}

@dataclass
class AgentStepResult:
    step: int                        # simplification step index (0-based)
    agent: str                       # "parser" | "rule_selector" | "transformer" | "checker"
    status: str                      # "correct" | "wrong" | "error" | "skip"
    expected: str                    # ground truth string for this check
    actual: str                      # what the agent actually produced
    failure_mode: Optional[str]      # MAST FM code, e.g. "FM-2.6", or None
    failure_mode_desc: Optional[str]      # MAST FM code description
    failure_category: Optional[str]  # "specification" | "inter-agent" | "verification"
    detail: str                      # human-readable explanation

    def is_correct(self) -> bool:
        return self.status == "correct"


@dataclass
class TraceEvaluation:
    task_id: str
    overall: str                          # "success" | "partial" | "failure"
    steps: list[AgentStepResult] = field(default_factory=list)
    first_failure_step: Optional[int] = None
    first_failure_agent: Optional[str] = None
    failure_propagated: bool = False      # did one error corrupt downstream steps?
    anomaly_detected: bool = False        # did any injected anomaly cause a step failure?
    summary: dict = field(default_factory=dict)

    def correct_steps(self) -> int:
        return sum(1 for s in self.steps if s.is_correct())

    def total_steps(self) -> int:
        return len(self.steps)

    def agent_accuracy(self) -> dict:
        agents = ["parser", "rule_selector", "transformer", "checker"]
        result = {}
        for a in agents:
            agent_steps = [s for s in self.steps if s.agent == a]
            if not agent_steps:
                continue
            correct = sum(1 for s in agent_steps if s.is_correct())
            result[a] = {"correct": correct, "total": len(agent_steps),
                         "accuracy": correct / len(agent_steps)}
        return result

    def failure_mode_counts(self) -> dict:
        counts = {}
        for s in self.steps:
            if s.failure_mode:
                counts[s.failure_mode] = counts.get(s.failure_mode, 0) + 1
        return counts


# (key:value _parse_*_output helpers removed — Protocol.parse() handles both
#  JSON and key:value formats automatically via sage_protocol._is_json())


# ---------------------------------------------------------------------------
# Numeric equivalence check (used as fallback for expression comparison)
# ---------------------------------------------------------------------------

def _eval_expr_str(expr_str: str, assignment: dict) -> Optional[float]:
    """Safely evaluate an expression string under a variable assignment."""
    try:
        sanitized = re.sub(r'[^0-9xyzvwabcdefg+\-*/().\s]', '', expr_str)
        return float(eval(sanitized, {"__builtins__": {}}, assignment))
    except Exception:
        return None


_TEST_ASSIGNMENTS = [
    {"x": 2, "y": 3, "z": 5, "w": 7, "a": 2, "b": 3},
    {"x": -1, "y": 4, "z": 2, "w": 3, "a": 5, "b": 1},
    {"x": 3, "y": 2, "z": 4, "w": 5, "a": 3, "b": 7},
]


def _simplify_str(expr_str: str) -> Optional[str]:
    """Parse an expression string and simplify it using the forward rules.
    Returns the simplified string, or None if parsing fails."""
    tree = _parse_expr_safe(expr_str)
    if tree is None:
        return None
    trace = full_simplify(tree)
    return trace[-1]["expr"]


def _strip_outer_parens(s: str) -> str:
    """Remove outer parentheses that wrap the entire expression."""
    while s.startswith("(") and s.endswith(")"):
        depth = 0
        spans_all = False
        for i, ch in enumerate(s):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if depth == 0:
                spans_all = (i == len(s) - 1)
                break
        if spans_all:
            s = s[1:-1].strip()
        else:
            break
    return s


def _str_equal(expr_a: str, expr_b: str) -> bool:
    """Strict string comparison: normalize whitespace and outer parens only."""
    a = _strip_outer_parens(re.sub(r'\s+', ' ', expr_a.strip()))
    b = _strip_outer_parens(re.sub(r'\s+', ' ', expr_b.strip()))
    return a == b


def _expressions_equivalent(expr_a: str, expr_b: str) -> bool:
    """Check equivalence: string equality → rule-based simplification → numeric fallback."""
    # Normalize whitespace
    a = re.sub(r'\s+', ' ', expr_a.strip())
    b = re.sub(r'\s+', ' ', expr_b.strip())
    if a == b:
        return True
    # Rule-based simplification: both expressions should reach the same normal form.
    # This handles cases like (y / 0) * 0 → 0 that numeric eval cannot.
    sa = _simplify_str(a)
    sb = _simplify_str(b)
    if sa is not None and sb is not None and sa == sb:
        return True
    # Numeric fallback
    matches = 0
    tested  = 0
    for assignment in _TEST_ASSIGNMENTS:
        va = _eval_expr_str(a, assignment)
        vb = _eval_expr_str(b, assignment)
        if va is None or vb is None:
            continue
        tested += 1
        if abs(va - vb) < 1e-9:
            matches += 1
    return tested > 0 and matches == tested


# ---------------------------------------------------------------------------
# Failure mode inference
# ---------------------------------------------------------------------------

def _infer_fm_parser(parsed: str, expected: str, status: str) -> tuple:
    """Returns (fm_code, category, detail)."""
    if status and status.upper() == "ERROR":
        return "FM-1.1", "specification", "Parser returned ERROR status"
    if not parsed:
        return "FM-1.1", "specification", "Parser produced no PARSED_TREE field"
    if not _expressions_equivalent(parsed, expected):
        return "FM-1.1", "specification", (
            f"Parsed tree '{parsed}' not equivalent to input '{expected}'"
        )
    return None, None, "correct"


def _resolve_rule(name: Optional[str]):
    """Look up a rule by either its long name (add_zero_r) or short id (r1)."""
    if not name:
        return None
    return RULE_BY_NAME.get(name) or RULE_BY_ID.get(name)


def _rule_name(name: Optional[str]) -> str:
    """Return the canonical rule name (e.g. 'numeric_fold'), falling back to the raw string."""
    if not name:
        return str(name)
    r = _resolve_rule(name)
    return r.name if r else name


def _topmost_applicable(parsed_tree) -> Optional[tuple]:
    """Return (depth, path, node) of the shallowest node where any rule applies.
    Among equal-depth nodes, the leftmost path (lexicographically) wins,
    matching the rule selector's specified selection policy."""
    best = None
    for path, node in all_nodes(parsed_tree):
        for rule in FORWARD_RULES:
            if rule.match(node):
                key = (len(path), path)
                if best is None or key < (best[0], best[1]):
                    best = (len(path), path, node)
                break  # one matching rule per node is enough for depth comparison
    return best


def _infer_fm_rule_selector(
    selected_rule: Optional[str],
    target_node: Optional[str],
    expected_rule: Optional[str],
    current_expr_str: str,
    next_expr_str: Optional[str] = None,
) -> tuple:
    no_rule = not selected_rule or selected_rule.strip().upper() == "NONE"
    if no_rule:
        # Check if any rule actually applies — if not, NONE is correct
        parsed_tree = _parse_expr_safe(current_expr_str)
        if parsed_tree is not None:
            any_applies = any(
                rule.match(node)
                for rule in FORWARD_RULES
                for _, node in all_nodes(parsed_tree)
            )
            if not any_applies:
                return None, None, "correct"
        # NONE when rules still apply = premature termination of simplification
        return "FM-3.1", "verification", f"Rule Selector returned NONE but simplifiable rules remain in expression: {current_expr_str}"

    r_sel = _resolve_rule(selected_rule)
    rule = r_sel

    # ── Step 1: Validate that current expression and target node are parseable ──
    parsed_tree = _parse_expr_safe(current_expr_str)
    if parsed_tree is None:
        return "FM-1.1", "specification", (
            f"Current expression is malformed (cannot parse): {current_expr_str}"
        )

    target_tree = None
    placeholder = target_node in (None, "(matched)", "matched")
    if not placeholder:
        target_tree = _parse_expr_safe(target_node)
        if target_tree is None:
            return "FM-1.1", "specification", (
                f"Target node is malformed (cannot parse): '{target_node}'"
            )

    # ── Step 2: Topmost / leftmost check ──
    # Find the target node's path in the full expression tree.
    target_canonical = str(target_tree) if target_tree is not None else None
    matched_path = None
    if not placeholder:
        for path, node in all_nodes(parsed_tree):
            node_str = str(node)
            if node_str == target_canonical or node_str == target_node:
                matched_path = path
                break
        if matched_path is None:
            return "FM-1.1", "specification", (
                f"Target node '{target_node}' not found in expression: {current_expr_str}"
            )

    topmost = _topmost_applicable(parsed_tree)
    if topmost is not None and matched_path is not None:
        best_depth, best_path, best_node = topmost
        if len(matched_path) > best_depth:
            return "FM-1.1", "specification", (
                f"Non-topmost node: selected '{target_node}' "
                f"(depth {len(matched_path)}) but topmost applicable "
                f"is '{str(best_node)}' (depth {best_depth})"
            )
        if len(matched_path) == best_depth and matched_path != best_path:
            return "FM-1.1", "specification", (
                f"Non-leftmost node at same depth: selected '{target_node}' "
                f"but should prefer '{str(best_node)}'"
            )

    # ── Step 3: Does the selected rule match the target node? ──
    if rule is None:
        return "FM-1.1", "specification", (
            f"Unknown rule: '{selected_rule}'"
        )

    if placeholder:
        # Agent didn't specify a concrete target; check if rule applies anywhere.
        target_valid = any(rule.match(n) for _, n in all_nodes(parsed_tree))
    else:
        target_valid = rule.match(target_tree)

    if not target_valid:
        return "FM-1.1", "specification", (
            f"Rule '{_rule_name(selected_rule)}' does not match "
            f"target node '{target_node}'"
        )

    return None, None, "correct"


def _infer_fm_transformer(
    result: Optional[str],
    expected_result: str,
    prev_result: Optional[str],
    rule_applied: Optional[str] = None,
    selected_rule: Optional[str] = None,
) -> tuple:
    if not result or result.strip() == "":
        return "FM-2.4", "inter-agent", "Transformer returned empty RESULT_EXPRESSION"

    # FM-1.3: step repetition — but only flag if the current expected_result
    # is DIFFERENT from the previous step's result. If they happen to be equal
    # (e.g. final two steps both produce the same simplified form), it is not a fault.
    if (prev_result
            and _expressions_equivalent(result, prev_result)
            and not _expressions_equivalent(result, expected_result)
            or (prev_result
                and _expressions_equivalent(result, prev_result)
                and not _expressions_equivalent(expected_result, prev_result))):
        return "FM-1.3", "specification", (
            "Transformer output matches previous step (step repetition)"
        )

    if not _expressions_equivalent(result, expected_result):
        return "FM-1.3", "specification", (
            f"Transformer output '{result}' != expected '{expected_result}'"
        )

    # FM-2.5: output correct but transformer applied a different rule than instructed
    if rule_applied and selected_rule:
        r_applied  = _resolve_rule(rule_applied)
        r_selected = _resolve_rule(selected_rule)
        if r_applied and r_selected and r_applied is not r_selected:
            return "FM-2.5", "inter-agent", (
                f"Rule-action discrepancy: instructed '{_rule_name(selected_rule)}' "
                f"but transformer applied '{_rule_name(rule_applied)}'"
            )

    return None, None, "correct"


def _infer_fm_checker(
    verdict: Optional[str],
    should_be_equivalent: bool,
    final_expr: str = "",
    ground_truth: str = "",
) -> tuple:
    if not verdict:
        return "FM-3.2", "verification", "Checker produced no VERDICT field"

    verdict_says_equivalent = verdict.strip().upper() == "EQUIVALENT"

    if verdict_says_equivalent != should_be_equivalent:
        if should_be_equivalent:
            return "FM-3.3", "verification", (
                f"Checker returned NOT_EQUIVALENT for equivalent expressions: "
                f"'{final_expr}' and '{ground_truth}'"
            )
        else:
            return "FM-3.3", "verification", (
                f"Checker returned EQUIVALENT for non-equivalent expressions: "
                f"'{final_expr}' and '{ground_truth}'"
            )

    return None, None, "correct"


# ---------------------------------------------------------------------------
# Safe expression parser (best-effort, for node matching)
# ---------------------------------------------------------------------------

def _parse_expr_safe(expr_str: str) -> Optional[Expr]:
    """
    Best-effort parser: converts expression string back to Expr tree.
    Used only for node-level validation in Rule Selector checking.
    """
    expr_str = expr_str.strip()
    try:
        return _parse_recursive(expr_str)
    except Exception:
        return None


_PREC_PARSE = {"+": 1, "-": 1, "*": 2, "/": 2}


def _parse_recursive(s: str) -> Expr:
    s = s.strip()

    # Strip any wrapping parentheses that span the whole expression
    while s.startswith("(") and s.endswith(")"):
        depth = 0
        spans_all = False
        for i, ch in enumerate(s):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if depth == 0:
                spans_all = (i == len(s) - 1)
                break
        if spans_all:
            s = s[1:-1].strip()
        else:
            break

    # Find the rightmost lowest-precedence operator at depth 0.
    # Scanning left-to-right with `<=` gives the rightmost match among
    # operators of equal precedence, which correctly models left-associativity.
    best_pos  = -1
    best_prec = float("inf")
    depth     = 0
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0 and ch in ("+", "-", "*", "/") and i > 0:
            prec = _PREC_PARSE.get(ch, 1)
            if prec <= best_prec:
                best_prec = prec
                best_pos  = i

    if best_pos > 0:
        left  = s[:best_pos].strip()
        right = s[best_pos + 1:].strip()
        return Expr(s[best_pos], _parse_recursive(left), _parse_recursive(right), None)

    return leaf(s)


# ---------------------------------------------------------------------------
# StepTracker — evaluates one step across all four agents
# ---------------------------------------------------------------------------

class StepTracker:
    """
    Evaluates a single simplification step.

    step_index : 0-based index into simplification_trace
    trace      : full simplification_trace from task instance
    """

    def __init__(self, step_index: int, trace: list):
        self.step_index  = step_index
        self.trace       = trace
        self.current_expr = trace[step_index]["expr"]
        self.next_expr    = trace[step_index + 1]["expr"] if step_index + 1 < len(trace) else None
        self.expected_rule = trace[step_index + 1]["rule"] if step_index + 1 < len(trace) else None

    def evaluate_parser(self, parser_output: str) -> AgentStepResult:
        parsed = ParserOutput.parse(parser_output)
        tree   = parsed.parsed_tree or ""
        status = parsed.status or ""

        fm, cat, detail = _infer_fm_parser(tree, self.current_expr, status)

        return AgentStepResult(
            step=self.step_index,
            agent="parser",
            status="correct" if fm is None else "wrong",
            expected=self.current_expr,
            actual=tree,
            failure_mode=fm,
            failure_mode_desc= FM_DESCRIPTIONS.get(fm, None),
            failure_category=cat,
            detail=detail,
        )

    def evaluate_rule_selector(
        self,
        rule_selector_output: str,
        parser_result: Optional[AgentStepResult] = None,
        actual_expression: Optional[str] = None,
    ) -> AgentStepResult:
        """
        parser_result:      if the parser already failed, mark this as "skip".
        actual_expression:  the expression actually shown to the agent (from MAS
                            trace). Falls back to GT expression if None.
        """
        if parser_result and not parser_result.is_correct():
            return AgentStepResult(
                step=self.step_index, agent="rule_selector",
                status="skip",
                expected=self.expected_rule or "",
                actual="(skipped — upstream parser failure)",
                failure_mode=None,
                failure_mode_desc= None,
                failure_category=None,
                detail="Parser failed at this step; Rule Selector evaluation skipped",
            )

        parsed  = RuleSelectorOutput.parse(rule_selector_output)
        sel     = parsed.selected_rule
        target  = parsed.target_node

        # Use the expression the agent actually saw; fall back to GT expression
        expr_for_validation = actual_expression or self.current_expr

        fm, cat, detail = _infer_fm_rule_selector(
            sel, target, self.expected_rule, expr_for_validation,
            next_expr_str=self.next_expr,
        )

        # Include target node in actual so the trace shows both rule and location
        actual_str = _rule_name(sel) if sel else ""
        if target:
            actual_str += f" on '{target}'"

        return AgentStepResult(
            step=self.step_index, agent="rule_selector",
            status="correct" if fm is None else "wrong",
            expected=self.expected_rule or "",
            actual=actual_str,
            failure_mode=fm,
            failure_mode_desc=FM_DESCRIPTIONS.get(fm, None),
            failure_category=cat,
            detail=detail,
        )

    def evaluate_transformer(
        self,
        transformer_output: str,
        prev_transformer_output: Optional[str] = None,
        upstream_failed: bool = False,
        rule_selector_output: str = "",
    ) -> AgentStepResult:
        if upstream_failed:
            return AgentStepResult(
                step=self.step_index, agent="transformer",
                status="skip",
                expected=self.next_expr or "",
                actual="(skipped — upstream failure)",
                failure_mode=None,
                failure_mode_desc=None,
                failure_category=None,
                detail="Upstream agent failed; Transformer evaluation skipped",
            )

        parsed       = TransformerOutput.parse(transformer_output)
        result       = parsed.result_expression or ""
        rule_applied = parsed.rule_applied or ""
        prev_result  = None
        if prev_transformer_output:
            prev_parsed = TransformerOutput.parse(prev_transformer_output)
            prev_result = prev_parsed.result_expression

        # Extract selected_rule from rule_selector output for discrepancy check
        selected_rule = RuleSelectorOutput.parse(rule_selector_output).selected_rule if rule_selector_output else ""

        fm, cat, detail = _infer_fm_transformer(
            result, self.next_expr or "", prev_result,
            rule_applied=rule_applied, selected_rule=selected_rule,
        )

        return AgentStepResult(
            step=self.step_index, agent="transformer",
            status="correct" if fm is None else "wrong",
            expected=self.next_expr or "",
            actual=result,
            failure_mode=fm,
            failure_mode_desc=FM_DESCRIPTIONS.get(fm, None),
            failure_category=cat,
            detail=detail,
        )

    def evaluate_checker(
        self,
        checker_output: str,
        ground_truth: str,
        transformer_result: Optional[AgentStepResult] = None,
    ) -> AgentStepResult:
        """
        Checker is evaluated at the FINAL step only.
        ground_truth: the expected fully-simplified expression.
        """
        parsed  = CheckerOutput.parse(checker_output)
        verdict = parsed.verdict

        # Determine what the correct verdict should be.
        # At the final step next_expr IS the ground truth (last trace entry).
        # We compare ground_truth with itself — should always be True.
        # The checker should say EQUIVALENT when the simplified result == GT.
        final_expr = self.next_expr if self.next_expr else self.current_expr
        should_be_equiv = _expressions_equivalent(final_expr, ground_truth)

        fm, cat, detail = _infer_fm_checker(verdict, should_be_equiv, final_expr, ground_truth)

        return AgentStepResult(
            step=self.step_index, agent="checker",
            status="correct" if fm is None else "wrong",
            expected="EQUIVALENT" if should_be_equiv else "NOT_EQUIVALENT",
            actual=verdict or "",
            failure_mode=fm,
            failure_mode_desc=FM_DESCRIPTIONS.get(fm, None),
            failure_category=cat,
            detail=detail,
        )


# ---------------------------------------------------------------------------
# SAGETracker — full trace evaluation
# ---------------------------------------------------------------------------

class SAGETracker:
    """
    Evaluates a complete MAS run against a SAGE task instance.

    Usage:
        tracker = SAGETracker(task)
        tracker.record_step(
            step=0,
            parser_output="PARSED_TREE: ...",
            rule_selector_output="SELECTED_RULE: ...",
            transformer_output="RESULT_EXPRESSION: ...",
        )
        # On the final step, also provide checker_output
        tracker.record_step(
            step=k-1,
            parser_output=...,
            rule_selector_output=...,
            transformer_output=...,
            checker_output="VERDICT: EQUIVALENT\n...",
        )
        evaluation = tracker.evaluate()
    """

    def __init__(self, task: dict):
        self.task        = task
        self.trace       = task["simplification_trace"]
        self.ground_truth = task["ground_truth"]
        self.k_steps     = len(task["simplification_trace"]) - 1  # GT rounds; overridden by runner with actual MAS rounds
        self._records: dict[int, dict] = {}   # step -> {agent -> output_str}
        self._prev_transformer: dict[int, str] = {}
        self.final_result = ""

    def record_from_trace(self, mas_trace: list[dict]) -> Optional[str]:
        """Parse MAS orchestrator trace and call record_step for each complete loop.

        Returns None on success, or an error string if recording fails.
        """
        by_loop: dict[int, dict] = {}
        checker_raw = ""
        self.final_result = self.get_final_answer(mas_trace)

        for entry in mas_trace:
            agent = entry["agent"]
            phase = entry["phase"]
            raw = (entry["output"].get("reasoning", "")
                   if isinstance(entry.get("output"), dict)
                   else str(entry.get("output", "")))

            if phase == "verification":
                checker_raw = raw
                continue
            if phase == "parsing":
                by_loop.setdefault(0, {})["parser"] = raw
                continue

            # phase = "loop_N"
            try:
                n = int(phase.split("_")[1]) - 1  # loop_1 → step 0
            except (IndexError, ValueError):
                continue

            by_loop.setdefault(n, {})[agent] = raw
            if "expression" in entry:
                by_loop[n]["expression"] = entry["expression"]

        # Complete loops = those with a transformer output
        complete_loops = sorted(
            n for n, slot in by_loop.items() if slot.get("transformer")
        )
        actual_k_steps = len(complete_loops)
        gt_k_steps = len(self.trace) - 1
        self.k_steps = min(actual_k_steps, gt_k_steps)

        k = actual_k_steps
        for step, loop_n in enumerate(complete_loops):
            slot = by_loop.get(loop_n, {})
            parser_out = slot.get("expression_parser", slot.get("parser", "")) if step == 0 else None
            rs_out = slot.get("rule_selector", "")
            tr_out = slot.get("transformer", "")
            ck_out = checker_raw if step == k - 1 else ""

            try:
                self.record_step(
                    step=step,
                    parser=parser_out,
                    rule_selector=rs_out,
                    transformer=tr_out,
                    checker=ck_out,
                    actual_expression=slot.get("expression"),
                )
            except ValueError as e:
                return f"record_step error at step {step}: {e}"

        return None

    def record_step(
        self,
        step: int,
        parser: Optional[str] = None,   # None for loop steps (parser not called)
        rule_selector: str = "",
        transformer: str = "",
        checker: str = "",
        actual_expression: Optional[str] = None,  # actual expr shown to agent (from MAS trace)
    ) -> None:
        """Record agent outputs for one simplification step.

        parser should only be provided for step 0 (initial parse).
        For loop steps (step > 0), pass None — the tracker will skip parser evaluation.
        """
        if step < 0 or step >= self.k_steps:
            raise ValueError(f"step {step} out of range [0, {self.k_steps})")
        self._records[step] = {
            "parser":             parser,            # None → skip parser eval
            "rule_selector":      rule_selector,
            "transformer":        transformer,
            "checker":            checker,
            "actual_expression":  actual_expression, # actual expr from MAS trace
        }

    def agent_evaluate(self, task_id: str = "unknown") -> TraceEvaluation:
        """
        Run evaluation across all recorded steps.
        Returns a TraceEvaluation with per-step, per-agent results.
        """
        all_agent_results: list[AgentStepResult] = []
        first_failure_step  = None
        first_failure_agent = None
        prev_transformer_output = None
        prev_step_had_transformer_failure = False

        for step in range(self.k_steps):
            outputs = self._records.get(step, {})
            st = StepTracker(step, self.trace)

            # --- Parser (step 0 only; parser does not run inside the loop) ---
            parser_raw = outputs.get("parser")
            if step == 0:
                parser_res: Optional[AgentStepResult] = st.evaluate_parser(parser_raw or "")
            else:
                parser_res = None   # not evaluated for loop steps

            # --- Rule Selector ---
            rs_res = st.evaluate_rule_selector(
                outputs.get("rule_selector", ""),
                parser_result=parser_res,            # None → no upstream skip check
                actual_expression=outputs.get("actual_expression"),
            )

            # --- Transformer ---
            # upstream_failed: parser wrong at step 0, or rs wrong, or
            # previous transformer produced a wrong result
            parser_failed = (parser_res is not None and not parser_res.is_correct())
            upstream_failed = (parser_failed
                               or not rs_res.is_correct()
                               or prev_step_had_transformer_failure)
            tr_res = st.evaluate_transformer(
                outputs.get("transformer", ""),
                prev_transformer_output=prev_transformer_output,
                upstream_failed=upstream_failed,
                rule_selector_output=outputs.get("rule_selector", ""),
            )
            if tr_res.status != "skip":
                prev_transformer_output = outputs.get("transformer", "")
            prev_step_had_transformer_failure = tr_res.status == "wrong"

            # --- Checker (final step only) ---
            ck_res = None
            if step == self.k_steps - 1:
                ck_res = st.evaluate_checker(
                    outputs.get("checker", ""),
                    ground_truth=self.ground_truth,
                    transformer_result=tr_res,
                )

            # Collect results — parser only included at step 0
            step_results: list[AgentStepResult] = []
            if parser_res is not None:
                step_results.append(parser_res)
            step_results.extend([rs_res, tr_res])
            if ck_res:
                step_results.append(ck_res)
            all_agent_results.extend(step_results)

            # Track first failure
            for res in step_results:
                if res.status == "wrong" and first_failure_step is None:
                    first_failure_step  = step
                    first_failure_agent = res.agent


        agent_wrong = self.get_wrong_agents(all_agent_results)
        # Anomaly detected: any wrong result
        anomaly_detected = len(agent_wrong) > 0

        overall = self.evaluate_task()
        if overall == "success" and agent_wrong:
            overall = "partial"

        # Check if error propagated: a skip appeared in a step AFTER the first failure
        propagated = (
            first_failure_step is not None and
            any(
                r.status == "skip" and r.step > first_failure_step
                for r in all_agent_results
            )
        )

        eval_result = TraceEvaluation(
            task_id=task_id,
            overall=overall,
            steps=all_agent_results,
            first_failure_step=first_failure_step,
            first_failure_agent=first_failure_agent,
            failure_propagated=propagated,
            anomaly_detected=anomaly_detected,
        )
        eval_result.summary = self._build_summary(eval_result)
        return eval_result

    @staticmethod
    def get_wrong_agents(all_agent_results: list) -> list:
        """Return list of AgentStepResults with status 'wrong'."""
        if not all_agent_results:
            return []
        return [r for r in all_agent_results if r.status == "wrong"]

    def get_final_answer(self, mas_trace: list[dict] = None) -> str:
        """Determine the final simplified expression.

        Priority:
          1. MAS trace: final_expression from verification phase (orchestrator's current_expression)
          2. Agent results: last transformer's actual output
          3. Fallback: original expression (no transformation happened)
        """
        # 1. From MAS trace (most reliable — orchestrator's own tracking)

        for entry in mas_trace:
            if entry.get("phase") == "verification":
                final = entry.get("final_expression")
                if final:
                    return final

        # 3. Fallback: no simplification happened
        return self.trace[0]["expr"]

    def evaluate_task(self):

        self.task_success = "success" if  _str_equal(self.final_result, self.ground_truth) else "failure"
        return self.task_success


    @staticmethod
    def _build_summary(ev: TraceEvaluation) -> dict:
        return {
            "overall":             ev.overall,
            "correct_steps":       ev.correct_steps(),
            "total_steps":         ev.total_steps(),
            "first_failure_step":  ev.first_failure_step,
            "first_failure_agent": ev.first_failure_agent,
            "failure_propagated":  ev.failure_propagated,
            "agent_accuracy":      ev.agent_accuracy(),
            "failure_mode_counts": ev.failure_mode_counts(),
        }


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_evaluation(ev: TraceEvaluation) -> None:
    status_icon = {"correct": "✓", "wrong": "✗", "skip": "~", "error": "!"}
    print(f"\n{'='*60}")
    print(f"Task: {ev.task_id}  |  Overall: {ev.overall.upper()}")
    print(f"{'='*60}")

    current_step = -1
    for res in ev.steps:
        if res.step != current_step:
            current_step = res.step
            print(f"\n  Step {res.step}")
        icon = status_icon.get(res.status, "?")
        fm   = f"  [{res.failure_mode}]" if res.failure_mode else ""
        print(f"    {icon} {res.agent:<16} {res.status:<8}{fm}")
        if res.status == "wrong":
            print(f"       expected : {res.expected[:60]}")
            print(f"       actual   : {res.actual[:60]}")
            print(f"       detail   : {res.detail[:80]}")

    print(f"\n  Summary:")
    for k, v in ev.summary.items():
        if k not in ("agent_accuracy", "failure_mode_counts"):
            print(f"    {k:<25}: {v}")
    print(f"    agent_accuracy:")
    for agent, acc in ev.summary.get("agent_accuracy", {}).items():
        print(f"      {agent:<16}: {acc['correct']}/{acc['total']}")
    if ev.summary.get("failure_mode_counts"):
        print(f"    failure_modes      : {ev.summary['failure_mode_counts']}")
    print()