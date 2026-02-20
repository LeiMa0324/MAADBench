"""
sage_orchestrator.py — SAGE-specific MAS Orchestrator

Inherits AutoGenMAS and overrides _run_sequential to implement
the symbolic simplification loop:

  expression_parser (×1)
    └─> loop:
          rule_selector  → if NONE or max_steps reached: break
          transformer    → update current expression
    └─> equivalence_checker (×1)
"""

import json
from typing import Any, Dict, Optional

from mas_framework.mas import AutoGenMAS, Agent, AgentOutput


# ── Helpers ────────────────────────────────────────────────────────────────────

def _print_step(step: int, role: str, output: AgentOutput) -> None:
    """Pretty-print a single agent step to stdout."""
    bar = "─" * 60
    print(f"\n{bar}")
    print(f"  STEP {step:>2} │ {role.upper()}")
    print(bar)
    content = output.content
    if "error" in content and content["error"]:
        print(f"  ⚠  ERROR: {content['error']}")
    else:
        print(json.dumps(content, indent=4, ensure_ascii=False))
    print(f"  confidence: {output.confidence:.2f}")


def _print_summary(
    original: str,
    steps: list[str],
    final: str,
    verdict: str,
    reason: str,
) -> None:
    """Print the final simplification summary."""
    bar = "═" * 60
    print(f"\n{bar}")
    print("  SAGE SIMPLIFICATION SUMMARY")
    print(bar)
    print(f"  Original   : {original}")
    for i, expr in enumerate(steps, 1):
        print(f"  Step {i:>2}    : {expr}")
    print(f"  Final      : {final}")
    print(f"  Verdict    : {verdict}")
    print(f"  Reason     : {reason}")
    print(bar)


# ── Orchestrator ───────────────────────────────────────────────────────────────

class SAGEOrchestrator(AutoGenMAS):
    """
    SAGE-specific orchestrator.
    Adds a simplification loop on top of the standard sequential pipeline.
    """

    def __init__(self, agents: Dict[str, Agent], max_steps: int = 40, architecture: str="sequential",coordinator_role: Optional[str] = None,):
        super().__init__(
            agents=agents,
            architecture=architecture,
            coordinator_role=coordinator_role,
        )
        self.max_steps = max_steps

    # ── Main entry ─────────────────────────────────────────────────────────────
    def _run_sequential(self, problem: str, context: Optional[str] = None):
        """
        problem: the raw expression string to simplify.
        """
        original_expression = problem
        current_expression = problem
        step_counter = 0
        intermediate_expressions: list[str] = []
        outputs: Dict[str, AgentOutput] = {}

        # ── Phase 1: Parse ─────────────────────────────────────────────────────
        step_counter += 1
        parser_agent = self.agents["expression_parser"]
        parser_input = parser_agent.build_input(
            problem=current_expression,
            previous_outputs={},
            context="Parse the expression into a tree. Do not simplify.",
        )
        parser_output = parser_agent.execute(parser_input)
        outputs["expression_parser"] = parser_output

        self.trace.append({
            "agent": "expression_parser",
            "phase": "parsing",
            "expression": current_expression,
            "output": parser_output.to_dict(),
            "call_statistic": parser_agent.last_call_statistic.to_dict(),
        })
        _print_step(step_counter, "expression_parser", parser_output)

        # ── Phase 2: Simplification loop ───────────────────────────────────────
        loop_count = 0
        last_transformer_output: Optional[AgentOutput] = None

        while loop_count < self.max_steps:
            loop_count += 1

            # ── Rule Selector ───────────────────────────────────────────────────
            step_counter += 1
            selector_agent = self.agents["rule_selector"]

            selector_problem = (
                f"Current expression: {current_expression}\n\n"
                f"Parsed tree from expression_parser:\n"
                f"{json.dumps(parser_output.content.get('tree', {}), indent=2)}"
            )
            selector_prev: Dict[str, AgentOutput] = {}
            if last_transformer_output:
                selector_prev["transformer"] = last_transformer_output

            selector_input = selector_agent.build_input(
                problem=selector_problem,
                previous_outputs=selector_prev,
                context="Select one rule to apply next, or NONE if no rule applies.",
            )
            selector_output = selector_agent.execute(selector_input)
            outputs[f"rule_selector_loop{loop_count}"] = selector_output

            self.trace.append({
                "agent": "rule_selector",
                "phase": f"loop_{loop_count}",
                "expression": current_expression,
                "output": selector_output.to_dict(),
                "call_statistic": selector_agent.last_call_statistic.to_dict(),
            })
            _print_step(step_counter, f"rule_selector (loop {loop_count})", selector_output)

            selected_rule = selector_output.content.get("selected_rule", "NONE")
            if selected_rule == "NONE" or not selected_rule:
                print(f"\n  ✓ No more rules apply after {loop_count - 1} loop(s). Exiting loop.")
                break

            # ── Transformer ─────────────────────────────────────────────────────
            step_counter += 1
            transformer_agent = self.agents["transformer"]

            transformer_problem = (
                f"Current expression: {current_expression}\n\n"
                f"Rule to apply:\n"
                f"  selected_rule   : {selected_rule}\n"
                f"  target_node     : {selector_output.content.get('target_node')}\n"
                f"  expected_result : {selector_output.content.get('expected_result')}"
            )
            transformer_input = transformer_agent.build_input(
                problem=transformer_problem,
                previous_outputs={"rule_selector": selector_output},
                context="Apply exactly the selected rule at the target node only.",
            )
            transformer_output = transformer_agent.execute(transformer_input)
            outputs[f"transformer_loop{loop_count}"] = transformer_output
            last_transformer_output = transformer_output

            self.trace.append({
                "agent": "transformer",
                "phase": f"loop_{loop_count}",
                "expression": current_expression,
                "output": transformer_output.to_dict(),
                "call_statistic": transformer_agent.last_call_statistic.to_dict(),
            })
            _print_step(step_counter, f"transformer (loop {loop_count})", transformer_output)

            # Check for transformer error
            if transformer_output.content.get("error"):
                print(f"\n  ✗ Transformer error — stopping loop.")
                break

            new_expr = transformer_output.content.get("result_expression")
            if not new_expr:
                print(f"\n  ✗ Transformer returned no expression — stopping loop.")
                break

            intermediate_expressions.append(new_expr)
            current_expression = new_expr

            # Update parser tree for next iteration (re-parse simplified expression)
            re_parse_input = parser_agent.build_input(
                problem=current_expression,
                previous_outputs={},
                context="Re-parse the updated expression. Do not simplify.",
            )
            parser_output = parser_agent.execute(re_parse_input)

        else:
            print(f"\n  ⚠ Reached max_steps ({self.max_steps}). Stopping loop.")

        # ── Phase 3: Equivalence Check ─────────────────────────────────────────
        step_counter += 1
        checker_agent = self.agents["equivalence_checker"]

        checker_problem = (
            f"original_expression  : {original_expression}\n"
            f"simplified_expression: {current_expression}"
        )
        checker_input = checker_agent.build_input(
            problem=checker_problem,
            previous_outputs={},
            context="Verify that the two expressions are mathematically equivalent.",
        )
        checker_output = checker_agent.execute(checker_input)
        outputs["equivalence_checker"] = checker_output

        self.trace.append({
            "agent": "equivalence_checker",
            "phase": "verification",
            "original_expression": original_expression,
            "final_expression": current_expression,
            "output": checker_output.to_dict(),
            "call_statistic": checker_agent.last_call_statistic.to_dict(),
        })
        _print_step(step_counter, "equivalence_checker", checker_output)

        # ── Summary ────────────────────────────────────────────────────────────
        verdict = checker_output.content.get("verdict", "UNKNOWN")
        reason = checker_output.content.get("reason", "")
        _print_summary(
            original=original_expression,
            steps=intermediate_expressions,
            final=current_expression,
            verdict=verdict,
            reason=reason,
        )

        return outputs
