"""
sage_orchestrator.py — SAGE-specific MAS Orchestrator

Inherits AutoGenMAS and overrides _run_sequential to implement
the symbolic simplification loop.

Changes from original:
  - Prompts built via sage_prompts.format_input() / get_system()
  - Agent outputs (JSON) parsed via sage_protocol.*Output.parse()
    which handles <think> tags, fenced blocks, trailing commas
  - Trace entries include parsed protocol objects for sage_tracker
  - Context chain forwarded to each agent (supports FM-1.4 detection)
  - System prompt injected per-agent via get_system()

Pipeline (unchanged):
  expression_parser (×1)
    └─> loop (max_steps):
          rule_selector  → NONE → break
          transformer    → update current_expression + re-parse
    └─> equivalence_checker (×1)
"""

from __future__ import annotations

import json
from typing import Dict, Optional

from mas_framework.mas import AutoGenMAS, Agent, AgentOutput

from sage_anomaly import AnomalyInjector, AnomalyConfig, AnomalyRecord
from sage_prompts import get_system, format_input


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _bar(char: str = "─", width: int = 64) -> str:
    return char * width


def _print_step(step: int, role: str, output: AgentOutput) -> None:
    print(f"\n{_bar()}")
    print(f"  STEP {step:>2} │ {role.upper()}")
    print(_bar())
    content = output.content
    if isinstance(content, dict) and content.get("error"):
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
    print(f"\n{_bar('═')}")
    print("  SAGE SIMPLIFICATION SUMMARY")
    print(_bar("═"))
    print(f"  Original : {original}")
    for i, expr in enumerate(steps, 1):
        print(f"  Step {i:>2}  : {expr}")
    print(f"  Final    : {final}")
    print(f"  Verdict  : {verdict}")
    print(f"  Reason   : {reason}")
    print(_bar("═"))


# ---------------------------------------------------------------------------
# Context chain builder
# ---------------------------------------------------------------------------

def _build_context(intermediate_expressions: list[str]) -> Optional[str]:
    if not intermediate_expressions:
        return None
    lines = ["Prior simplification steps:"]
    for i, expr in enumerate(intermediate_expressions, 1):
        lines.append(f"  step {i}: {expr}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SAGEOrchestrator
# ---------------------------------------------------------------------------

class SAGEOrchestrator(AutoGenMAS):
    """
    SAGE-specific orchestrator.

    Agent outputs are JSON, parsed by sage_protocol.*Output.parse().
    The parsed dataclass is stored in trace["parsed"] for sage_tracker.
    """

    def __init__(
        self,
        agents: Dict[str, Agent],
        max_steps: int = 40,
        architecture: str = "sequential",
        coordinator_role: Optional[str] = None,
    ):
        super().__init__(
            agents=agents,
            architecture=architecture,
            coordinator_role=coordinator_role,
        )
        self.max_steps = max_steps
        self._anomaly_injector: Optional[AnomalyInjector] = None
        self._anomaly_config: Optional[AnomalyConfig] = None
        self.anomaly_record: Optional[AnomalyRecord] = None

    # -----------------------------------------------------------------------
    # Anomaly injection
    # -----------------------------------------------------------------------

    def set_anomaly(
        self,
        injector: Optional[AnomalyInjector],
        config: Optional[AnomalyConfig],
    ) -> None:
        """Arm anomaly injection for the next _run_sequential call."""
        self._anomaly_injector = injector
        self._anomaly_config = config
        self.anomaly_record = None

    def _maybe_inject(
        self,
        agent_name: str,
        output: AgentOutput,
        step: int,
    ) -> AgentOutput:
        """If anomaly is armed for this agent+step, corrupt output.content."""
        self._last_injection = None
        cfg = self._anomaly_config
        if cfg is None or self._anomaly_injector is None:
            return output
        if cfg.target_agent != agent_name or cfg.injection_step != step:
            return output

        # Gather previous transformer content for FM-1.3
        previous = None
        if cfg.anomaly_type == "FM-1.3":
            previous = self._prev_transformer_content

        corrupted_content, record = self._anomaly_injector.inject(
            anomaly_type=cfg.anomaly_type,
            agent_output=output.content,
            previous_output=previous,
            injection_step=step,
        )
        self.anomaly_record = record
        self._last_injection = {
            "anomaly_type": cfg.anomaly_type,
            "description": record.description,
            "original_content": record.original_output,
            "corrupted_content": record.corrupted_output,
        }

        return AgentOutput(
            agent_role=output.agent_role,
            content=corrupted_content,
            confidence=output.confidence,
            reasoning=output.reasoning,
        )

    def _finalize_trace_entry(self, trace_entry: dict) -> None:
        """If injection happened, restore original content and annotate."""
        if self._last_injection is None:
            return
        inj = self._last_injection
        # Keep original agent output in trace
        trace_entry["output"]["content"] = inj["original_content"]
        # Add injection annotation
        trace_entry["injected_anomaly"] = {
            "anomaly_type": inj["anomaly_type"],
            "description": inj["description"],
            "corrupted_content": inj["corrupted_content"],
        }

    # -----------------------------------------------------------------------
    # Main entry
    # -----------------------------------------------------------------------

    def _run_sequential(self, problem: str, context: Optional[str] = None):
        """
        problem : raw expression string to simplify
        """
        original_expression  = problem
        current_expression   = problem
        step_counter         = 0
        intermediate_exprs: list[str] = []
        outputs: Dict[str, AgentOutput] = {}
        self._prev_transformer_content: Optional[dict] = None

        # ── Phase 1: Parse ─────────────────────────────────────────────────

        step_counter += 1
        parser_agent = self.agents["expression_parser"]

        parser_input = parser_agent.build_input(
            problem=format_input("expression_parser", expression=current_expression),
            previous_outputs={},
            context=get_system("expression_parser"),
        )
        parser_output = parser_agent.execute(parser_input)
        parser_output = self._maybe_inject("parser", parser_output, 0)
        outputs["expression_parser"] = parser_output

        self.trace.append({
            "agent":      "expression_parser",
            "phase":      "parsing",
            "expression": current_expression,
            "output":     parser_output.to_dict(),
            "call_statistic": parser_agent.last_call_statistic.to_dict(),
        })
        self._finalize_trace_entry(self.trace[-1])
        _print_step(step_counter, "expression_parser", parser_output)

        if parser_output.content.get("error"):
            print("  ✗ Parser returned ERROR — aborting.")
            return outputs

        # ── Phase 2: Simplification loop ───────────────────────────────────

        loop_count = 0
        last_transformer_output: Optional[AgentOutput] = None

        while loop_count < self.max_steps:
            loop_count += 1
            ctx = _build_context(intermediate_exprs)

            # ── Rule Selector ─────────────────────────────────────────────

            step_counter += 1
            selector_agent = self.agents["rule_selector"]

            # Pass parser output as JSON string in the user turn
            parser_json_str = json.dumps(parser_output.content, ensure_ascii=False)
            selector_problem = format_input(
                "rule_selector",
                expression=current_expression,
                parser_output=parser_json_str,
                context=ctx,
            )

            selector_prev: Dict[str, AgentOutput] = {}
            if last_transformer_output:
                selector_prev["transformer"] = last_transformer_output

            selector_input = selector_agent.build_input(
                problem=selector_problem,
                previous_outputs=selector_prev,
                context=get_system("rule_selector"),
            )
            selector_output = selector_agent.execute(selector_input)
            selector_output = self._maybe_inject("rule_selector", selector_output, loop_count - 1)
            outputs[f"rule_selector_loop{loop_count}"] = selector_output

            self.trace.append({
                "agent":      "rule_selector",
                "phase":      f"loop_{loop_count}",
                "expression": current_expression,
                "output":     selector_output.to_dict(),
                "call_statistic": selector_agent.last_call_statistic.to_dict(),
            })
            self._finalize_trace_entry(self.trace[-1])
            _print_step(step_counter, f"rule_selector (loop {loop_count})", selector_output)

            selected_rule = selector_output.content.get("selected_rule", "NONE")
            if not selected_rule or selected_rule == "NONE":
                print(f"\n  ✓ No more rules apply after {loop_count - 1} iteration(s).")
                break

            # ── Transformer ───────────────────────────────────────────────

            step_counter += 1
            transformer_agent = self.agents["transformer"]

            transformer_problem = format_input(
                "transformer",
                expression=current_expression,
                rule_selector_output=json.dumps(selector_output.content, ensure_ascii=False),
                context=ctx,
            )

            transformer_input = transformer_agent.build_input(
                problem=transformer_problem,
                previous_outputs={"rule_selector": selector_output},
                context=get_system("transformer"),
            )
            transformer_output = transformer_agent.execute(transformer_input)
            transformer_output = self._maybe_inject("transformer", transformer_output, loop_count - 1)
            outputs[f"transformer_loop{loop_count}"] = transformer_output
            self._prev_transformer_content = last_transformer_output.content if last_transformer_output else None
            last_transformer_output = transformer_output

            self.trace.append({
                "agent":      "transformer",
                "phase":      f"loop_{loop_count}",
                "expression": current_expression,
                "output":     transformer_output.to_dict(),
                "call_statistic": transformer_agent.last_call_statistic.to_dict(),
            })
            self._finalize_trace_entry(self.trace[-1])
            _print_step(step_counter, f"transformer (loop {loop_count})", transformer_output)

            if transformer_output.content.get("error"):
                print("  ✗ Transformer error — stopping loop.")
                break

            new_expr = transformer_output.content.get("result_expression")
            if not new_expr:
                print("  ✗ Transformer returned no expression — stopping loop.")
                break

            intermediate_exprs.append(new_expr)
            current_expression = new_expr

            # Re-parse for next iteration
            reparse_input = parser_agent.build_input(
                problem=format_input("expression_parser",
                                      expression=current_expression,
                                      context=ctx),
                previous_outputs={},
                context=get_system("expression_parser"),
            )
            parser_output = parser_agent.execute(reparse_input)

        else:
            print(f"\n  ⚠ Reached max_steps ({self.max_steps}).")

        # ── Phase 3: Equivalence Check ─────────────────────────────────────

        step_counter += 1
        checker_agent = self.agents["equivalence_checker"]

        checker_problem = format_input(
            "equivalence_checker",
            original_expression=original_expression,
            simplified_expression=current_expression,
            context=_build_context(intermediate_exprs),
        )

        checker_input = checker_agent.build_input(
            problem=checker_problem,
            previous_outputs={},
            context=get_system("equivalence_checker"),
        )
        checker_output = checker_agent.execute(checker_input)
        checker_output = self._maybe_inject("checker", checker_output, 0)
        outputs["equivalence_checker"] = checker_output

        self.trace.append({
            "agent":               "equivalence_checker",
            "phase":               "verification",
            "original_expression": original_expression,
            "final_expression":    current_expression,
            "output":              checker_output.to_dict(),
            "call_statistic":      checker_agent.last_call_statistic.to_dict(),
        })
        self._finalize_trace_entry(self.trace[-1])
        _print_step(step_counter, "equivalence_checker", checker_output)

        # ── Summary ────────────────────────────────────────────────────────

        verdict = checker_output.content.get("verdict", "UNKNOWN")
        reason  = checker_output.content.get("reason", "")
        _print_summary(
            original=original_expression,
            steps=intermediate_exprs,
            final=current_expression,
            verdict=verdict,
            reason=reason,
        )

        return outputs