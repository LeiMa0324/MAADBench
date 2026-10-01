"""
mas.py — EscapeRoom MAS Orchestrator with verification.

Architecture
============
Actions and agents are fully decoupled:

  Action  = defines WHAT to do (prompt, context, output schema)
  Agent   = defines WHO does it (LLM call, role prompt)
  Orchestrator = defines the FLOW (which actions in which order)

Two communication channels (never mixed):
  structured_output  → program-readable, used for verification & failure capture
  message            → natural-language, passed between agents (lossy channel)

Pipeline per puzzle (Sequential architecture):
  OBSERVE_CLUE   (observer)       → identify real clue
  SOLVE_CLUE     (clue_solver)    → solve math problem
  OBSERVE_ITEM   (observer)       → identify real item
  APPLY_DELTA    (item_manager)   → compute new reading, submit answer

Failure capture
===============
After each action, structured_output is validated against output_schema.
After the room evaluates, ground-truth comparison yields:

  SOLVE_CLUE.answer       vs  clue.answer            → math reasoning failure
  OBSERVE_ITEM.item_state vs  item.state              → item identification failure
  APPLY_DELTA.answer      vs  item._groundtruth_answer → system-level failure

  SOLVE_CLUE correct + APPLY_DELTA wrong              → emergent communication failure
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from benchmark_core.framework import MASOrchestrator, Agent, AgentOutput
from benchmark_core.room_core import Room
from benchmark_core.action_prompts import (
    Action,
    ActionResult,
    actions_from_names,
)

from benchmark_core.injection import InjectionController
from benchmark_core.verifier import PuzzleVerifier
from benchmark_core.unit_tools import (
    build_tool_action_entries,
    diagnose as diagnose_unit_tools,
    run_tool_loop,
)
from benchmark_core.trace_utils import assign_action_ids
# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _bar(char: str = "─", width: int = 64) -> str:
    return char * width


def _print_action_result(step: int, result: ActionResult) -> None:
    print(f"\n{_bar()}")
    print(f"  Action {step:>2} │ Agent: {result.agent_role.upper()} → Action: {result.action_name}")
    print(_bar())
    print("  [output message]")
    print(f"  {result.message}")
    print("  [output structured]")
    print(json.dumps(result.structured_output, indent=4, ensure_ascii=False))
    if result.schema_errors:
        print(f"  ⚠  SCHEMA ERRORS: {result.schema_errors}")
    print(f"  confidence: {result.confidence:.2f}")


# ---------------------------------------------------------------------------
# EROrchestrator
# ---------------------------------------------------------------------------

class EROrchestrator(MASOrchestrator):
    """
    EscapeRoom MAS orchestrator.

    Uses Action objects to build prompts and capture structured outputs.
    Agent communication uses natural-language messages (lossy channel).
    Verification uses structured_output (never shown to agents).
    """

    def __init__(
        self,
        agents: Dict[str, Agent],
        actions: List[Action],
        architecture: str = "sequential",
        coordinator_role: Optional[str] = None,
        mode: str = "natural",
        fm_id: Optional[str] = None,
        injection_type: Optional[str] = None,
        stop_on_fail: bool = True,
        unit_tools: bool = False,
        max_unit_tool_calls: int = 3,
    ):
        super().__init__(
            agents=agents,
            architecture=architecture,
            coordinator_role=coordinator_role,
        )
        self.room: Optional[Room] = None
        self.actions = actions
        self.stop_on_fail = stop_on_fail
        self.unit_tools = unit_tools
        self.max_unit_tool_calls = max(0, int(max_unit_tool_calls))

        # Build injection controller from FM + type
        if fm_id and injection_type:
            self.injection_ctrl = InjectionController.from_fm(fm_id, injection_type)
            self.mode = "structured"  # injection always requires structured mode
        else:
            self.injection_ctrl = InjectionController.none()
            self.mode = mode

    # -------------------------------------------------------------------
    # Override run() — parent's run() expects a different trace format
    # -------------------------------------------------------------------

    def run(self, room: Optional[Room] = None, problem: str = "", context: Optional[str] = None):
        """Run the escape room and return a summary dict."""
        if room is not None:
            self.room = room
        if self.room is None:
            raise RuntimeError("Pass room to run() or set self.room before calling")
        self.trace = []
        self._run_sequential(problem, context)
        assign_action_ids(self.trace, self.room.room_id)
        return {
            "difficulty": self.room.difficulty,
            "escaped": self.room.completed,
            "puzzles_solved": self.room.current_index,
            "puzzles_total": len(self.room.puzzles),
            "trace": self.trace,
        }

    # -------------------------------------------------------------------
    # Core: execute a single action
    # -------------------------------------------------------------------

    def _execute_action(
        self,
        action:        Action,
        context:       Dict[str, Any],
        prev_messages: Dict[str, str],   # action_name → message string
        prompt_suffix: str = "",
    ) -> ActionResult:
        """
        Build prompt, call the responsible agent, parse and validate output.

        Returns ActionResult with:
          structured_output  → parsed from LLM's "structured" key
          message            → parsed from LLM's "message" key
          schema_errors      → validation results
        """
        agent: Agent = self.agents[action.agent_role]

        # Build prompt from action template
        prompt = action.build_prompt(
            context=context,
            prev_messages=prev_messages,
        )
        if prompt_suffix:
            prompt += prompt_suffix

        try:
            # Call LLM directly — action prompt already contains all context
            tool_calls = []
            if action.name == "APPLY_DELTA" and self.unit_tools:
                agent_output, tool_calls, _llm_calls, prompt = run_tool_loop(
                    agent, prompt, self.max_unit_tool_calls
                )
            else:
                agent_output = agent.execute(
                    agent.build_input(problem=prompt, previous_outputs={})
                )

            # Parse response
            structured, message, raw = agent_output.parse_action(action.name)
            confidence = agent_output.confidence

        except Exception as e:
            # LLM call failed (context too long, timeout, API error, etc.)
            error_msg = f"{type(e).__name__}: {e}"
            print(f"  ⚠  LLM ERROR in {action.name}: {error_msg}")
            structured = {"error": error_msg}
            message = f"LLM call failed: {error_msg}"
            raw = ""
            confidence = 0.0

        # Validate structured output against schema
        schema_errors = action.validate_structured(structured)

        result = ActionResult(
            action_name       = action.name,
            agent_role        = action.agent_role,
            structured_output = structured,
            message           = message,
            schema_errors     = schema_errors,
            raw_response      = raw,
            confidence        = confidence,
        )

        # Log to trace (puzzle_id, action_id set by _run_sequential)
        call_stat = agent.last_call_statistic.to_dict() if agent._llm.call_statistics else {}
        trace_entry = {
            "puzzle_id":     "",
            "action_id":     "",
            "action":        action.name,
            "agent":         action.agent_role,
            "input":         prompt,
            "output_message":    message,
            "output_structured": structured,
            "schema_errors": schema_errors,
            "confidence":    confidence,
            "call_statistic": call_stat,
        }
        # Temporary parent metadata used to verify and materialize tool calls.
        # It is removed after the failure report is built.
        if action.name == "APPLY_DELTA" and self.unit_tools:
            trace_entry["tool_calls"] = tool_calls
        self.trace.append(trace_entry)

        return result

    # -------------------------------------------------------------------
    # Sequential pipeline
    # -------------------------------------------------------------------

    def _run_sequential(self, problem: str = "", context: Optional[str] = None):
        """
        Run the escape room pipeline using Action objects.

        For each puzzle:
          Step 1: OBSERVE_CLUE   — observer identifies the real clue
          Step 2: SOLVE_CLUE     — clue_solver solves the math problem
          Step 3: OBSERVE_ITEM   — observer identifies the real item
          Step 4: APPLY_DELTA    — item_manager computes the new reading
          Step 5: OBSERVE_PUZZLE — observer checks if puzzle was solved

        Each puzzle gets up to 2 attempts. The puzzle advances only when both
        the programmatic eval and OBSERVE_PUZZLE agree the puzzle is solved.
        After 2 failed attempts the room is considered failed and the run stops.
        """
        MAX_ATTEMPTS = 1
        step        = 0
        all_results: Dict[str, Dict[str, ActionResult]] = {}
        ctrl = self.injection_ctrl

        print(f"\n  [MODE] {ctrl.description()}" if ctrl.active
              else f"\n  [MODE] {self.mode}")

        while not self.room.completed:
            pi  = self.room.current_index + 1
            puzzle = self.room.puzzles[self.room.current_index]
            room_failed = False

            for attempt in range(1, MAX_ATTEMPTS + 1):
                context = {"room_desc": self.room.desc()}

                prev_messages:  Dict[str, str] = {}
                action_results: Dict[str, ActionResult] = {}

                print(f"\n\033[1;35m{_bar('━')}")
                print(f"  {self.room.room_id} │ Puzzle {pi}/{len(self.room.puzzles)} │ {puzzle.puzzle_id} │ attempt {attempt}/{MAX_ATTEMPTS}")
                print(f"{_bar('━')}\033[0m")
                print(f"\n{_bar()}")
                print(f"  Room Description")
                print(_bar())
                print(self.room.desc())
                print(_bar())

                eval_result = None

                for action in self.actions:
                    step += 1

                    # ── prepare: prompt injection (before Agent action) ───────────
                    suffix, forged_msgs = ctrl.prepare(
                        action_name    = action.name,
                        prev_messages  = prev_messages,
                        action_results = action_results,
                        puzzle         = puzzle,
                    )
                    effective_prev = forged_msgs if forged_msgs is not None else prev_messages

                    result = self._execute_action(
                        action        = action,
                        context       = context,
                        prev_messages = effective_prev,
                        prompt_suffix = suffix,
                    )

                    # Fill in IDs on the trace entry just appended
                    self.trace[-1]["puzzle_id"]  = puzzle.puzzle_id
                    self.trace[-1]["action_id"] = (
                        f"{self.room.room_id}_{puzzle.puzzle_id}_{action.name}"
                    )
                    self.trace[-1]["attempt"]    = attempt
                    self.trace[-1]["clue_domain"] = getattr(puzzle.clue, "domain", "gsm-hard")
                    self.trace[-1]["clue_problem_id"] = getattr(puzzle.clue, "problem_id", None)
                    if suffix or forged_msgs is not None:
                        self.trace[-1]["prompt_injection"] = ctrl.history[-1].to_dict()

                    # ── post: output injection (after Agent action) ───────────────
                    result, inj_record = ctrl.post(action.name, result, puzzle)

                    if inj_record:
                        self.trace[-1]["injection"] = inj_record

                    if action.name == "APPLY_DELTA" and self.unit_tools:
                        try:
                            received = json.loads(effective_prev.get("OBSERVE_ITEM", "{}"))
                        except (TypeError, json.JSONDecodeError):
                            received = {}
                        parent_entry = self.trace[-1]
                        parent_entry["unit_tool_diagnostics"] = diagnose_unit_tools(
                            received=received,
                            final_output=result.structured_output,
                            calls=parent_entry.get("tool_calls", []),
                            gold={
                                "item_type": puzzle.item.item_type.value,
                                "delta": puzzle.clue.answer,
                                "delta_unit": puzzle.clue.delta_unit,
                                "item_state": puzzle.item.state,
                                "item_unit": puzzle.item.unit,
                                "final_answer": puzzle.item._groundtruth_answer(
                                    puzzle.clue.answer, puzzle.clue.delta_unit
                                ),
                            },
                        )
                        tool_entries = build_tool_action_entries(
                            parent_entry, parent_entry["unit_tool_diagnostics"]
                        )
                        self.trace[-1:] = tool_entries + [parent_entry]

                    print(f"\n{_bar()}")
                    print(f"  Room Description")
                    print(_bar())
                    print(self.room.desc())
                    print(_bar())

                    _print_action_result(step, result)

                    # Pass structured output + reasoning as JSON to next agent
                    prev_messages[action.name] = json.dumps(
                        {**result.structured_output, "reasoning": result.message},
                        ensure_ascii=False,
                    )

                    action_results[action.name] = result

                    # ── Prompt injection verify ───────────────────────────
                    inj_verify = ctrl.verify(action_results, puzzle)
                    if inj_verify is not None:
                        self.trace[-1]["prompt_injection_induced"] = inj_verify

                    # ── room.update: submit answer if action declares it ──
                    answer = result.structured_output.get("answer") if action.submits_answer else None
                    step_eval = self.room.update_item(answer)
                    if step_eval is not None:
                        eval_result = step_eval
                        self.trace[-1]["eval_result"] = eval_result
                        # Refresh context for subsequent actions
                        context = {"room_desc": self.room.desc()}

                all_results[f"puzzle_{pi}_attempt_{attempt}"] = action_results

                # ── Puzzle verification ────────────────────────────
                checks = PuzzleVerifier.verify(
                    pi, action_results, self.room, eval_result,
                    self.trace, actions=self.actions,
                )

                # ── Determine puzzle outcome ───────────────────────────
                obs_check = checks.get("OBSERVE_PUZZLE", {})
                actual_solved = obs_check.get("actual_solved", False)
                observer_solved = obs_check.get("observer_solved", False)
                agreement = obs_check.get("agreement", True)

                if not agreement:
                    # Observer disagrees with reality → observer failure, room fails
                    print(f"\n  [Puzzle {pi}] OBSERVER MISMATCH (observer={observer_solved}, actual={actual_solved}) — room failed")
                    room_failed = True
                elif actual_solved and observer_solved:
                    # Both agree solved → next puzzle
                    print(f"\n  [Puzzle {pi}] SOLVED (attempt {attempt})")
                    break
                else:
                    # Both agree not solved → retry or fail
                    if attempt < MAX_ATTEMPTS:
                        print(f"\n  [Puzzle {pi}] NOT SOLVED (attempt {attempt}/{MAX_ATTEMPTS}), retrying...")
                    else:
                        print(f"\n  [Puzzle {pi}] FAILED after {MAX_ATTEMPTS} attempts")
                        room_failed = True

            if room_failed and self.stop_on_fail:
                break

        return all_results

    def get_failure_report(self) -> Dict[str, Any]:
        """Delegate to PuzzleVerifier."""
        return PuzzleVerifier.get_failure_report(self.trace)

    # -------------------------------------------------------------------
    # Config loader
    # -------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config_path: str,
        mode: str = "natural",
        fm_id: Optional[str] = None,
        injection_type: Optional[str] = None,
        actions_override: Optional[List[Action]] = None,
        unit_tools: Optional[bool] = None,
        max_unit_tool_calls: Optional[int] = None,
    ) -> "EROrchestrator":
        """Load from YAML config. Room must be set separately via orch.room = ..."""
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        system_config = config.get("system", {})
        agents_config = config.get("agents", [])
        llm_config = config.get("llm", {})

        agents: Dict[str, Agent] = {}
        for agent_cfg in agents_config:
            agent_llm_config = {**llm_config}
            if "max_tokens" in agent_cfg:
                agent_llm_config["max_tokens"] = agent_cfg["max_tokens"]
            agent = Agent(
                role=agent_cfg["role"],
                role_prompt=agent_cfg.get("prompt"),
                llm_config=agent_llm_config,
            )
            agents[agent_cfg["role"]] = agent

        # Use override if provided, else read from config, else default
        if actions_override is not None:
            actions = actions_override
        else:
            action_names = system_config.get("actions", [
                "OBSERVE_CLUE", "SOLVE_CLUE", "OBSERVE_ITEM", "APPLY_DELTA", "OBSERVE_PUZZLE",
            ])
            actions = actions_from_names(action_names)

        return cls(
            agents=agents,
            actions=actions,
            architecture=system_config.get("architecture", "sequential"),
            mode=mode,
            fm_id=fm_id,
            injection_type=injection_type,
            unit_tools=(system_config.get("unit_tools", False) if unit_tools is None else unit_tools),
            max_unit_tool_calls=(system_config.get("max_unit_tool_calls", 3)
                                 if max_unit_tool_calls is None else max_unit_tool_calls),
        )


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml"
    difficulty = sys.argv[2] if len(sys.argv) > 2 else "easy"
    orch = EROrchestrator.from_config(config_path)
    orch.room = Room.create(difficulty=difficulty)
    result = orch.run()
    print(f"\n{'='*64}")
    print(f"  RESULT: {'ESCAPED!' if result['escaped'] else 'STUCK'}")
    print(f"  Puzzles solved: {result['puzzles_solved']}/{result['puzzles_total']}")
    print(f"  Difficulty: {result['difficulty']}")
    print(f"{'='*64}")
