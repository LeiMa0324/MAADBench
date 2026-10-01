"""
mas_planner.py — EscapeRoom MAS Orchestrator with Planning

Architecture
============
Extends the base orchestrator with a planning layer:

  Action  = defines WHAT to do (prompt, context, output schema)
  Agent   = defines WHO does it (LLM call, role prompt)
  Orchestrator = defines the FLOW (which actions in which order)

Pipeline per iteration (Sequential architecture):
  SELECT_PUZZLE        (planner)     → choose which puzzle to solve next
  OBSERVE_PLANNED_CLUE (observer)    → find the clue for the chosen puzzle
  SOLVE_CLUE           (clue_solver) → solve math problem
  OBSERVE_ITEM         (observer)    → identify real item
  APPLY_DELTA          (item_manager)→ compute new reading, submit answer
  OBSERVE_PUZZLE       (observer)    → check if puzzle was solved

Repeats until all puzzles solved, deadend, or failure.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from benchmark_core.framework import MASOrchestrator, Agent, AgentOutput
from benchmark_core.room_planner import PlanningRoom, _item_name
from benchmark_core.action_prompts import (
    Action,
    ActionResult,
    ACTION_REGISTRY,
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
# Action pipeline
# ---------------------------------------------------------------------------

SELECT_PUZZLE = ACTION_REGISTRY["SELECT_PUZZLE"]


# ---------------------------------------------------------------------------
# ERPlannerOrchestrator
# ---------------------------------------------------------------------------

class ERPlannerOrchestrator(MASOrchestrator):
    """
    EscapeRoom MAS orchestrator with planning.

    Each iteration:
      1. Planner selects a puzzle (SELECT_PUZZLE — always first)
      2. Pipeline runs for that puzzle (remaining actions from config)
      3. Room state updates (unlock/lock items via DAG effects)

    Repeats until completed, deadend, or failure.
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
        self.room: Optional[PlanningRoom] = None
        self.actions = actions
        self.stop_on_fail = stop_on_fail
        self.unit_tools = unit_tools
        self.max_unit_tool_calls = max(0, int(max_unit_tool_calls))

        if fm_id and injection_type:
            self.injection_ctrl = InjectionController.from_fm(fm_id, injection_type)
            self.mode = "structured"
        else:
            self.injection_ctrl = InjectionController.none()
            self.mode = mode

    # -------------------------------------------------------------------
    # run()
    # -------------------------------------------------------------------

    def run(self, room: Optional[PlanningRoom] = None, problem: str = "", context: Optional[str] = None):
        """Run the planning escape room and return a summary dict."""
        if room is not None:
            self.room = room
        if self.room is None:
            raise RuntimeError("Pass room to run() or set self.room before calling")
        self.trace = []
        self._run_sequential(problem, context)
        assign_action_ids(self.trace, self.room.room_id)
        return {
            "room":     self.room.to_dict(),
            "difficulty": self.room.difficulty,
            "escaped":    self.room.completed,
            "puzzles_solved": len(self.room.solved_ids),
            "puzzles_total":  len(self.room.puzzles),
            "trace":    self.trace,
        }

    # -------------------------------------------------------------------
    # Core: execute a single action
    # -------------------------------------------------------------------

    def _execute_action(
        self,
        action:        Action,
        context:       Dict[str, Any],
        prev_messages: Dict[str, str],
        prompt_suffix: str = "",
    ) -> ActionResult:
        """
        Build prompt, call the responsible agent, parse and validate output.
        """
        agent: Agent = self.agents[action.agent_role]

        prompt = action.build_prompt(
            context=context,
            prev_messages=prev_messages,
        )
        if prompt_suffix:
            prompt += prompt_suffix

        try:
            tool_calls = []
            if action.name == "APPLY_DELTA" and self.unit_tools:
                agent_output, tool_calls, _llm_calls, prompt = run_tool_loop(
                    agent, prompt, self.max_unit_tool_calls
                )
            else:
                agent_output = agent.execute(
                    agent.build_input(problem=prompt, previous_outputs={})
                )
            structured, message, raw = agent_output.parse_action(action.name)
            confidence = agent_output.confidence

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            print(f"  ⚠  LLM ERROR in {action.name}: {error_msg}")
            structured = {"error": error_msg}
            message = f"LLM call failed: {error_msg}"
            raw = ""
            confidence = 0.0

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

        call_stat = agent.last_call_statistic.to_dict() if agent._llm.call_statistics else {}
        trace_entry = {
            "puzzle_id":      "",
            "action_id":      "",
            "action":         action.name,
            "agent":          action.agent_role,
            "input":          prompt,
            "output_message":    message,
            "output_structured": structured,
            "schema_errors":  schema_errors,
            "confidence":     confidence,
            "call_statistic": call_stat,
        }
        # Temporary parent metadata used to verify and materialize tool calls.
        # It is removed after the failure report is built.
        if action.name == "APPLY_DELTA" and self.unit_tools:
            trace_entry["tool_calls"] = tool_calls
        self.trace.append(trace_entry)

        return result

    # -------------------------------------------------------------------
    # Sequential pipeline with planning
    # -------------------------------------------------------------------

    def _run_sequential(self, problem: str = "", context: Optional[str] = None):
        """
        Run the planning escape room pipeline.

        Each iteration:
          Step 0: SELECT_PUZZLE        — planner chooses puzzle
          Step 1: OBSERVE_PLANNED_CLUE — observer finds clue for chosen puzzle
          Step 2: SOLVE_CLUE           — clue_solver solves the math
          Step 3: OBSERVE_ITEM         — observer identifies the real item
          Step 4: APPLY_DELTA          — item_manager computes new reading
          Step 5: OBSERVE_PUZZLE       — observer verifies puzzle solved

        Repeats until all puzzles solved, deadend, or failure.
        """
        step = 0
        iteration = 0
        max_iterations = len(self.room.puzzles) + 2
        all_results: Dict[str, Dict[str, ActionResult]] = {}
        ctrl = self.injection_ctrl

        print(f"\n  [MODE] {ctrl.description()}" if ctrl.active
              else f"\n  [MODE] {self.mode}")

        while not self.room.completed and iteration < max_iterations:
            iteration += 1

            # Check for deadend
            if self.room.leads_to_deadend:
                print(f"\n  ⚠ DEADEND — no puzzles available. Room cannot be escaped.")
                break

            print(f"\n\033[1;35m{_bar('━')}")
            print(f"  {self.room.room_id} │ Puzzle {iteration}/{len(self.room.puzzles)}")
            print(f"{_bar('━')}\033[0m")

            # ── Planning description ──────────────────────────────────
            planning_desc = self.room.planning_desc()
            print(f"\n{_bar()}")
            print(f"  Planning Description")
            print(_bar())
            print(planning_desc)
            print(_bar())

            # ── Step 0: SELECT_PUZZLE ──────────────────────────────────
            step += 1
            ctx_planning = {"planning_desc": planning_desc}

            select_result = self._execute_action(
                action=SELECT_PUZZLE,
                context=ctx_planning,
                prev_messages={},
            )
            _print_action_result(step, select_result)

            chosen_id = select_result.structured_output.get("puzzle_id", "")

            # Snapshot state at selection time (before solve)
            avail_at_select = [p.puzzle_id for p in self.room.available_puzzles()]
            solved_at_select = list(self.room.solved_ids)

            # Tag trace entry with the chosen puzzle_id
            self.trace[-1]["puzzle_id"] = chosen_id
            self.trace[-1]["action_id"] = (
                f"{self.room.room_id}_{chosen_id}_SELECT_PUZZLE"
            )
            self.trace[-1]["avail_at_select"] = avail_at_select
            self.trace[-1]["solved_at_select"] = solved_at_select
            puzzle = self.room.get_puzzle(chosen_id)

            if puzzle is None:
                print(f"\n  ⚠ Puzzle '{chosen_id}' not found or invalid — stopping.")
                break

            # Check puzzle is actually available
            avail_ids = [p.puzzle_id for p in self.room.available_puzzles()]
            if chosen_id not in avail_ids:
                print(f"\n  ⚠ Puzzle '{chosen_id}' is LOCKED — stopping.")
                break

            print(f"\n  Selected: {chosen_id} (item={_item_name(puzzle)})")

            # ── Steps 1-5: Pipeline for chosen puzzle ─────────────────
            ctx = {"room_desc": self.room.desc()}
            prev_messages: Dict[str, str] = {}
            action_results: Dict[str, ActionResult] = {}

            # Pass SELECT_PUZZLE output to OBSERVE_PLANNED_CLUE
            prev_messages[SELECT_PUZZLE.name] = json.dumps(
                {**select_result.structured_output, "reasoning": select_result.message},
                ensure_ascii=False,
            )
            action_results[SELECT_PUZZLE.name] = select_result

            print(f"\n{_bar()}")
            print(f"  Room Description")
            print(_bar())
            print(self.room.desc())
            print(_bar())

            eval_result = None

            for action in self.actions:
                step += 1

                # ── prepare: prompt injection (before Agent action) ───
                suffix, forged_msgs = ctrl.prepare(
                    action_name    = action.name,
                    prev_messages  = prev_messages,
                    action_results = action_results,
                    puzzle         = puzzle,
                )
                effective_prev = forged_msgs if forged_msgs is not None else prev_messages

                result = self._execute_action(
                    action=action,
                    context=ctx,
                    prev_messages=effective_prev,
                    prompt_suffix=suffix,
                )

                self.trace[-1]["puzzle_id"] = chosen_id
                self.trace[-1]["action_id"] = (
                    f"{self.room.room_id}_{chosen_id}_{action.name}"
                )
                self.trace[-1]["clue_domain"] = getattr(puzzle.clue, "domain", "gsm-hard")
                self.trace[-1]["clue_problem_id"] = getattr(puzzle.clue, "problem_id", None)
                if suffix or forged_msgs is not None:
                    self.trace[-1]["prompt_injection"] = ctrl.history[-1].to_dict()

                # ── post: output injection (after Agent action) ───────
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

                _print_action_result(step, result)

                msg_json = json.dumps(
                    {**result.structured_output, "reasoning": result.message},
                    ensure_ascii=False,
                )
                prev_messages[action.name] = msg_json
                action_results[action.name] = result

                # Alias: OBSERVE_PLANNED_CLUE → OBSERVE_CLUE so downstream
                # actions (SOLVE_CLUE, OBSERVE_ITEM) can find the output
                if action.name == "OBSERVE_PLANNED_CLUE":
                    prev_messages["OBSERVE_CLUE"] = msg_json
                    action_results["OBSERVE_CLUE"] = result

                # ── Prompt injection verify ────────────────────────────
                inj_verify = ctrl.verify(action_results, puzzle)
                if inj_verify is not None:
                    self.trace[-1]["prompt_injection_induced"] = inj_verify

                # ── Submit answer if action declares it ──
                if action.submits_answer:
                    answer = result.structured_output.get("answer")
                    if answer is not None:
                        eval_result = self.room.update_item(chosen_id, answer)
                        self.trace[-1]["eval_result"] = eval_result
                        # Refresh context for subsequent actions
                        ctx = {"room_desc": self.room.desc()}

            all_results[f"iter_{iteration}"] = action_results

            # ── Verification ──────────────────────────────────────────
            all_actions = [SELECT_PUZZLE] + self.actions
            PuzzleVerifier.verify(
                puzzle_index=iteration,
                results=action_results,
                room=self.room,
                eval_result=eval_result,
                trace=self.trace,
                actions=all_actions,
                puzzle=puzzle,
            )

            # ── Determine puzzle outcome ──────────────────────────────
            actual_solved = (eval_result == "correct")

            if actual_solved:
                print(f"\n  [Iteration {iteration}] {chosen_id} SOLVED!")
                print(f"  Item states: {[k for k, v in self.room.item_states.items() if v]}")
            else:
                print(f"\n  [Iteration {iteration}] {chosen_id} FAILED (eval={eval_result})")
                if self.stop_on_fail:
                    break

        # ── Final summary ─────────────────────────────────────────────
        print(f"\n{_bar('═')}")
        if self.room.completed:
            print(f"  ✓ ESCAPED! All {len(self.room.puzzles)} puzzles solved.")
        elif self.room.leads_to_deadend:
            print(f"  ✗ DEADEND after solving {len(self.room.solved_ids)}/{len(self.room.puzzles)} puzzles.")
        else:
            print(f"  ✗ FAILED. Solved {len(self.room.solved_ids)}/{len(self.room.puzzles)} puzzles.")
        print(_bar('═'))

        return all_results

    def get_failure_report(self) -> Dict[str, Any]:
        """Collect failure info from trace."""
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
    ) -> "ERPlannerOrchestrator":
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
                "OBSERVE_PLANNED_CLUE", "SOLVE_CLUE", "OBSERVE_ITEM",
                "APPLY_DELTA", "OBSERVE_PUZZLE",
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
    difficulty = sys.argv[2] if len(sys.argv) > 2 else "medium"
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42

    orch = ERPlannerOrchestrator.from_config(config_path)
    orch.room = PlanningRoom.create(difficulty=difficulty, room_id="room_0000", seed=seed)
    result = orch.run()
    print(f"\n{'='*64}")
    print(f"  RESULT: {'ESCAPED!' if result['escaped'] else 'STUCK'}")
    print(f"  Puzzles solved: {result['puzzles_solved']}/{result['puzzles_total']}")
    print(f"  Difficulty: {result['difficulty']}")
    print(f"{'='*64}")
