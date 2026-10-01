# EscapeRoom Action → MAST Failure Mode Mapping

This document defines how each action-level failure in the EscapeRoom pipeline maps to failure modes across 5 failure categories: MAST original (FC1–FC3) and our extensions (FC4–FC5).

## Taxonomy

### Reference: Full MAST Taxonomy (Cemri et al., 2025)

For context, MAST identifies 14 failure modes across 3 categories, based on Grounded Theory analysis of 1,600+ traces from 7 MAS frameworks (Cemri et al., "Why Do Multi-Agent LLM Systems Fail?", NeurIPS 2025 D&B; arXiv:2503.13657). The full taxonomy is reproduced below as a reference; the subsequent sections describe which subset our EscapeRoom pipeline exercises and how our extensions (FC4, FC5) relate.

| FC | FM | Name | Description | Freq. |
|----|-----|------|-------------|-------|
| FC1: System Design | FM-1.1 | Disobey task specification | Failure to adhere to specified constraints or output requirements of a given task. | 11.8% |
| FC1: System Design | FM-1.2 | Disobey role specification | Failure to adhere to the defined responsibilities and constraints of an assigned role; an agent may behave like another. | 1.5% |
| FC1: System Design | FM-1.3 | Step repetition | Unnecessary reiteration of previously completed steps. | 15.7% |
| FC1: System Design | FM-1.4 | Loss of conversation history | Unexpected truncation or loss of prior interaction context, causing agents to act on stale or missing information. | 2.8% |
| FC1: System Design | FM-1.5 | Unaware of termination conditions | Failure to recognize that task-completion criteria have been met, leading to unnecessary continuation. | 12.4% |
| FC2: Inter-Agent Misalignment | FM-2.1 | Conversation reset | Unexpected or unwarranted restart of a dialogue, losing accumulated progress. | — |
| FC2: Inter-Agent Misalignment | FM-2.2 | Fail to ask for clarification | Failure to request clarification when facing ambiguous or incomplete information. | — |
| FC2: Inter-Agent Misalignment | FM-2.3 | Task derailment | Deviation from a viable path toward the objective. | — |
| FC2: Inter-Agent Misalignment | FM-2.4 | Information withholding | Failure to share critical information with agents that need it. | — |
| FC2: Inter-Agent Misalignment | FM-2.5 | Ignored other agent's input | Failure to use information provided by the system or upstream agents. | — |
| FC2: Inter-Agent Misalignment | FM-2.6 | Reasoning-action mismatch | Agent's reasoning arrives at a result but its output differs. | — |
| FC3: Task Verification | FM-3.1 | Premature termination | Ending the task before all required steps have been completed. | 6.2% |
| FC3: Task Verification | FM-3.2 | No or incomplete verification | Failure to fully verify intermediate or final outputs. | 8.2% |
| FC3: Task Verification | FM-3.3 | Incorrect verification | Verification step produces a wrong judgment. | 9.1% |

Category totals: FC1 ≈ 41.8%, FC2 ≈ 36.9%, FC3 ≈ 21.3% of failures in MAST-Data. FC2 sub-mode frequencies are not reported individually in the original paper.

### MAST Original (FC1–FC3)

| FC | FM | Name | Description |
|----|-----|------|-------------|
| FC1: System Design | FM-1.1 | Disobey task specification | Failure to adhere to specified constraints or output requirements |
| FC1: System Design | FM-1.3 | Step repetition | Attempting to repeat a previously completed step |
| FC2: Inter-Agent Misalignment | FM-2.3 | Task derailment | Deviation from a viable path toward the objective |
| FC2: Inter-Agent Misalignment | FM-2.5 | Ignored other agent's input | Failure to use information provided by the system or upstream agents |
| FC2: Inter-Agent Misalignment | FM-2.6 | Reasoning-action mismatch | Agent's reasoning arrives at the correct result but output differs |
| FC3: Task Verification | FM-3.3 | Incorrect verification | Verification step produces a wrong judgment |

### Extended: FC4 (Agent Capability) and FC5 (Error Propagation)

| FC | FM | Name | Description |
|----|-----|------|-------------|
| FC4: Agent Capability | FM-4.1 | Deception susceptibility | Agent fooled by fake items/clues designed to mislead |
| FC4: Agent Capability | FM-4.2 | Reasoning error | Mathematical/logical reasoning produces incorrect result |
| FC5: Error Propagation | FM-4.3 | Error propagation | Downstream failure caused by upstream error; system lacks error recovery |

**Rationale for FC4:** MAST focuses on multi-agent system-level failures (design, coordination, verification). It does not cover single-agent capability failures, which dominate in our benchmark. FM-4.1 and FM-4.2 capture these capability gaps.

**Rationale for FC5:** In a sequential pipeline, upstream errors propagate to downstream agents. FM-4.3 captures the system's lack of error recovery mechanisms — a system-level issue distinct from individual agent capability (FC4).

## Annotation Pipeline

Annotation is done in two stages by `FM_annotate.py`:

### Stage 1: Programmatic (summary.csv only)

Deterministic rules using `action_name` for rows whose `action_status` is `wrong`, together with `desc` and cross-row cascade detection. Only assigns FM when classification is **unambiguous**.

### Stage 2: Trace heuristic (reads trace JSONs)

Extracts numbers/IDs from agent reasoning text (`output_message` field) and compares against ground truth and upstream outputs. Classifies ambiguous cases that Stage 1 cannot determine. Optional LLM-based classification available via `--use-llm`.

## Mapping Table

### Stage 1: Programmatic Rules

| Action | Error (`desc`) | Condition | FM | Annotator |
|--------|---------------|-----------|-----|-----------|
| SELECT_PUZZLE | `already_solved` | — | FM-1.3 | programmatic |
| SELECT_PUZZLE | `puzzle_locked` | — | FM-1.1 | programmatic |
| SELECT_PUZZLE | `leads_to_deadend` | — | FM-2.3 | programmatic |
| SELECT_PUZZLE | `puzzle_not_found` | — | FM-2.6 | programmatic |
| SOLVE_CLUE | any | OBSERVE_CLUE failed on same puzzle | FM-4.3 | programmatic |
| SOLVE_CLUE | `math_unit_wrong` only | value correct | FM-1.1 | programmatic |
| OBSERVE_ITEM | `item_state_wrong` | type correct | FM-2.5 | programmatic |
| APPLY_DELTA | any | upstream OBSERVE_CLUE/SOLVE_CLUE/OBSERVE_ITEM failed | FM-4.3 | programmatic |
| UNIT_TOOL_CALL | `TC_REQUIRED_NOT_CALLED`, `TC_DIMENSION_MISMATCH`, `TC_INVALID_REQUEST` | tool-call protocol or argument constraints violated | FM-1.1 | programmatic |
| UNIT_TOOL_CALL | `TC_WRONG_ARGUMENTS`, `TC_WRONG_TOOL` | wrong conversion inputs or tool selected | FM-2.5 | programmatic |
| UNIT_TOOL_CALL | `TC_MISSING_MESSAGE` only | missing tool-call justification/message | FM-3.2 | programmatic |
| OBSERVE_PUZZLE | `observer_disagrees` | — | FM-3.3 | programmatic |
| API_TIMEOUT | — | — | (empty) | — |

### Stage 2: Trace Heuristic

| Action | Error | Check | FM | Annotator |
|--------|-------|-------|-----|-----------|
| OBSERVE_CLUE | `clue_select_wrong` | correct item_id in reasoning? | Yes → FM-2.6, No → FM-4.1 | programmatic |
| SOLVE_CLUE | `math_value_wrong` | agent unit in problem text? | Yes → FM-1.1 | programmatic |
| SOLVE_CLUE | `math_value_wrong` | correct answer in reasoning? | Yes → FM-2.6, No → FM-4.2 | programmatic |
| OBSERVE_ITEM | `item_type_wrong` | correct type in reasoning? | Yes → FM-2.6, No → FM-4.1 | programmatic |
| APPLY_DELTA | `apply_delta_wrong` (own) | upstream values in reasoning? | Yes → FM-4.2, No → FM-2.5 | programmatic |

**SOLVE_CLUE priority:** For rows with both `math_value_wrong` and `math_unit_wrong`: unit-from-problem check (→ FM-1.1) takes priority over value reasoning check (→ FM-4.2/FM-2.6).

## Error Propagation (FM-4.3)

When an upstream action fails, downstream actions receive wrong inputs. Their failures are labeled FM-4.3 rather than being attributed to the downstream agent.

### Propagation chains:
```
OBSERVE_CLUE ✗ → SOLVE_CLUE  = FM-4.3
OBSERVE_CLUE ✗ → APPLY_DELTA = FM-4.3
SOLVE_CLUE ✗   → APPLY_DELTA = FM-4.3
OBSERVE_ITEM ✗  → APPLY_DELTA = FM-4.3
```

Detection: `failure_index[(room_id, temperature, puzzle_id)]` tracks which actions failed per puzzle. If any upstream action is in the set, downstream failures are FM-4.3.

## Reasoning

### SELECT_PUZZLE

**`already_solved` → FM-1.3 (Step repetition)**
The agent selected a puzzle already marked SOLVED. The planning description re-displays full room state at every step. The agent is attempting to repeat a completed step.

**`puzzle_locked` → FM-1.1 (Disobey task specification)**
The planning description explicitly marks locked puzzles with "LOCKED — cannot interact". Selecting a locked puzzle violates an explicit, visible constraint.

**`leads_to_deadend` → FM-2.3 (Task derailment)**
The chosen puzzle is valid (unlocked, unsolved) — no explicit constraint is violated. But this choice leads to a state from which completion is impossible. The agent has deviated from any viable path toward the objective.

**`puzzle_not_found` → FM-2.6 (Reasoning-action mismatch)**
The agent output a puzzle ID that does not exist (hallucination). The reasoning produced an action with no valid referent.

### OBSERVE_CLUE / OBSERVE_PLANNED_CLUE

**`clue_select_wrong` → FM-4.1 or FM-2.6 (trace heuristic)**
The observer picked a fake clue instead of the real one. If the reasoning text mentions the correct item_id but the output differs → FM-2.6 (reasoning-action mismatch). If the reasoning itself was fooled → FM-4.1 (deception susceptibility).

### SOLVE_CLUE

**Cascade → FM-4.3:** If OBSERVE_CLUE failed on the same puzzle, the solver received a fake clue. Its wrong answer is error propagation.

**`math_unit_wrong` only (value correct) → FM-1.1:** The task requires providing the correct answer with the correct unit. Getting the value right but the unit wrong means the agent partially failed the output specification.

**`math_value_wrong` + `math_unit_wrong` with unit from problem text → FM-1.1:** The agent confused the math problem's internal unit (e.g., "liters", "hectares") with the clue's target delta_unit (e.g., "kelvin", "seconds"). This is a task specification misunderstanding.

**`math_value_wrong` (own, reasoning contains GT answer) → FM-2.6:** The solver's step-by-step reasoning arrived at the correct answer but the structured output contains a different number. The reasoning and action are inconsistent.

**`math_value_wrong` (own, reasoning does NOT contain GT answer) → FM-4.2:** The solver's reasoning itself contains mathematical errors. This is a capability failure, not a system-level issue.

### OBSERVE_ITEM

**`item_type_wrong` → FM-4.1 or FM-2.6 (trace heuristic):**
The observer picked a fake item. If reasoning mentions the correct item type but output differs → FM-2.6. If reasoning was fooled by the fake → FM-4.1 (deception susceptibility).

**`item_state_wrong` (type correct) → FM-2.5 (Ignored input):**
The agent found the correct item (type is right). The state value is written explicitly in the description (e.g., "reads 88 celsius"). Outputting a different state means the agent did not correctly use the information provided to it.

### APPLY_DELTA

**Cascade → FM-4.3:** If any upstream action failed, the item_manager received wrong inputs. Error propagation.

**Own error (upstream values in reasoning) → FM-4.2:** The item_manager used the correct upstream values but made a computation error (e.g., wrong arithmetic). Reasoning error.

**Own error (upstream values NOT in reasoning) → FM-2.5:** The item_manager used different values than upstream agents provided. Ignored input.

### UNIT_TOOL_CALL

**`TC_REQUIRED_NOT_CALLED`, `TC_DIMENSION_MISMATCH`, `TC_INVALID_REQUEST` → FM-1.1:**
The tool-enabled setting specifies when and how the unit-conversion tool should be called. Missing a required conversion, issuing an invalid request, or violating unit/tool compatibility constraints fails to follow the tool-call specification.

**`TC_WRONG_ARGUMENTS`, `TC_WRONG_TOOL` → FM-2.5:**
The tool call should be parameterized by the upstream clue answer, delta unit, instrument state, and instrument unit. Using the wrong value, source/target unit, or conversion tool means the item manager did not correctly use the information provided by upstream actions and the environment.

**`TC_MISSING_MESSAGE` only → FM-3.2:**
The tool protocol requires an auditable message explaining the conversion. When this is the only error, the call is treated as incomplete verification of the tool interaction. If it co-occurs with wrong arguments, the wrong-argument mapping takes priority.

### OBSERVE_PUZZLE

**`observer_disagrees` → FM-3.3 (Incorrect verification)**
The observer must judge whether the puzzle was solved. A wrong judgment is a failure to adequately validate a crucial decision.

## Coverage Summary

| Category | FMs | Description |
|----------|-----|-------------|
| FC1: System Design | FM-1.1, FM-1.3 | Task specification violations |
| FC2: Inter-Agent Misalignment | FM-2.3, FM-2.5, FM-2.6 | Coordination and communication failures |
| FC3: Task Verification | FM-3.2, FM-3.3 | Verification step errors |
| **FC4: Agent Capability (ours)** | **FM-4.1, FM-4.2** | **Individual agent capability failures** |
| **FC5: Error Propagation (ours)** | **FM-4.3** | **Upstream errors cascading to downstream agents** |

MAST FMs not triggered by current pipeline: FM-1.2, FM-1.4, FM-1.5, FM-2.1, FM-2.2, FM-2.4, FM-3.1. These require architectures with richer agent interaction (debate, delegation, role-switching) or different failure patterns.
