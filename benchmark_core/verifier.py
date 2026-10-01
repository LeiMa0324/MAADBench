"""
verifier.py — Verification and failure analysis for EscapeRoom puzzles.

PuzzleVerifier compares LLM structured outputs against ground truth,
prints aligned verification tables, tags trace entries, and produces
failure reports.

Verification is data-driven: it iterates over the actions list and
uses per-action verifier functions registered in _ACTION_VERIFIERS.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

from benchmark_core.room_core import Room, Clock
from benchmark_core.action_prompts import Action, ActionResult
from benchmark_core.MAST_annotator.MAST_FM import MAST_TAXONOMY


def _bar(char: str = "─", width: int = 64) -> str:
    return char * width


# ---------------------------------------------------------------------------
# MAST FM classification
# ---------------------------------------------------------------------------
# Maps (action_name, error_keyword) → FM-ID.
# See MAST_FM_MAPPING.md for full reasoning.

_FM_RULES: List[Tuple[str, str, str]] = [
    # (action_name, desc_keyword, fm_id)
    # SELECT_PUZZLE
    ("SELECT_PUZZLE", "already_solved",    "FM-1.3"),
    ("SELECT_PUZZLE", "puzzle_locked",     "FM-1.1"),
    ("SELECT_PUZZLE", "leads_to_deadend",  "FM-2.3"),
    ("SELECT_PUZZLE", "puzzle_not_found",  "FM-2.6"),
    # OBSERVE_CLUE / OBSERVE_PLANNED_CLUE
    ("OBSERVE_CLUE",         "clue_select_wrong", "FM-1.1"),
    ("OBSERVE_PLANNED_CLUE", "clue_select_wrong", "FM-1.1"),
    # SOLVE_CLUE
    ("SOLVE_CLUE", "math_value_wrong", "FM-1.1"),
    ("SOLVE_CLUE", "math_unit_wrong",  "FM-2.5"),  # only when value is correct
    # OBSERVE_ITEM
    ("OBSERVE_ITEM", "item_type_wrong",  "FM-1.1"),
    ("OBSERVE_ITEM", "item_state_wrong", "FM-3.2"),
    # APPLY_DELTA
    ("APPLY_DELTA", "apply_delta_wrong", "FM-2.5"),  # default; overridden to None if inherited
    # OBSERVE_PUZZLE
    ("OBSERVE_PUZZLE", "observer_disagrees", "FM-3.3"),
]


def classify_unit_tool_call(error_type: str = "", desc: str = "") -> Optional[str]:
    """Map audited unit-tool failures to MAST failure modes."""
    error_codes = {code.strip() for code in str(error_type or "").split("+") if code.strip()}
    desc = str(desc or "")

    if "TC_WRONG_ARGUMENTS" in error_codes or "TC_WRONG_TOOL" in error_codes:
        return "FM-2.5"
    if error_codes & {
        "TC_REQUIRED_NOT_CALLED",
        "TC_DIMENSION_MISMATCH",
        "TC_INVALID_REQUEST",
        "TC_UNKNOWN_TOOL",
        "TC_UNKNOWN_UNIT",
        "TC_CALL_LIMIT_EXCEEDED",
        "TC_EXECUTION_ERROR",
    }:
        return "FM-1.1"
    if error_codes == {"TC_MISSING_MESSAGE"}:
        return "FM-3.2"

    if any(
        marker in desc
        for marker in ("wrong_tool", "wrong_value", "wrong_from_unit", "wrong_to_unit")
    ):
        return "FM-2.5"
    if any(
        marker in desc
        for marker in (
            "required unit conversion was not called",
            "incompatible",
            "finite number",
        )
    ):
        return "FM-1.1"
    if "missing_tool_message" in desc:
        return "FM-3.2"
    return None


def classify_fm(
    action_name: str,
    checks: Dict[str, Any],
    all_checks: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Optional[str]:
    """Return the MAST FM-ID for a failed action, or None if correct/cascade.

    Parameters
    ----------
    action_name : action that was verified
    checks : the checks dict for this action (must contain "status" and "desc")
    all_checks : all action checks so far (needed to detect cascade for APPLY_DELTA)
    """
    if checks.get("status") == "correct":
        return None

    desc = checks.get("desc", "")

    if action_name == "UNIT_TOOL_CALL":
        return classify_unit_tool_call(checks.get("error_type", ""), desc)

    # SOLVE_CLUE special case: math_unit_wrong alone → FM-2.5,
    # but if math_value_wrong is also present → FM-1.1 (dominant)
    if action_name == "SOLVE_CLUE":
        has_value_err = "math_value_wrong" in desc
        has_unit_err = "math_unit_wrong" in desc
        if has_value_err:
            return "FM-1.1"
        if has_unit_err:
            return "FM-2.5"

    # APPLY_DELTA: own vs inherited
    if action_name == "APPLY_DELTA" and all_checks:
        solve_ok = all_checks.get("SOLVE_CLUE", {}).get("status") == "correct"
        observe_ok = all_checks.get("OBSERVE_ITEM", {}).get("status") == "correct"
        if not (solve_ok and observe_ok):
            return None  # inherited cascade — no new FM

    # General rule matching
    for rule_action, keyword, fm_id in _FM_RULES:
        if action_name == rule_action and keyword in desc:
            return fm_id

    return None


def fm_label(fm_id: Optional[str]) -> str:
    """Return 'FM-X.Y: Name' or empty string."""
    if fm_id is None:
        return ""
    entry = MAST_TAXONOMY.get(fm_id, {})
    return f"{fm_id}: {entry.get('name', '?')}"


# ---------------------------------------------------------------------------
# Per-action verifier functions
# ---------------------------------------------------------------------------
# Each returns a list of (label, llm_str, gt_str, ok) rows for display,
# plus a checks dict for trace tagging.
#
# Signature: (puzzle, results, eval_result) -> (rows, checks)
#   rows:   List[(str, str, str, bool)]
#   checks: dict with "answer", "gt", "status", "desc", and any extras

VerifyResult = Tuple[List[Tuple[str, str, str, bool]], Dict[str, Any]]


def _error_type_from_desc(desc: str) -> str:
    """Turn verifier details into stable error category names."""
    categories = []
    for detail in (desc or "").split("+"):
        category = detail.partition("(")[0].strip().upper()
        if category and category not in categories:
            categories.append(category)
    return "+".join(categories)


def _verify_observe_clue(puzzle, results, eval_result) -> VerifyResult:
    so = results["OBSERVE_CLUE"].structured_output
    clue_id = so.get("item_id")
    gt_id = puzzle.clue.item_id
    ok = (clue_id == gt_id)
    rows = [("Clue Select", f"id:{clue_id}", f"id:{gt_id}", ok)]
    checks = {
        "answer": clue_id, "gt": gt_id,
        "status": "correct" if ok else "wrong",
        "desc": "" if ok else f"clue_select_wrong(id:{clue_id}!=id:{gt_id})",
    }
    return rows, checks


def _verify_solve_clue(puzzle, results, eval_result) -> VerifyResult:
    so = results["SOLVE_CLUE"].structured_output
    ans = so.get("answer")
    unit = so.get("unit")
    gt_ans = puzzle.clue.answer
    gt_unit = puzzle.clue.delta_unit
    from benchmark_core.domains import get_domain
    domain = get_domain(getattr(puzzle.clue, "domain", "gsm-hard"))
    val_ok = domain.solve_answer_matches(ans, gt_ans) if ans is not None else False
    unit_ok = (unit == gt_unit)
    rows = [
        ("Math Value", str(ans), str(gt_ans), val_ok),
        ("Math Unit", str(unit), str(gt_unit), unit_ok),
    ]
    errors = []
    if not val_ok:
        errors.append(f"math_value_wrong({ans}!={gt_ans})")
    if not unit_ok:
        errors.append(f"math_unit_wrong({unit}!={gt_unit})")
    checks = {
        "answer": ans, "answer_unit": unit,
        "gt": gt_ans, "gt_unit": gt_unit,
        "status": "correct" if (val_ok and unit_ok) else "wrong",
        "desc": "+".join(errors),
    }
    return rows, checks


def _verify_observe_item(puzzle, results, eval_result) -> VerifyResult:
    so = results["OBSERVE_ITEM"].structured_output
    item_index = so.get("item_index")
    item_type = so.get("item_type")
    item_state = so.get("item_state")
    item_unit = so.get("item_unit")
    gt_type = puzzle.item.item_type.value
    gt_state = puzzle.item.state
    gt_unit = puzzle.item.unit

    type_ok = (item_type == gt_type)
    try:
        state_ok = math.isclose(float(item_state), float(gt_state), rel_tol=1e-4) if item_state is not None else False
    except (TypeError, ValueError):
        state_ok = False

    is_clock = isinstance(puzzle.item, Clock)
    if is_clock:
        hms = Clock._seconds_to_hms
        llm_state = f"{item_state}s ({hms(int(item_state)) if item_state else '?'})"
        gt_state_str = f"{gt_state}s ({hms(int(gt_state))})"
    else:
        llm_state = f"{item_state} {item_unit}"
        gt_state_str = f"{gt_state} {gt_unit}"

    rows = [
        ("Item Type", f"id:{item_index} {item_type}", f"id:{puzzle.item.item_id} {gt_type}", type_ok),
        ("Item State", llm_state, gt_state_str, state_ok),
    ]
    errors = []
    if not type_ok:
        errors.append(f"item_type_wrong({item_type}!={gt_type})")
    if not state_ok:
        errors.append(f"item_state_wrong({item_state}!={gt_state})")
    checks = {
        "answer": item_type, "answer_state": item_state,
        "gt": gt_type, "gt_state": gt_state,
        "status": "correct" if (type_ok and state_ok) else "wrong",
        "desc": "+".join(errors),
    }
    return rows, checks


def _verify_apply_delta(puzzle, results, eval_result) -> VerifyResult:
    so = results["APPLY_DELTA"].structured_output
    final_ans = so.get("answer")
    gt_ans = puzzle.clue.answer
    gt_unit = puzzle.clue.delta_unit
    gt_final = puzzle.item._groundtruth_answer(gt_ans, gt_unit)

    is_clock = isinstance(puzzle.item, Clock)
    if is_clock:
        hms = Clock._seconds_to_hms
        llm_ans_s = Clock._parse_time_to_seconds(final_ans)
        llm_str = f"{final_ans} → {hms(llm_ans_s)}({llm_ans_s}s)" if llm_ans_s is not None else str(final_ans)
        gt_str = f"{hms(gt_final)}({gt_final}s)"
    else:
        llm_str = str(final_ans)
        gt_str = str(gt_final)

    ok = (eval_result == "correct")
    rows = [("Apply Delta", llm_str, gt_str, ok)]

    if not ok:
        try:
            diff = float(final_ans) - gt_final if final_ans is not None and gt_final is not None else None
        except (TypeError, ValueError):
            diff = None
        if diff is not None:
            rows.append(("", f"diff={diff}", "", False))

    checks = {
        "answer": final_ans, "gt": gt_final,
        "status": "correct" if ok else "wrong",
        "desc": "" if ok else f"apply_delta_wrong({final_ans}!={gt_final})",
    }
    return rows, checks


def _verify_observe_puzzle(puzzle, results, eval_result) -> VerifyResult:
    so = results["OBSERVE_PUZZLE"].structured_output
    observer_solved = so.get("solved", False)
    actual_solved = (eval_result == "correct")
    agreement = observer_solved == actual_solved
    rows = [
        ("Puzzle Check", str(observer_solved), str(actual_solved), agreement),
    ]
    checks = {
        "observer_solved": observer_solved,
        "actual_solved": actual_solved,
        "agreement": agreement,
        "status": "correct" if agreement else "wrong",
        "desc": "" if agreement else f"observer_disagrees(obs={observer_solved},actual={actual_solved})",
    }
    return rows, checks


def _verify_select_puzzle(puzzle, results, eval_result, room=None, trace=None) -> VerifyResult:
    so = results["SELECT_PUZZLE"].structured_output
    chosen_id = so.get("puzzle_id", "")

    # Get snapshots from selection time in trace
    avail_ids = set()
    solved_ids = set()
    if trace is not None:
        for entry in reversed(trace):
            if entry.get("action") == "SELECT_PUZZLE" and "avail_at_select" in entry:
                avail_ids = set(entry["avail_at_select"])
                solved_ids = set(entry.get("solved_at_select", []))
                break

    errors = []

    # Check 1: was the chosen puzzle available at selection time?
    if avail_ids or solved_ids:
        all_ids = {p.puzzle_id for p in room.puzzles} if room else set()

        if chosen_id in solved_ids:
            errors.append(f"already_solved({chosen_id})")
        elif chosen_id not in avail_ids:
            if chosen_id in all_ids:
                errors.append(f"puzzle_locked({chosen_id})")
            else:
                errors.append(f"puzzle_not_found({chosen_id})")

    # Check 2: does this choice lead to deadend? (checked post-hoc)
    if room is not None:
        _leads_to_deadend = getattr(room, "leads_to_deadend", False)
        if _leads_to_deadend:
            errors.append("leads_to_deadend")

    ok = len(errors) == 0
    avail_str = ",".join(sorted(avail_ids))
    rows = [("Puzzle Select", chosen_id, f"avail:[{avail_str}]", ok)]
    checks = {
        "answer": chosen_id, "gt": avail_str,
        "status": "correct" if ok else "wrong",
        "desc": "+".join(errors),
    }
    return rows, checks


def _verify_observe_planned_clue(puzzle, results, eval_result) -> VerifyResult:
    so = results["OBSERVE_PLANNED_CLUE"].structured_output
    clue_id = so.get("item_id")
    gt_id = puzzle.clue.item_id
    ok = (clue_id == gt_id)
    rows = [("Clue Select", f"id:{clue_id}", f"id:{gt_id}", ok)]
    checks = {
        "answer": clue_id, "gt": gt_id,
        "status": "correct" if ok else "wrong",
        "desc": "" if ok else f"clue_select_wrong(id:{clue_id}!=id:{gt_id})",
    }
    return rows, checks


# Registry: action_name → verifier function
_ACTION_VERIFIERS: Dict[str, Callable] = {
    "OBSERVE_CLUE":         _verify_observe_clue,
    "SOLVE_CLUE":           _verify_solve_clue,
    "OBSERVE_ITEM":         _verify_observe_item,
    "APPLY_DELTA":          _verify_apply_delta,
    "OBSERVE_PUZZLE":       _verify_observe_puzzle,
    "SELECT_PUZZLE":        _verify_select_puzzle,
    "OBSERVE_PLANNED_CLUE": _verify_observe_planned_clue,
}


# ---------------------------------------------------------------------------
# PuzzleVerifier
# ---------------------------------------------------------------------------

class PuzzleVerifier:
    """
    Stateless verifier that compares action results against puzzle ground truth.

    Verification is driven by the actions list: for each action that has a
    registered verifier and results, it prints comparison rows and tags trace.
    """

    @staticmethod
    def verify(
        puzzle_index: int,
        results: Dict[str, ActionResult],
        room,
        eval_result: str,
        trace: List[Dict[str, Any]],
        actions: Optional[List[Action]] = None,
        puzzle=None,
    ) -> None:
        """Verify all actions: print table and tag trace entries.

        If puzzle is provided, use it directly. Otherwise look up by index.
        """
        if puzzle is None:
            puzzle = room.puzzles[puzzle_index - 1]
        action_names = [a.name for a in actions] if actions else list(_ACTION_VERIFIERS.keys())

        # Run all verifiers once
        all_rows: List[Tuple[str, str, str, bool]] = []
        checks: Dict[str, Dict[str, Any]] = {}
        for name in action_names:
            if name not in results or name not in _ACTION_VERIFIERS:
                continue
            verifier = _ACTION_VERIFIERS[name]
            try:
                rows, chk = verifier(puzzle, results, eval_result, room=room, trace=trace)
            except TypeError:
                try:
                    rows, chk = verifier(puzzle, results, eval_result, room=room)
                except TypeError:
                    rows, chk = verifier(puzzle, results, eval_result)
            all_rows.extend(rows)
            checks[name] = chk

        # Classify MAST FM for each failed action
        for name, chk in checks.items():
            chk["error_type"] = (
                _error_type_from_desc(chk.get("desc", ""))
                if chk.get("status") == "wrong" else ""
            )
            fm_id = classify_fm(name, chk, all_checks=checks)
            if fm_id:
                chk["fm_id"] = fm_id
                chk["fm_name"] = MAST_TAXONOMY[fm_id]["name"]
                chk["fm_category"] = MAST_TAXONOMY[fm_id]["category"]

        # Print verification table
        print(f"\n{_bar('═')}")
        room_id = getattr(room, "room_id", "")
        pid = puzzle.puzzle_id if puzzle else ""
        header = f"  {room_id} │ Puzzle {puzzle_index}/{pid} VERIFICATION" if pid else f"  {room_id} │ Puzzle {puzzle_index} VERIFICATION"
        print(header)
        print(_bar('═'))
        L = "  "
        print(f"{L}{'Phase':<16} {'LLM Output':<28} {'Ground Truth':<28} {'':>2}")
        print(f"{L}{'-'*16} {'-'*28} {'-'*28} {'-'*2}")
        # Map phase labels to their action's check desc
        phase_to_action = {}
        for name, chk in checks.items():
            # Match rows to checks by iterating in order
            for label, _, _, ok in all_rows:
                if label and label not in phase_to_action:
                    phase_to_action[label] = chk.get("desc", "")
                    break

        # Rebuild: map each row to its check desc via action order
        row_descs = []
        action_iter = iter(checks.values())
        current_chk = None
        for label, llm_val, gt_val, ok in all_rows:
            if label:  # new phase
                current_chk = next(action_iter, {})
            desc = current_chk.get("desc", "") if current_chk else ""
            row_descs.append(desc)

        # Build FM labels aligned with rows
        row_fms = []
        action_iter2 = iter(checks.items())
        current_action_name = None
        current_chk2 = None
        for label, llm_val, gt_val, ok in all_rows:
            if label:  # new phase
                current_action_name, current_chk2 = next(action_iter2, (None, {}))
            row_fms.append(current_chk2.get("fm_id") if current_chk2 else None)

        for (label, llm_val, gt_val, ok), desc, fid in zip(all_rows, row_descs, row_fms):
            mark = "✓" if ok else "✗"
            reason = f"  ({desc})" if not ok and desc else ""
            fm_tag = f"  [{fm_label(fid)}]" if not ok and fid else ""
            print(f"{L}{label:<16} {llm_val:<28} {gt_val:<28} {mark}{reason}{fm_tag}")
        print(_bar('═'))

        # Tag trace entries
        pid = puzzle.puzzle_id
        for entry in trace:
            if entry.get("puzzle_id") != pid:
                continue
            action_name = entry.get("action")
            if action_name in checks:
                entry["verification"] = checks[action_name]
                if (action_name == "APPLY_DELTA"
                        and entry.get("unit_tool_diagnostics", {}).get("result_use_errors")):
                    entry["verification"]["error_type"] = "TOOL_RESULT_USE_ERROR"
        return checks

    @staticmethod
    def get_failure_report(trace: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyse the trace and return a structured failure report.

        Failure categories:
          schema_failures    -> structured_output violated output_schema
          eval_results       -> per-puzzle: correct / trapped / wrong
        """
        report: Dict[str, Any] = {
            "schema_failures": [],
            "eval_results":    [],
            "unit_tool_failures": [],
        }
        tool_actions = 0
        total_calls = 0
        successful_calls = 0
        failed_calls = 0
        actions_with_call_error = 0
        recovered_actions = 0

        for entry in trace:
            errors = entry.get("schema_errors", {})
            if errors:
                report["schema_failures"].append({
                    "action": entry.get("action"),
                    "agent":  entry.get("agent"),
                    "errors": errors,
                })
            if "eval_result" in entry:
                report["eval_results"].append(entry["eval_result"])
            diagnostics = entry.get("unit_tool_diagnostics")
            if diagnostics:
                tool_actions += 1
                calls = entry.get("tool_calls", [])
                total_calls += len(calls)
                successful_calls += sum(c.get("status") == "success" for c in calls)
                failed_calls += sum(c.get("status") == "error" for c in calls)
                had_call_error = any(c.get("status") == "error" for c in calls)
                actions_with_call_error += int(had_call_error)
                if (had_call_error
                        and entry.get("verification", {}).get("status") == "correct"):
                    recovered_actions += 1
                for category in ("invocation_errors", "semantic_errors", "result_use_errors"):
                    for error in diagnostics.get(category, []):
                        report["unit_tool_failures"].append({
                            "puzzle_id": entry.get("puzzle_id"),
                            "action": entry.get("action"),
                            "category": category,
                            "error": error,
                        })

        report["unit_tool_summary"] = {
            "actions": tool_actions,
            "total_calls": total_calls,
            "successful_calls": successful_calls,
            "failed_calls": failed_calls,
            "actions_with_call_error": actions_with_call_error,
            "actions_recovered_after_call_error": recovered_actions,
            "call_success_rate": (
                successful_calls / total_calls if total_calls else None
            ),
            "error_recovery_rate": (
                recovered_actions / actions_with_call_error
                if actions_with_call_error else None
            ),
        }

        return report
