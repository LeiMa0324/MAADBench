"""Unit-conversion tools and diagnostics for the APPLY_DELTA action."""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional


UNIT_GROUPS: Dict[str, Dict[str, float]] = {
    "convert_time": {"seconds": 1.0, "minutes": 60.0, "hours": 3600.0},
    # These are temperature *delta* factors. Absolute-temperature offsets do
    # not apply when a value is being added to an existing reading.
    "convert_temperature_delta": {
        "celsius": 1.0, "fahrenheit": 5.0 / 9.0, "kelvin": 1.0,
    },
    "convert_angle": {"degrees": 1.0, "radians": 180.0 / math.pi, "turns": 360.0},
    "convert_mass": {"kg": 1.0, "g": 0.001, "lbs": 0.45359237, "oz": 0.028349523},
}

UNIT_TO_TOOL = {
    unit: tool_name
    for tool_name, units in UNIT_GROUPS.items()
    for unit in units
}

TOOL_PROTOCOL = """

UNIT CONVERSION TOOLS ARE AVAILABLE. When a conversion is needed, call exactly
one tool by returning this JSON (and no final answer yet):
{"message":"<brief reason this conversion is needed>","tool_call":{"name":"<tool name>","arguments":{"value":<number>,"from_unit":"<unit>","to_unit":"<unit>"}}}

The message must be a concise, auditable justification for the tool choice and
arguments. Do not provide a hidden chain of thought or the final answer there.

Available tools:
- convert_time: seconds, minutes, hours
- convert_temperature_delta: celsius, fahrenheit, kelvin (converts a temperature delta; no absolute offset)
- convert_angle: degrees, radians, turns
- convert_mass: kg, g, lbs, oz

After receiving the tool result, either call another tool or return the final
APPLY_DELTA JSON required above. Use the returned value in the calculation.
"""


_INVOCATION_LABELS = {
    "tool_not_found": "TC_UNKNOWN_TOOL",
    "unknown_unit": "TC_UNKNOWN_UNIT",
    "dimension_mismatch": "TC_DIMENSION_MISMATCH",
    "tool_call_limit_exceeded": "TC_CALL_LIMIT_EXCEEDED",
    "internal_error": "TC_EXECUTION_ERROR",
}


class UnitToolError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def convert(tool_name: str, arguments: Any) -> float:
    if tool_name not in UNIT_GROUPS:
        raise UnitToolError("tool_not_found", f"unknown tool '{tool_name}'")
    if not isinstance(arguments, dict):
        raise UnitToolError("invalid_arguments", "arguments must be an object")
    missing = [key for key in ("value", "from_unit", "to_unit") if key not in arguments]
    if missing:
        raise UnitToolError("missing_arguments", f"missing required arguments: {missing}")

    value = arguments["value"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise UnitToolError("invalid_value", "value must be a finite number")
    from_unit = arguments["from_unit"]
    to_unit = arguments["to_unit"]
    if not isinstance(from_unit, str) or not isinstance(to_unit, str):
        raise UnitToolError("invalid_unit_type", "from_unit and to_unit must be strings")

    units = UNIT_GROUPS[tool_name]
    all_units = set(UNIT_TO_TOOL)
    for field, unit in (("from_unit", from_unit), ("to_unit", to_unit)):
        if unit not in all_units:
            raise UnitToolError("unknown_unit", f"unknown {field} '{unit}'")
        if unit not in units:
            raise UnitToolError(
                "dimension_mismatch",
                f"{field} '{unit}' is incompatible with {tool_name}",
            )
    return float(value) * units[from_unit] / units[to_unit]


def execute_call(tool_call: Any, call_index: int) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "call_index": call_index,
        "status": "error",
        "tool": None,
        "arguments": None,
        "result": None,
        "error": None,
    }
    try:
        if not isinstance(tool_call, dict):
            raise UnitToolError("invalid_tool_call", "tool_call must be an object")
        record["tool"] = tool_call.get("name")
        record["arguments"] = tool_call.get("arguments")
        result = convert(record["tool"], record["arguments"])
        record.update(status="success", result=result)
    except UnitToolError as exc:
        record["error"] = {"type": exc.code, "message": str(exc)}
    except Exception as exc:  # keep tool failures observable without stopping a run
        record["error"] = {"type": "internal_error", "message": f"{type(exc).__name__}: {exc}"}
    return record


def tool_feedback(records: List[Dict[str, Any]]) -> str:
    return (
        "\n\nUNIT TOOL CALL HISTORY:\n"
        + json.dumps(records, ensure_ascii=False, indent=2)
        + "\nContinue from these results. You may retry a failed call. If you have "
          "enough information, return the final APPLY_DELTA JSON now."
    )


def _print_tool_call(record: Dict[str, Any]) -> None:
    """Print the model's tool decision and the execution result."""
    print(f"\n  [Tool Call {record['call_index']}]")
    print(f"  [output message] {record.get('output_message', '')}")
    print("  [output structured]")
    print(json.dumps({
        "tool_name": record.get("tool"),
        "arguments": record.get("arguments"),
    }, ensure_ascii=False, indent=2))
    print("  [tool execution]")
    print(json.dumps({
        "status": record.get("status"),
        "result": record.get("result"),
        "error": record.get("error"),
    }, ensure_ascii=False, indent=2))


def run_tool_loop(agent, prompt: str, max_calls: int):
    """Run the model/tool/model loop and return final output plus audit data."""
    records: List[Dict[str, Any]] = []
    statistic_start = len(agent._llm.call_statistics)
    current_prompt = prompt + TOOL_PROTOCOL

    while True:
        output = agent.execute(agent.build_input(problem=current_prompt, previous_outputs={}))
        content = output.content if isinstance(output.content, dict) else {}
        tool_call = content.get("tool_call")
        if tool_call is None:
            break
        if len(records) >= max_calls:
            record = {
                "call_index": len(records) + 1,
                "status": "error",
                "tool": tool_call.get("name") if isinstance(tool_call, dict) else None,
                "arguments": tool_call.get("arguments") if isinstance(tool_call, dict) else None,
                "output_message": str(content.get("message", "")),
                "result": None,
                "error": {
                    "type": "tool_call_limit_exceeded",
                    "message": f"maximum of {max_calls} unit-tool calls reached",
                },
            }
            records.append(record)
            _print_tool_call(record)
            current_prompt = (
                prompt + TOOL_PROTOCOL + tool_feedback(records)
                + "\nThe tool-call limit is reached. Return the final APPLY_DELTA JSON; do not call another tool."
            )
            output = agent.execute(agent.build_input(problem=current_prompt, previous_outputs={}))
            repeated = output.content.get("tool_call") if isinstance(output.content, dict) else None
            if repeated is not None:
                repeated_content = output.content
                record = {
                    "call_index": len(records) + 1,
                    "status": "error",
                    "tool": repeated.get("name") if isinstance(repeated, dict) else None,
                    "arguments": repeated.get("arguments") if isinstance(repeated, dict) else None,
                    "output_message": str(repeated_content.get("message", "")),
                    "result": None,
                    "error": {
                        "type": "tool_call_limit_exceeded",
                        "message": "model requested another tool after the final-answer instruction",
                    },
                }
                records.append(record)
                _print_tool_call(record)
            break
        record = execute_call(tool_call, len(records) + 1)
        record["output_message"] = str(content.get("message", ""))
        records.append(record)
        _print_tool_call(record)
        current_prompt = prompt + TOOL_PROTOCOL + tool_feedback(records)

    statistics = [s.to_dict() for s in agent._llm.call_statistics[statistic_start:]]
    return output, records, statistics, prompt + TOOL_PROTOCOL


def _as_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _close(a: Any, b: Any) -> bool:
    left, right = _as_number(a), _as_number(b)
    return left is not None and right is not None and math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)


def _parse_final(answer: Any, item_type: Any) -> Optional[int]:
    if item_type == "clock" and isinstance(answer, str) and ":" in answer:
        try:
            parts = [int(x) for x in answer.split(":")]
            if len(parts) == 2:
                parts.append(0)
            return parts[0] * 3600 + parts[1] * 60 + parts[2] if len(parts) == 3 else None
        except (TypeError, ValueError):
            return None
    number = _as_number(answer)
    return int(round(number)) if number is not None else None


def diagnose(
    received: Dict[str, Any],
    final_output: Dict[str, Any],
    calls: List[Dict[str, Any]],
    gold: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Separate invocation, semantic-selection, and result-use failures."""
    gold = gold or {}

    def expected_value(key: str, received_key: Optional[str] = None):
        value = gold.get(key)
        return value if value is not None else received.get(received_key or key)

    value = expected_value("delta")
    from_unit = expected_value("delta_unit")
    to_unit = expected_value("item_unit")
    item_state = expected_value("item_state")
    item_type = expected_value("item_type")
    conversion_required = from_unit != to_unit
    expected_tool = (
        UNIT_TO_TOOL.get(from_unit)
        if conversion_required
        and UNIT_TO_TOOL.get(from_unit) == UNIT_TO_TOOL.get(to_unit)
        else None
    )
    expected_converted = _as_number(value) if not conversion_required else None
    if conversion_required and expected_tool:
        try:
            expected_converted = convert(expected_tool, {
                "value": value, "from_unit": from_unit, "to_unit": to_unit,
            })
        except UnitToolError:
            pass

    invocation_errors = [
        {"call_index": c["call_index"], **(c.get("error") or {})}
        for c in calls if c.get("status") == "error"
    ]
    semantic_errors: List[Dict[str, Any]] = []
    matching_success = False
    for call in calls:
        if call.get("status") != "success":
            continue
        args = call.get("arguments") or {}
        reasons = []
        if call.get("tool") != expected_tool:
            reasons.append("wrong_tool")
        if not _close(args.get("value"), value):
            reasons.append("wrong_value")
        if args.get("from_unit") != from_unit:
            reasons.append("wrong_from_unit")
        if args.get("to_unit") != to_unit:
            reasons.append("wrong_to_unit")
        if reasons:
            semantic_errors.append({"call_index": call["call_index"], "reasons": reasons})
        else:
            matching_success = True
    if conversion_required and expected_tool and not matching_success:
        semantic_errors.append({"call_index": None, "reasons": ["required_conversion_not_called_correctly"]})

    use_errors = []
    if expected_converted is not None:
        if not _close(final_output.get("converted_delta"), expected_converted):
            use_errors.append("converted_delta_wrong")
        if final_output.get("converted_unit") != to_unit:
            use_errors.append("converted_unit_wrong")

    expected_final = None
    state = _as_number(item_state)
    if state is not None and expected_converted is not None:
        raw = state + expected_converted
        if item_type == "clock":
            raw %= 86400.0
        elif item_type == "compass":
            wrap = 360.0 / UNIT_GROUPS["convert_angle"][to_unit]
            raw %= wrap
        expected_final = int(round(raw))
        if _parse_final(final_output.get("answer"), item_type) != expected_final:
            use_errors.append("final_answer_wrong_from_expected_input")

    gold_match = None
    if gold.get("final_answer") is not None:
        gold_match = _parse_final(final_output.get("answer"), gold.get("item_type")) == gold["final_answer"]

    return {
        "received_input": {k: received.get(k) for k in (
            "item_type", "item_state", "item_unit", "delta", "delta_unit"
        )},
        "expected": {
            "tool": expected_tool,
            "arguments": {
                "value": value,
                "from_unit": from_unit,
                "to_unit": to_unit,
            },
            "converted_delta": expected_converted,
            "converted_unit": to_unit,
            "final_answer": expected_final,
        },
        "invocation_errors": invocation_errors,
        "semantic_errors": semantic_errors,
        "result_use_errors": use_errors,
        "gold_comparison": {"matches_gold": gold_match, **gold},
    }


def build_tool_action_entries(
    parent_entry: Dict[str, Any],
    diagnostics: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Materialize audited tool calls as child actions of APPLY_DELTA.

    A tool action is labelled against the puzzle's gold conversion. The input
    received from upstream remains visible separately for propagation analysis.
    """
    parent_id = parent_entry.get("action_id", "")
    calls = parent_entry.get("tool_calls", []) or []
    invocation_by_index = {
        error.get("call_index"): error
        for error in diagnostics.get("invocation_errors", [])
        if isinstance(error, dict)
    }
    semantic_by_index: Dict[Any, List[str]] = {}
    for error in diagnostics.get("semantic_errors", []):
        if not isinstance(error, dict):
            continue
        semantic_by_index.setdefault(error.get("call_index"), []).extend(
            error.get("reasons", [])
        )

    entries: List[Dict[str, Any]] = []
    for call in calls:
        call_index = call.get("call_index", len(entries) + 1)
        invocation = invocation_by_index.get(call_index)
        reasons = semantic_by_index.get(call_index, [])
        if invocation:
            error_type = invocation.get("type", "invalid_request")
            error_code = _INVOCATION_LABELS.get(error_type, "TC_INVALID_REQUEST")
            status = "wrong"
            desc = invocation.get("message", error_type)
        elif reasons:
            error_type = "semantic_error"
            error_code = "TC_WRONG_TOOL" if "wrong_tool" in reasons else "TC_WRONG_ARGUMENTS"
            status = "wrong"
            desc = ",".join(reasons)
        else:
            error_type = ""
            error_code = ""
            status = "correct"
            desc = ""

        if not str(call.get("output_message", "")).strip():
            error_code = "+".join(filter(None, (error_code, "TC_MISSING_MESSAGE")))
            status = "wrong"
            desc = ",".join(filter(None, (desc, "missing_tool_message")))

        entries.append({
            "puzzle_id": parent_entry.get("puzzle_id", ""),
            "action_id": f"{parent_id}_TOOL_CALL_{call_index}",
            "parent_action_id": parent_id,
            "action": "UNIT_TOOL_CALL",
            "agent": parent_entry.get("agent", "item_manager"),
            "attempt": parent_entry.get("attempt"),
            "call_index": call_index,
            "clue_domain": parent_entry.get("clue_domain"),
            "clue_problem_id": parent_entry.get("clue_problem_id"),
            "input": {
                "available_tools": list(UNIT_GROUPS),
                "received_input": diagnostics.get("received_input", {}),
            },
            "output_message": call.get("output_message", ""),
            "output_structured": {
                "tool_name": call.get("tool"),
                "arguments": call.get("arguments"),
            },
            "tool_execution": {
                "status": call.get("status"),
                "result": call.get("result"),
                "error": call.get("error"),
            },
            "schema_errors": {},
            "verification": {
                "status": status,
                "anomaly_label": 0 if status == "correct" else 1,
                "error_type": error_code,
                "desc": desc,
                "expected_tool": diagnostics.get("expected", {}).get("tool"),
                "expected_arguments": diagnostics.get("expected", {}).get("arguments", {}),
                "expected_result": diagnostics.get("expected", {}).get("converted_delta"),
                "actual_tool": call.get("tool"),
                "actual_arguments": call.get("arguments"),
            },
        })

    missing_required = any(
        isinstance(error, dict)
        and error.get("call_index") is None
        and "required_conversion_not_called_correctly" in error.get("reasons", [])
        for error in diagnostics.get("semantic_errors", [])
    )
    if not calls and missing_required:
        expected = diagnostics.get("expected", {})
        received = diagnostics.get("received_input", {})
        entries.append({
            "puzzle_id": parent_entry.get("puzzle_id", ""),
            "action_id": f"{parent_id}_TOOL_CALL_MISSING",
            "parent_action_id": parent_id,
            "action": "UNIT_TOOL_CALL",
            "agent": parent_entry.get("agent", "item_manager"),
            "attempt": parent_entry.get("attempt"),
            "call_index": None,
            "clue_domain": parent_entry.get("clue_domain"),
            "clue_problem_id": parent_entry.get("clue_problem_id"),
            "input": {
                "available_tools": list(UNIT_GROUPS),
                "received_input": received,
            },
            "output_message": "",
            "output_structured": {
                "tool_name": None,
                "arguments": None,
            },
            "tool_execution": None,
            "schema_errors": {},
            "verification": {
                "status": "wrong",
                "anomaly_label": 1,
                "error_type": "TC_REQUIRED_NOT_CALLED",
                "desc": "required unit conversion was not called",
                "expected_tool": expected.get("tool"),
                "expected_arguments": expected.get("arguments", {}),
                "expected_result": expected.get("converted_delta"),
                "actual_tool": None,
                "actual_arguments": None,
            },
        })

    return entries
