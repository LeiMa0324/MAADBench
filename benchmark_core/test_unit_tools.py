import unittest

from benchmark_core.unit_tools import (
    build_tool_action_entries,
    convert,
    diagnose,
    execute_call,
    run_tool_loop,
)
from benchmark_core.verifier import PuzzleVerifier


class _Statistic:
    def to_dict(self):
        return {"duration": 0, "input_tokens": 0, "output_tokens": 0}


class _LLM:
    def __init__(self):
        self.call_statistics = []


class _Output:
    def __init__(self, content):
        self.content = content


class _Agent:
    def __init__(self, responses):
        self.responses = iter(responses)
        self._llm = _LLM()

    def build_input(self, problem, previous_outputs):
        return {"problem": problem}

    def execute(self, input_data):
        self._llm.call_statistics.append(_Statistic())
        return _Output(next(self.responses))


class UnitToolTests(unittest.TestCase):
    def test_all_conversion_domains(self):
        cases = [
            ("convert_time", 2, "hours", "seconds", 7200),
            ("convert_temperature_delta", 18, "fahrenheit", "celsius", 10),
            ("convert_angle", 1, "turns", "degrees", 360),
            ("convert_mass", 1000, "g", "kg", 1),
        ]
        for tool, value, source, target, expected in cases:
            with self.subTest(tool=tool):
                actual = convert(tool, {
                    "value": value, "from_unit": source, "to_unit": target,
                })
                self.assertAlmostEqual(actual, expected)

    def test_loop_returns_error_to_agent_then_accepts_retry(self):
        agent = _Agent([
            {"message": "Convert the delta to seconds.",
             "tool_call": {"name": "convert_time", "arguments": {
                "value": 2, "from_unit": "kg", "to_unit": "seconds",
            }}},
            {"message": "Retry with the observed minutes unit.",
             "tool_call": {"name": "convert_time", "arguments": {
                "value": 2, "from_unit": "minutes", "to_unit": "seconds",
            }}},
            {"structured": {
                "answer": "00:01:40", "converted_delta": 120,
                "converted_unit": "seconds",
            }, "message": "converted and wrapped"},
        ])
        output, calls, statistics, _ = run_tool_loop(agent, "Apply delta", 3)
        self.assertEqual([c["status"] for c in calls], ["error", "success"])
        self.assertEqual(calls[0]["error"]["type"], "dimension_mismatch")
        self.assertEqual(calls[0]["output_message"], "Convert the delta to seconds.")
        self.assertEqual(calls[1]["output_message"], "Retry with the observed minutes unit.")
        self.assertEqual(output.content["structured"]["converted_delta"], 120)
        self.assertEqual(len(statistics), 3)

    def test_diagnostics_separate_call_and_result_use_errors(self):
        received = {
            "item_type": "clock", "item_state": 86380, "item_unit": "seconds",
            "delta": 2, "delta_unit": "minutes",
        }
        good_call = execute_call({"name": "convert_time", "arguments": {
            "value": 2, "from_unit": "minutes", "to_unit": "seconds",
        }}, 1)
        result = diagnose(
            received,
            {"answer": "00:02:00", "converted_delta": 2, "converted_unit": "minutes"},
            [good_call],
            {"item_type": "clock", "final_answer": 100},
        )
        self.assertFalse(result["invocation_errors"])
        self.assertFalse(result["semantic_errors"])
        self.assertCountEqual(result["result_use_errors"], [
            "converted_delta_wrong", "converted_unit_wrong",
            "final_answer_wrong_from_expected_input",
        ])
        self.assertFalse(result["gold_comparison"]["matches_gold"])

    def test_expected_tool_and_arguments_use_gold_input(self):
        call = execute_call({"name": "convert_angle", "arguments": {
            "value": 183933.6, "from_unit": "degrees", "to_unit": "degrees",
        }}, 1)
        result = diagnose(
            {"item_type": "compass", "item_state": 350, "item_unit": "degrees",
             "delta": 183933.6, "delta_unit": "dollars"},
            {"answer": 204, "converted_delta": 183933.6,
             "converted_unit": "degrees"},
            [call],
            {"item_type": "compass", "item_state": 350, "item_unit": "degrees",
             "delta": 183931.4666666667, "delta_unit": "turns",
             "final_answer": 158},
        )
        self.assertEqual(result["expected"]["tool"], "convert_angle")
        self.assertEqual(result["expected"]["arguments"], {
            "value": 183931.4666666667,
            "from_unit": "turns",
            "to_unit": "degrees",
        })
        self.assertAlmostEqual(
            result["expected"]["converted_delta"], 66215328.0
        )
        self.assertIn("wrong_value", result["semantic_errors"][0]["reasons"])
        self.assertIn("wrong_from_unit", result["semantic_errors"][0]["reasons"])
        self.assertNotIn("wrong_tool", result["semantic_errors"][0]["reasons"])

    def test_call_limit_is_captured(self):
        call = {"tool_call": {"name": "convert_time", "arguments": {
            "value": 1, "from_unit": "minutes", "to_unit": "seconds",
        }}}
        agent = _Agent([call, call, {"structured": {
            "answer": 60, "converted_delta": 60, "converted_unit": "seconds",
        }}])
        _, calls, _, _ = run_tool_loop(agent, "Apply delta", 1)
        self.assertEqual(calls[0]["status"], "success")
        self.assertEqual(calls[1]["error"]["type"], "tool_call_limit_exceeded")

    def test_tool_call_is_materialized_as_verified_child_action(self):
        call = execute_call({"name": "convert_mass", "arguments": {
            "value": 16, "from_unit": "oz", "to_unit": "kg",
        }}, 1)
        call["output_message"] = "Convert ounces to kilograms for the scale."
        diagnostics = diagnose(
            {"item_type": "scale", "item_state": 1, "item_unit": "kg",
             "delta": 16, "delta_unit": "oz"},
            {"answer": 1, "converted_delta": call["result"], "converted_unit": "kg"},
            [call],
        )
        entries = build_tool_action_entries({
            "action_id": "room_test_P1_APPLY_DELTA", "puzzle_id": "P1",
            "agent": "item_manager", "attempt": 1, "tool_calls": [call],
        }, diagnostics)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["action"], "UNIT_TOOL_CALL")
        self.assertEqual(entries[0]["parent_action_id"], "room_test_P1_APPLY_DELTA")
        self.assertEqual(entries[0]["verification"]["error_type"], "")
        self.assertEqual(entries[0]["verification"]["anomaly_label"], 0)
        self.assertAlmostEqual(
            entries[0]["verification"]["expected_result"], call["result"]
        )
        self.assertEqual(entries[0]["output_message"], call["output_message"])
        self.assertEqual(entries[0]["output_structured"], {
            "tool_name": "convert_mass",
            "arguments": {"value": 16, "from_unit": "oz", "to_unit": "kg"},
        })
        self.assertEqual(entries[0]["tool_execution"]["result"], call["result"])
        self.assertIn("convert_mass", entries[0]["input"]["available_tools"])
        self.assertNotIn("message", entries[0])
        self.assertNotIn("structured", entries[0])

    def test_missing_required_call_is_materialized(self):
        received = {
            "item_type": "clock", "item_state": 0, "item_unit": "seconds",
            "delta": 2, "delta_unit": "minutes",
        }
        diagnostics = diagnose(
            received,
            {"answer": "00:02:00", "converted_delta": 120,
             "converted_unit": "seconds"},
            [],
        )
        entries = build_tool_action_entries({
            "action_id": "room_test_P1_APPLY_DELTA", "puzzle_id": "P1",
            "agent": "item_manager", "attempt": 1, "tool_calls": [],
        }, diagnostics)
        self.assertEqual(len(entries), 1)
        self.assertEqual(
            entries[0]["action_id"],
            "room_test_P1_APPLY_DELTA_TOOL_CALL_MISSING",
        )
        self.assertEqual(
            entries[0]["verification"]["error_type"], "TC_REQUIRED_NOT_CALLED"
        )
        self.assertEqual(entries[0]["verification"]["anomaly_label"], 1)

    def test_missing_tool_message_is_an_anomaly(self):
        call = execute_call({"name": "convert_time", "arguments": {
            "value": 2, "from_unit": "minutes", "to_unit": "seconds",
        }}, 1)
        diagnostics = diagnose(
            {"item_type": "clock", "item_state": 0, "item_unit": "seconds",
             "delta": 2, "delta_unit": "minutes"},
            {"answer": "00:02:00", "converted_delta": 120,
             "converted_unit": "seconds"},
            [call],
        )
        entry = build_tool_action_entries({
            "action_id": "room_test_P1_APPLY_DELTA", "puzzle_id": "P1",
            "agent": "item_manager", "attempt": 1, "tool_calls": [call],
        }, diagnostics)[0]
        self.assertEqual(entry["verification"]["error_type"], "TC_MISSING_MESSAGE")
        self.assertEqual(entry["verification"]["anomaly_label"], 1)

    def test_failed_call_gets_specific_anomaly_label(self):
        call = execute_call({"name": "convert_time", "arguments": {
            "value": 2, "from_unit": "kg", "to_unit": "seconds",
        }}, 1)
        call["output_message"] = "Convert the supplied delta to seconds."
        diagnostics = diagnose(
            {"item_type": "clock", "item_state": 0, "item_unit": "seconds",
             "delta": 2, "delta_unit": "kg"},
            {"answer": 0, "converted_delta": 0, "converted_unit": "seconds"},
            [call],
        )
        entry = build_tool_action_entries({
            "action_id": "room_test_P1_APPLY_DELTA", "puzzle_id": "P1",
            "agent": "item_manager", "attempt": 1, "tool_calls": [call],
        }, diagnostics)[0]
        self.assertEqual(entry["verification"]["error_type"], "TC_DIMENSION_MISMATCH")
        self.assertEqual(entry["verification"]["anomaly_label"], 1)
        self.assertEqual(
            entry["output_structured"]["arguments"]["from_unit"], "kg"
        )
        self.assertEqual(entry["tool_execution"]["status"], "error")

    def test_recovery_requires_correct_parent_action(self):
        entry = {
            "action": "APPLY_DELTA",
            "tool_calls": [{"status": "error"}],
            "unit_tool_diagnostics": {
                "invocation_errors": [{"type": "dimension_mismatch"}],
                "semantic_errors": [],
                "result_use_errors": [],
            },
            "verification": {"status": "wrong"},
        }
        report = PuzzleVerifier.get_failure_report([entry])
        self.assertEqual(
            report["unit_tool_summary"]["actions_recovered_after_call_error"], 0
        )
        entry["verification"]["status"] = "correct"
        report = PuzzleVerifier.get_failure_report([entry])
        self.assertEqual(
            report["unit_tool_summary"]["actions_recovered_after_call_error"], 1
        )


if __name__ == "__main__":
    unittest.main()
