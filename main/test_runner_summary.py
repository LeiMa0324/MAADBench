import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from main.runner import EscapeRoomRunner
from benchmark_core.trace_utils import assign_action_ids


class RunnerToolSummaryTests(unittest.TestCase):
    def test_rooms_file_domain_is_detected_for_run_directory_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            rooms_path = Path(temp_dir) / "rooms.jsonl"
            rooms_path.write_text(json.dumps({
                "puzzles": [{"clue": {"domain": "livecodebench"}}],
            }) + "\n", encoding="utf-8")

            self.assertEqual(
                EscapeRoomRunner._file_clue_domain(str(rooms_path)),
                "livecodebench",
            )

    def test_tool_metadata_is_removed_only_from_llm_actions(self):
        parent = {
            "action": "APPLY_DELTA",
            "unit_tool_mode": "enabled",
            "tool_calls": [{"status": "success"}],
            "unit_tool_diagnostics": {"semantic_errors": []},
            "llm_calls": [{"duration": 1.0}],
        }
        child = {
            "action": "UNIT_TOOL_CALL",
            "call_index": 1,
            "input": {"available_tools": ["convert_time"]},
            "output_message": "Convert minutes to seconds.",
            "output_structured": {
                "tool_name": "convert_time", "arguments": {},
            },
            "tool_execution": {"status": "success", "result": 60},
        }

        EscapeRoomRunner._strip_tool_metadata_from_llm_actions([parent, child])

        for field in (
            "unit_tool_mode", "tool_calls", "unit_tool_diagnostics", "llm_calls",
        ):
            self.assertNotIn(field, parent)
        self.assertEqual(child["call_index"], 1)
        self.assertIn("convert_time", child["input"]["available_tools"])
        self.assertEqual(child["output_structured"]["tool_name"], "convert_time")
        self.assertEqual(child["tool_execution"]["result"], 60)

    def test_every_action_gets_one_row_without_puzzle_result(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = EscapeRoomRunner.__new__(EscapeRoomRunner)
            runner.summary_path = Path(temp_dir) / "summary.csv"
            runner.model = "test-model"
            runner.temperature = 0
            runner._init_summary_csv()

            room = SimpleNamespace(
                room_id="room_test",
                difficulty="easy",
                puzzles=[SimpleNamespace(
                    puzzle_id="P1",
                    solved=False,
                    item=SimpleNamespace(item_type=SimpleNamespace(value="scale")),
                )],
            )
            tool_entry = {
                "puzzle_id": "P1",
                "action_id": "room_test_P1_APPLY_DELTA_TOOL_CALL_1",
                "parent_action_id": "room_test_P1_APPLY_DELTA",
                "action": "UNIT_TOOL_CALL",
                "agent": "item_manager",
                "call_index": 1,
                "input": {
                    "available_tools": ["convert_mass"],
                },
                "output_message": "Convert ounces to kilograms.",
                "output_structured": {
                    "tool_name": "convert_mass",
                    "arguments": {"value": 16, "from_unit": "oz", "to_unit": "kg"},
                },
                "tool_execution": {
                    "status": "success", "result": 0.453592368, "error": None,
                },
                "verification": {
                    "status": "correct", "error_type": "", "anomaly_label": 0,
                    "expected_result": 0.453592368,
                    "desc": "",
                },
            }
            parent_entry = {
                "puzzle_id": "P1",
                "action_id": "room_test_P1_APPLY_DELTA",
                "action": "APPLY_DELTA",
                "agent": "item_manager",
                "structured": {"answer": 1},
                "verification": {
                    "status": "wrong", "answer": 1, "gt": 2,
                    "error_type": "APPLY_DELTA_WRONG", "desc": "wrong",
                },
            }
            successful_entry = {
                "puzzle_id": "P1",
                "action_id": "room_test_P1_OBSERVE_ITEM",
                "action": "OBSERVE_ITEM",
                "agent": "observer",
                "structured": {"item_type": "scale"},
                "verification": {
                    "status": "correct", "answer": "scale", "gt": "scale", "desc": "",
                },
            }

            trace = [successful_entry, tool_entry, parent_entry]
            assign_action_ids(trace, "room_test")
            runner._append_summary_csv(
                room,
                {"escaped": False, "trace": trace},
                "trace_test",
            )
            with runner.summary_path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))

            self.assertEqual(len(rows), 3)
            self.assertEqual([row["action_type"] for row in rows], [
                "LLM_action", "tool_action", "LLM_action",
            ])
            self.assertEqual([row["action_name"] for row in rows], [
                "OBSERVE_ITEM", "UNIT_TOOL_CALL", "APPLY_DELTA",
            ])
            self.assertEqual(rows[0]["action_status"], "correct")
            self.assertEqual(rows[0]["anomaly_label"], "0")
            self.assertEqual(rows[0]["agent"], "observer")
            self.assertEqual(
                rows[1]["parent_action_id"], "room_test_P1_APPLY_DELTA_3"
            )
            self.assertEqual(rows[1]["action_id"], trace[1]["action_id"])
            self.assertNotIn("action_index", rows[1])
            self.assertNotIn("action_level", rows[1])
            self.assertEqual(rows[1]["error_type"], "")
            self.assertEqual(rows[1]["anomaly_label"], "0")
            self.assertEqual(rows[1]["gt"], "0.453592368")
            self.assertEqual(rows[2]["action_status"], "wrong")
            self.assertEqual(rows[2]["anomaly_label"], "1")
            self.assertEqual(rows[2]["error_type"], "APPLY_DELTA_WRONG")
            self.assertNotIn("label", rows[2])
            self.assertNotIn("tool_error_type", rows[2])
            for removed_column in (
                "action_source", "call_index", "tool_name", "tool_arguments",
                "tool_result", "tool_error_message",
            ):
                self.assertNotIn(removed_column, rows[2])
            self.assertNotIn("failed_action", rows[2])
            self.assertNotIn("failed_agent", rows[2])
            self.assertNotIn("failure_source", rows[2])
            header = list(rows[0])
            self.assertEqual(header.index("puzzle_solved"), header.index("task_success") + 1)


if __name__ == "__main__":
    unittest.main()
