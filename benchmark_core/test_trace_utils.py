import unittest

from benchmark_core.trace_utils import assign_action_ids


class TraceActionIdTests(unittest.TestCase):
    def test_numeric_indexes_are_unique_and_tool_parent_is_rewritten(self):
        trace = [
            {
                "puzzle_id": "P1", "action": "OBSERVE_ITEM",
                "action_id": "room_test_P1_OBSERVE_ITEM", "attempt": 1,
            },
            {
                "puzzle_id": "P1", "action": "UNIT_TOOL_CALL",
                "action_id": "room_test_P1_APPLY_DELTA_TOOL_CALL_1",
                "parent_action_id": "room_test_P1_APPLY_DELTA", "attempt": 1,
            },
            {
                "puzzle_id": "P1", "action": "UNIT_TOOL_CALL",
                "action_id": "room_test_P1_APPLY_DELTA_TOOL_CALL_2",
                "parent_action_id": "room_test_P1_APPLY_DELTA", "attempt": 1,
            },
            {
                "puzzle_id": "P1", "action": "APPLY_DELTA",
                "action_id": "room_test_P1_APPLY_DELTA", "attempt": 1,
            },
        ]

        assign_action_ids(trace, "room_test")

        self.assertEqual(trace[0]["action_id"], "room_test_P1_OBSERVE_ITEM_1")
        self.assertEqual(
            trace[1]["action_id"], "room_test_P1_APPLY_DELTA_TOOL_CALL_1_2"
        )
        self.assertEqual(
            trace[2]["action_id"], "room_test_P1_APPLY_DELTA_TOOL_CALL_2_3"
        )
        self.assertEqual(trace[3]["action_id"], "room_test_P1_APPLY_DELTA_4")
        self.assertEqual(trace[1]["parent_action_id"], trace[3]["action_id"])
        self.assertEqual(trace[2]["parent_action_id"], trace[3]["action_id"])
        self.assertEqual(len({entry["action_id"] for entry in trace}), len(trace))
        self.assertTrue(all("action_index" not in entry for entry in trace))


if __name__ == "__main__":
    unittest.main()
