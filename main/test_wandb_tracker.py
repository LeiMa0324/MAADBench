import argparse
import unittest
from unittest.mock import patch

from main.run_trace_matrix import CONFIG_FOLDERS, TEMPERATURES, _config_paths, main
from main.wandb_tracker import SuccessMetrics


class SuccessMetricsTests(unittest.TestCase):
    def test_room_puzzle_and_action_rates_include_tool_actions(self):
        metrics = SuccessMetrics()
        values = metrics.update("easy", {
            "escaped": True,
            "puzzles_solved": 1,
            "puzzles_total": 2,
            "trace": [
                {"action": "SOLVE_CLUE", "verification": {"status": "correct"}},
                {"action": "APPLY_DELTA", "verification": {"status": "wrong"}},
                {"action": "UNIT_TOOL_CALL", "verification": {"status": "correct"}},
            ],
        })
        self.assertEqual(values["room/success_rate"], 1.0)
        self.assertEqual(values["puzzle/success_rate"], 0.5)
        self.assertAlmostEqual(values["action/success_rate"], 2 / 3)
        self.assertEqual(values["action/llm_success_rate"], 0.5)
        self.assertEqual(values["action/tool_success_rate"], 1.0)
        self.assertEqual(values["difficulty/easy/puzzle_success_rate"], 0.5)

    def test_empty_trace_matches_summary_timeout_action(self):
        metrics = SuccessMetrics()
        values = metrics.update("hard", {
            "escaped": False,
            "puzzles_solved": 0,
            "puzzles_total": 3,
            "trace": [],
        })
        self.assertEqual(values["action/total"], 1)
        self.assertEqual(values["action/llm_total"], 1)
        self.assertEqual(values["action/success_rate"], 0.0)

    def test_matrix_has_three_temperatures_for_each_model(self):
        paths = _config_paths()
        self.assertEqual(len(paths), len(CONFIG_FOLDERS) * len(TEMPERATURES))
        self.assertTrue(all(path.is_file() for path in paths))
        deepseek_paths = _config_paths(("deepseek",))
        self.assertEqual(len(deepseek_paths), 3)
        self.assertTrue(all(path.parent.name == "deepseek" for path in deepseek_paths))

    def test_matrix_runs_five_model_lanes_in_parallel(self):
        class ImmediateProcess:
            active = 0
            maximum_active = 0
            launches = 0

            def __init__(self, *_args, **_kwargs):
                self.done = False
                type(self).active += 1
                type(self).launches += 1
                type(self).maximum_active = max(
                    type(self).maximum_active, type(self).active
                )

            def poll(self):
                if not self.done:
                    self.done = True
                    type(self).active -= 1
                return 0

        args = argparse.Namespace(
            rooms_files=None,
            output_dir="output",
            unit_tools="enabled",
            wandb_project="test",
            wandb_entity=None,
            wandb_mode="disabled",
            no_upload_artifacts=True,
            stop_on_error=False,
            max_parallel_models=5,
            state_file=None,
        )
        with (
            patch("main.run_trace_matrix.parse_args", return_value=args),
            patch("main.run_trace_matrix._write_state"),
            patch("main.run_trace_matrix.subprocess.Popen", ImmediateProcess),
            patch("main.run_trace_matrix.time.sleep"),
            patch("builtins.print"),
        ):
            self.assertEqual(main(), 0)
        self.assertEqual(ImmediateProcess.launches, 30)
        self.assertEqual(ImmediateProcess.maximum_active, 5)


if __name__ == "__main__":
    unittest.main()
