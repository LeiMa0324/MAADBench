import unittest

from benchmark_core.domains import get_domain
from benchmark_core.domains.gsm_hard import problem_source as gsm_source
from benchmark_core.action_prompts import SOLVE_CLUE
from benchmark_core.agent_prompts import get_actions, get_agent_prompt
from benchmark_core.room_core import Clue, PuzzleItemType


class DomainSwitchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_gsm_cache = gsm_source._CACHE
        gsm_source._CACHE = [{
            "problem_id": "gsm-test-1",
            "problem": "What is 1 + 1?",
            "answer": 2.0,
            "domain": "gsm-hard",
        }]

    @classmethod
    def tearDownClass(cls):
        gsm_source._CACHE = cls.old_gsm_cache

    def test_switching_does_not_mutate_shared_actions(self):
        gsm = get_domain("gsm-hard")
        live = get_domain("livecodebench")
        gsm_action = next(a for a in get_actions("easy", gsm) if a.name == "SOLVE_CLUE")
        live_action = next(a for a in get_actions("easy", live) if a.name == "SOLVE_CLUE")
        gsm_again = next(a for a in get_actions("easy", gsm) if a.name == "SOLVE_CLUE")

        self.assertIn("math problem", gsm_action.prompt_template)
        self.assertIn("chains TWO programs", live_action.prompt_template)
        self.assertEqual(gsm_action.prompt_template, gsm_again.prompt_template)
        self.assertEqual(gsm_action.prompt_template, SOLVE_CLUE.prompt_template)
        self.assertIn("Clue Solver", get_agent_prompt("clue_solver", gsm))
        self.assertIn("Code Executor", get_agent_prompt("clue_solver", live))

    def test_both_domains_produce_numeric_serializable_clues(self):
        for name in ("gsm-hard", "livecodebench", "gsm-hard"):
            domain = get_domain(name)
            clue = Clue.create(PuzzleItemType.SCALE, domain)
            self.assertEqual(clue.domain, name)
            self.assertIsInstance(clue.answer, (int, float))
            restored = Clue.from_dict(clue.to_dict())
            self.assertEqual(restored.domain, name)
            self.assertEqual(restored.problem_id, clue.problem_id)

    def test_livecode_distractor_is_domain_scoped(self):
        clue = Clue.create_fake(clue_domain=get_domain("livecodebench"))
        self.assertEqual(clue.domain, "livecodebench")
        self.assertIn("FIRST program", clue.hint)


if __name__ == "__main__":
    unittest.main()
