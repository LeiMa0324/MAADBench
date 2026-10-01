from benchmark_core.domains.base import ClueDomain
from benchmark_core.domains.livecodebench import distractors, problem_source, prompts, verifier


class LiveCodeBenchDomain(ClueDomain):
    name = "livecodebench"

    def load_problems(self):
        return problem_source.load_problems()

    def create_fake_clue(self, clue_class, hint=None):
        return distractors.create_fake_clue(clue_class, hint)

    def action_overrides(self):
        return prompts.action_overrides()

    def agent_prompt_overrides(self):
        return prompts.AGENT_PROMPT_OVERRIDES

    def solve_answer_matches(self, predicted, gold):
        return verifier.solve_answer_matches(predicted, gold)

    @property
    def item_absolute_tolerance(self):
        return verifier.ITEM_ABSOLUTE_TOLERANCE
