"""Common contract for EscapeRoom clue domains."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class ClueDomain(ABC):
    """A self-contained source of problems, prompts, distractors, and grading policy."""

    name: str

    @abstractmethod
    def load_problems(self) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def create_fake_clue(self, clue_class, hint: Optional[str] = None):
        pass

    def action_overrides(self) -> Dict[str, Dict[str, str]]:
        return {}

    def agent_prompt_overrides(self) -> Dict[str, str]:
        return {}

    def solve_answer_matches(self, predicted: Any, gold: Any) -> bool:
        try:
            return float(predicted) == float(gold)
        except (TypeError, ValueError):
            return False

    @property
    def item_absolute_tolerance(self) -> Optional[float]:
        return None
