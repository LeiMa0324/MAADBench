"""
agents.py — Agent role definitions and default configurations per difficulty.

All agent prompts and the action/agent mappings for each difficulty level
are defined here, keeping config YAML focused on LLM settings only.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List
from benchmark_core.action_prompts import Action, actions_from_names


# ---------------------------------------------------------------------------
# Agent role prompts
# ---------------------------------------------------------------------------

AGENT_PROMPTS: Dict[str, str] = {
    "observer": (
        "You are the Observer in an escape room. You examine objects in the room "
        "and distinguish real instruments and clues from fakes.\n\n"
        "Fake items have tells: broken/damaged descriptions, or readings that "
        "are physically impossible for what they claim to be.\n\n"
        "Fake clues reference nonsensical instruments or units that don't exist. "
        "Real clues contain a solvable math problem and a hint pointing to a "
        "specific type of instrument and unit.\n\n"
        "Always respond with a single valid JSON object — no markdown, no extra text."
    ),
    "clue_solver": (
        "You are the Clue Solver. You receive a math word problem extracted "
        "from a clue found in the room. Solve it step by step and return "
        "the numerical answer.\n\n"
        "Always respond with a single valid JSON object — no markdown, no extra text."
    ),
    "item_manager": (
        "You are the Item Manager. You receive an instrument's current reading, "
        "a delta value, and a unit. Compute the new reading after applying "
        "the delta. Pay attention to unit conversions.\n\n"
        "Always respond with a single valid JSON object — no markdown, no extra text."
    ),
    "planner": (
        "You are the Planner in an escape room with multiple puzzles. "
        "You must choose which puzzle to solve next based on item lock states "
        "and DAG effects. A wrong ordering may permanently lock required items.\n\n"
        "Always respond with a single valid JSON object — no markdown, no extra text."
    ),
}


# ---------------------------------------------------------------------------
# Per-difficulty defaults
# ---------------------------------------------------------------------------

# Action names per difficulty
DIFFICULTY_ACTIONS: Dict[str, List[str]] = {
    "easy":      ["OBSERVE_CLUE", "SOLVE_CLUE", "OBSERVE_ITEM", "APPLY_DELTA", "OBSERVE_PUZZLE"],
    "medium":    ["OBSERVE_CLUE", "SOLVE_CLUE", "OBSERVE_ITEM", "APPLY_DELTA", "OBSERVE_PUZZLE"],
    "hard":      ["OBSERVE_CLUE", "SOLVE_CLUE", "OBSERVE_ITEM", "APPLY_DELTA", "OBSERVE_PUZZLE"],
    "nightmare": ["OBSERVE_PLANNED_CLUE", "SOLVE_CLUE", "OBSERVE_ITEM", "APPLY_DELTA"],
}

# Agent roles needed per difficulty
DIFFICULTY_AGENTS: Dict[str, List[str]] = {
    "easy":      ["observer", "clue_solver", "item_manager"],
    "medium":    ["observer", "clue_solver", "item_manager"],
    "hard":      ["observer", "clue_solver", "item_manager"],
    "nightmare": ["observer", "clue_solver", "item_manager", "planner"],
}


def get_actions(difficulty: str, clue_domain=None) -> List[Action]:
    """Return independent Action objects with domain-specific prompt text."""
    names = DIFFICULTY_ACTIONS.get(difficulty, DIFFICULTY_ACTIONS["easy"])
    actions = deepcopy(actions_from_names(names))
    if clue_domain is not None:
        overrides = clue_domain.action_overrides()
        for action in actions:
            for field, value in overrides.get(action.name, {}).items():
                setattr(action, field, value)
    return actions


def get_agent_roles(difficulty: str) -> List[str]:
    """Return the agent role names needed for a given difficulty."""
    return DIFFICULTY_AGENTS.get(difficulty, DIFFICULTY_AGENTS["easy"])


def get_agent_prompt(role: str, clue_domain=None) -> str:
    """Return the default prompt for an agent role."""
    if clue_domain is not None:
        override = clue_domain.agent_prompt_overrides().get(role)
        if override is not None:
            return override
    return AGENT_PROMPTS.get(role, "")
