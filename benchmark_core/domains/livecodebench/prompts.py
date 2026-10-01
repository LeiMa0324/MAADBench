"""
prompts.py — retune the clue-related prompts for the code-EXECUTION task.

In the execution variant a clue no longer carries a math word-problem but a
Python program + a function call ("what does this return?"). This module
rewrites, at run time, the few prompts that explicitly say "math problem" so the
wording matches the task. It monkeypatches:

  * action.OBSERVE_CLUE / OBSERVE_PLANNED_CLUE / SOLVE_CLUE  (prompt_template)
  * agents.AGENT_PROMPTS["observer"] / ["clue_solver"]       (role prompts)

All placeholders ({room_desc}, {prev_input_block}) and the structured-output
JSON schema are kept identical, so the MAS pipeline is unaffected. No original
file is modified.
"""

from __future__ import annotations

# --- Action prompt templates (execution-flavored) --------------------------

_OBSERVE_CLUE_TMPL = """You are observing an escape room. Here is everything you can see:

{room_desc}

Your task: identify the ONE real clue among the notes on the floor.

Each note contains a CHAIN of two Python programs. Exactly ONE note has BOTH
programs valid and runnable. Every other note has at least one BROKEN program that
would not run — invalid syntax, a missing colon, unbalanced brackets, a mangled
keyword, or wrong indentation.

Decide ONLY by checking whether the code would actually run: mentally parse the
programs in each note and pick the single note whose Python is ALL valid.

Return a JSON object with exactly these two keys:

"message": "<State which programs are broken and why, then give the FULL TEXT of the valid program, its function call, and its hint. The next agent needs the exact program to run it.>",
"structured": {{
  "item_id": <integer, the item id of the clue>,
  "clue":    "<exact full text of the clue>"
}}
"""

_OBSERVE_PLANNED_CLUE_TMPL = """You are observing an escape room. Here is everything you can see:

{room_desc}

{prev_input_block}

Your task: find the ONE real clue that corresponds to the instrument type
specified by the planner above.

Many notes are decoys whose Python is BROKEN and would not run (invalid syntax,
missing colon, unbalanced brackets, mangled keyword, or wrong indentation). A REAL
clue must have BOTH:
  1. A VALID, runnable Python program (a function definition plus a call to it) —
     first check that the code would actually run, and discard any broken note.
  2. Among the notes whose code is valid, the hint that matches the specified
     instrument type and a unit of measurement for that instrument.

Select the single real clue for the specified instrument.

Return a JSON object with exactly these two keys:

"message": "<Include the FULL TEXT of the program and its function call, and the FULL TEXT of the hint. The next agent needs the exact program to run it.>",
"structured": {{
  "item_id": <integer, the item id of the clue>,
  "clue":    "<exact full text of the clue>"
}}
"""

# {valid_units} and {amplify_instruction} are substituted in apply().
_SOLVE_CLUE_TMPL = """You are a code executor in an escape room.

{prev_input_block}

Your task has TWO steps:
  1. The clue chains TWO programs. Run the FIRST program on its call to get a
     number a; then substitute a into the SECOND program's call as the clue
     instructs and run it — the value the SECOND program returns is the clue's result.
  2. {amplify_instruction}

IMPORTANT: your final answer is the number AFTER step 2 — NOT the raw value the
program returns. This transformed number will be used as a delta with a specified
unit to adjust an instrument in the room.

Instructions:
  - Trace through the code step by step in step 1
  - Do NOT forget step 2: the final answer must be the transformed number
  - The input above tells you which unit your answer should be expressed in
    — include that unit in your structured output
  - Do NOT apply the delta to any instrument yet

Return a JSON object with exactly these two keys:

"message": "<step-by-step trace of the execution, then the transformation applied>",
"structured": {{
  "answer": <number — the final transformed value (after step 2)>,
  "unit":   "<unit string — must be one of: {valid_units}>"
}}
"""

_SOLVE_CLUE_PREAMBLE = """\
The room observer has provided the following structured JSON:
---
{OBSERVE_CLUE}
---

The JSON contains:
  "item_id" — the clue's item id
  "clue"    — the full text of the clue (the program to run + instrument hint)"""

# --- Agent role prompts ----------------------------------------------------

_OBSERVER_PROMPT = (
    "You are the Observer in an escape room. You examine objects in the room "
    "and distinguish real instruments and clues from fakes.\n\n"
    "Fake items have tells: broken/damaged descriptions, or readings that "
    "are physically impossible for what they claim to be.\n\n"
    "Fake clues carry a normal-looking instrument hint just like real clues, so the "
    "hint is NOT a tell. The ONLY difference is the code: a fake clue's Python program "
    "is broken and would not run (invalid syntax or indentation), while a real clue's "
    "program is valid and runnable. Decide which clue is real purely by code validity.\n\n"
    "Always respond with a single valid JSON object — no markdown, no extra text."
)

_CLUE_SOLVER_PROMPT = (
    "You are the Code Executor. You receive a Python program (a function and a "
    "call to it) extracted from a clue found in the room. Execute it step by step "
    "and return the numerical value it produces.\n\n"
    "Always respond with a single valid JSON object — no markdown, no extra text."
)


def action_overrides():
    """Return per-action text without mutating the shared Action registry."""
    from benchmark_core import action_prompts
    from benchmark_core.domains.livecodebench import problem_source

    solve = _SOLVE_CLUE_TMPL.replace("{valid_units}", action_prompts._VALID_UNITS_STR)
    solve = solve.replace("{amplify_instruction}", problem_source._amplify_instruction())
    return {
        "OBSERVE_CLUE": {"prompt_template": _OBSERVE_CLUE_TMPL},
        "OBSERVE_PLANNED_CLUE": {"prompt_template": _OBSERVE_PLANNED_CLUE_TMPL},
        "SOLVE_CLUE": {"prompt_template": solve, "preamble": _SOLVE_CLUE_PREAMBLE},
    }


AGENT_PROMPT_OVERRIDES = {
    "observer": _OBSERVER_PROMPT,
    "clue_solver": _CLUE_SOLVER_PROMPT,
}
