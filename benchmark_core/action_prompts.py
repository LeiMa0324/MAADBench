"""
action.py — Action definitions for the EscapeRoom MAS pipeline.

Each Action defines:
  - name:                      unique identifier
  - agent_role:                which agent executes this action
  - context_keys:              fields needed from the environment context
  - prev_output_keys:          fields needed from previous actions' outputs
  - prompt_template:           template with {prev_input_block} placeholder
  - preamble:                  text for {prev_input_block} (describes previous agent's JSON)
  - output_schema:             expected fields in structured_output (for verification)

Communication
=============
  structured_output → serialised as JSON with "reasoning" field (from message)
                      → passed to next agent
  message           → stored as "reasoning" in the JSON passed downstream

Output format (returned by LLM):
  {
    "structured": { ...verified fields... },
    "message":    "natural language reasoning (passed as 'reasoning' field in JSON)"
  }
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


# ---------------------------------------------------------------------------
# Output schema field descriptor
# ---------------------------------------------------------------------------

@dataclass
class OutputField:
    """Describes one field in an action's structured_output."""
    key:           str
    type:          Union[type, tuple]
    description:   str
    valid_values:  Optional[List[Any]] = None
    nullable:      bool = False


# ---------------------------------------------------------------------------
# Action dataclass
# ---------------------------------------------------------------------------

@dataclass
class Action:
    """
    A single step in the MAS pipeline.

    One prompt_template with a {prev_input_block} placeholder, filled by
    the preamble which describes the previous agent's JSON output.

    Agents always receive structured JSON (with a "reasoning" field).
    """
    name:                      str
    agent_role:                str
    context_keys:              List[str]
    prev_output_keys:          List[str]
    prompt_template:           str
    output_schema:             List[OutputField]
    preamble:                  str = ""
    submits_answer:            bool = False

    def build_prompt(
        self,
        context:       Dict[str, Any],
        prev_messages: Dict[str, str],   # action_name → JSON string
    ) -> str:
        """
        Fill the template with context and previous outputs.

        prev_messages values are JSON strings containing structured_output + reasoning.
        """
        template = self.prompt_template.replace("{prev_input_block}", self.preamble)

        values: Dict[str, Any] = {}
        for key in self.context_keys:
            values[key] = context.get(key, "")
        for key in self.prev_output_keys:
            action_name = key.split("__")[0]
            values[key] = prev_messages.get(action_name, "")
        try:
            return template.format(**values)
        except KeyError as e:
            raise ValueError(
                f"[Action:{self.name}] Missing placeholder {e} in template"
            )

    def validate_structured(self, structured: Dict[str, Any]) -> Dict[str, str]:
        """
        Validate structured_output against output_schema.
        Returns dict of field -> error message for any violations.
        Empty dict means all fields are valid.
        """
        errors: Dict[str, str] = {}
        for field_def in self.output_schema:
            val = structured.get(field_def.key)
            if val is None and not field_def.nullable:
                errors[field_def.key] = "missing required field"
                continue
            if val is not None and not isinstance(val, field_def.type):
                if field_def.type is float and isinstance(val, int):
                    pass
                else:
                    expected = (field_def.type.__name__ if isinstance(field_def.type, type)
                                else "|".join(t.__name__ for t in field_def.type))
                    errors[field_def.key] = (
                        f"expected {expected}, got {type(val).__name__}"
                    )
            if val is not None and field_def.valid_values is not None:
                val_lower = val.lower() if isinstance(val, str) else val
                valid_lower = [v.lower() if isinstance(v, str) else v
                               for v in field_def.valid_values]
                if val_lower not in valid_lower:
                    errors[field_def.key] = (
                        f"value '{val}' not in valid_values={field_def.valid_values}"
                    )
        return errors


# ---------------------------------------------------------------------------
# Valid unit lists
# ---------------------------------------------------------------------------

from benchmark_core.room_core import (
    Clock, Thermometer, Compass, Scale, PuzzleItemType,
)

_CLOCK_UNITS        = list(Clock._TO_BASE.keys())
_THERMOMETER_UNITS  = list(Thermometer._TO_BASE.keys())
_COMPASS_UNITS      = list(Compass._TO_BASE.keys())
_SCALE_UNITS        = list(Scale._TO_BASE.keys())
_ALL_UNITS          = _CLOCK_UNITS + _THERMOMETER_UNITS + _COMPASS_UNITS + _SCALE_UNITS
_ALL_ITEM_TYPES     = [t.value for t in PuzzleItemType]

_VALID_UNITS_STR    = ", ".join(_ALL_UNITS)
_VALID_TYPES_STR    = ", ".join(_ALL_ITEM_TYPES)

# ---------------------------------------------------------------------------
# Action definitions
# ---------------------------------------------------------------------------

OBSERVE_CLUE = Action(
    name       = "OBSERVE_CLUE",
    agent_role = "observer",

    context_keys     = ["room_desc"],
    prev_output_keys = [],

    # No previous agent input — preambles are empty.
    prompt_template = """You are observing an escape room. Here is everything you can see:

{room_desc}

Your task: identify the ONE real clue among the notes on the floor.

A REAL clue has both:
  1. A math problem (a question involving numbers and calculation)
  2. A hint that describes a real instrument
     and a unit of measurement


Select the single real clue.

Return a JSON object with exactly these two keys:

"message": "<Include the FULL TEXT of the math problem and the FULL TEXT of the hint. The next agent needs the exact problem to solve it.>",
"structured": {{
  "item_id": <integer, the item id of the clue>,
  "clue":    "<exact full text of the clue>"
}}
""",

    output_schema = [
        OutputField("item_id", int, "the item id of the clue"),
        OutputField("clue",    str, "exact full text of the clue"),
    ],
)


# ---------------------------------------------------------------------------

SOLVE_CLUE = Action(
    name       = "SOLVE_CLUE",
    agent_role = "clue_solver",

    context_keys     = [],
    prev_output_keys = ["OBSERVE_CLUE"],

    prompt_template = """You are a clue solver in an escape room.

{prev_input_block}

Your task: extract the math problem from the input above and solve it.
Your numerical answer will later be used as a delta with a specified unit to adjust
an instrument in the room.

Instructions:
  - Solve step by step
  - Your final answer must be a single number
  - The input above tells you which unit your answer should be expressed in
    — include that unit in your structured output
  - Do NOT apply the delta to any instrument yet

Return a JSON object with exactly these two keys:

"message": "<step-by-step solution of the math problem, show all work>",
"structured": {{
  "answer": <number — the final solution>,
  "unit":   "<unit string — must be one of: {valid_units}>"
}}
""".replace("{valid_units}", _VALID_UNITS_STR),

    preamble = """\
The room observer has provided the following structured JSON:
---
{OBSERVE_CLUE}
---

The JSON contains:
  "item_id" — the clue's item id
  "clue"    — the full text of the clue (math problem + instrument hint)""",

    output_schema = [
        OutputField("answer", float, "numerical solution to the math problem"),
        OutputField("unit",   str,   "unit of the answer", valid_values=_ALL_UNITS),
    ],
)


# ---------------------------------------------------------------------------

OBSERVE_ITEM = Action(
    name       = "OBSERVE_ITEM",
    agent_role = "observer",

    context_keys     = ["room_desc"],
    prev_output_keys = ["SOLVE_CLUE"],

    prompt_template = """You are observing an escape room. Here is everything you can see:

{room_desc}

{prev_input_block}

Your task: identify the ONE real instrument that the clue's hint refers to.

A REAL instrument:
  - Has a physically valid reading and a working description
  - Matches the instrument type described in the clue

Select the single real instrument.

Return a JSON object with exactly these two keys:

"message": "<Include: (1) the instrument you found and its current reading with unit, (2) the delta value and delta unit from the solver's input above. The item manager needs ALL of this to compute the final answer.>",
"structured": {{
  "item_index":  <integer, 1-based index of the instrument in the item list>,
  "item_type":   "<one of: {valid_types}>",
  "item_state":  <current reading as a number — for clocks use total seconds since midnight>,
  "item_unit":   "<unit of the current reading — must be one of: {valid_units}>",
  "delta":       <number — the delta value from the solver's input above>,
  "delta_unit":  "<unit of the delta from the solver's input above>"
}}
""".replace("{valid_units}", _VALID_UNITS_STR).replace("{valid_types}", _VALID_TYPES_STR),

    preamble = """\
The clue solver has provided the following structured JSON:
---
{SOLVE_CLUE}
---

The JSON contains:
  "answer" — the numerical delta value computed from the clue
  "unit"   — the unit of that delta""",

    output_schema = [
        OutputField("item_index", int,   "1-based index of the chosen item"),
        OutputField("item_type",  str,   "instrument type", valid_values=_ALL_ITEM_TYPES),
        OutputField("item_state", float, "current reading as a number"),
        OutputField("item_unit",  str,   "unit of the current reading", valid_values=_ALL_UNITS),
        OutputField("delta",      float, "delta value from the math solver"),
        OutputField("delta_unit", str,   "unit of the delta", valid_values=_ALL_UNITS),
    ],
)


# ---------------------------------------------------------------------------

APPLY_DELTA = Action(
    name       = "APPLY_DELTA",
    agent_role = "item_manager",

    context_keys     = [],
    prev_output_keys = ["OBSERVE_ITEM"],

    prompt_template = """You are operating an instrument in an escape room.

{prev_input_block}

Your task: compute the new instrument reading after applying the delta.

Step-by-step instructions:
  1. Extract the instrument's current reading, unit, delta value, and delta unit
     from the input above
  2. If the delta unit differs from the instrument's unit, convert the delta first
  3. Add the converted delta to the current reading
  4. Apply wrapping if needed (clocks wrap at 24h, compasses at 360°)

Return a JSON object with exactly these two keys:

"message": "<step-by-step calculation showing unit conversion and final result>",
"structured": {{
  "answer":          <final instrument reading. For clocks use "HH:MM:SS" string; for others use an integer>,
  "converted_delta": <number — the delta after unit conversion, in the instrument's unit>,
  "converted_unit":  "<unit after conversion — same as instrument unit>"
}}
""",

    preamble = """\
The item observer has provided the following structured JSON:
---
{OBSERVE_ITEM}
---

The JSON contains:
  "item_type"  — instrument type
  "item_state" — current reading of the instrument (number)
  "item_unit"  — unit of the current reading
  "delta"      — the value to add to the reading (from the clue solver)
  "delta_unit" — the unit of the delta

IMPORTANT: use ONLY the values given in the JSON. Do not substitute or guess.""",

    output_schema = [
        OutputField("answer",          (int, str), "final answer: int or HH:MM:SS for clocks"),
        OutputField("converted_delta", float,      "delta after unit conversion"),
        OutputField("converted_unit",  str,        "unit after conversion", valid_values=_ALL_UNITS),
    ],

    submits_answer = True,
)


# ---------------------------------------------------------------------------

OBSERVE_PUZZLE = Action(
    name       = "OBSERVE_PUZZLE",
    agent_role = "observer",

    context_keys     = ["room_desc"],
    prev_output_keys = ["OBSERVE_CLUE", "OBSERVE_ITEM"],

    prompt_template = """You are observing an escape room. Here is everything you can see:

{room_desc}

{prev_input_block}

Your task: determine whether the current puzzle has been solved by comparing the
current room contents against your previous observation.

Step 1 — Inventory changes:
  List every item that DISAPPEARED (was in your previous observation but is
  no longer visible) and every item that is NEW (visible now but was not before).

Step 2 — Check the clue and instrument you previously identified:
  - Is the clue you identified still present in the room?
  - Is the instrument you identified still present in the room?

Step 3 — Determine solved:
  The puzzle is solved ONLY if BOTH the previously identified clue AND
  the previously identified instrument have DISAPPEARED from the room.
  If either one is still present, the puzzle is NOT solved.

Return a JSON object with exactly these two keys:

"message": "<Step 1: list disappeared and new items. Step 2: check clue and item presence. Step 3: conclusion.>",
"structured": {{
  "solved":       <true ONLY if both clue and item are gone, false otherwise>,
  "clue_present": <true if the previously identified clue is still visible>,
  "item_present": <true if the previously identified instrument is still visible>
}}
""",

    preamble = """\
Previously, you identified the following (structured JSON):

Clue observation:
---
{OBSERVE_CLUE}
---

Item observation:
---
{OBSERVE_ITEM}
---""",

    output_schema = [
        OutputField("solved",       bool, "whether the puzzle was solved"),
        OutputField("clue_present", bool, "whether the previous clue is still visible"),
        OutputField("item_present", bool, "whether the previous instrument is still visible"),
    ],
)


# ---------------------------------------------------------------------------

SELECT_PUZZLE = Action(
    name       = "SELECT_PUZZLE",
    agent_role = "planner",

    context_keys     = ["planning_desc"],
    prev_output_keys = [],

    prompt_template = """You are a planner in an escape room with multiple puzzles.
Each puzzle is tied to an item. Items can be UNLOCKED or LOCKED.
You can only solve puzzles whose item is UNLOCKED.
Solving a puzzle may unlock or lock other items — choose wisely.

{planning_desc}

Your task: choose ONE puzzle to solve next.
Consider the unlock/lock effects carefully — a wrong ordering
may permanently lock required items (deadend).

Return a JSON object with exactly these two keys:

"message": "<your reasoning about which puzzle to solve and why>",
"structured": {{
  "puzzle_id":  "<the puzzle_id you choose to solve next>",
  "item_type": "<the item type associated with the chosen puzzle — one of: {valid_types}>"
}}
""".replace("{valid_types}", _VALID_TYPES_STR),

    output_schema = [
        OutputField("puzzle_id", str, "the puzzle_id to solve next"),
        OutputField("item_type", str, "the item type of the chosen puzzle",
                    valid_values=_ALL_ITEM_TYPES),
    ],
)


# ---------------------------------------------------------------------------

OBSERVE_PLANNED_CLUE = Action(
    name       = "OBSERVE_PLANNED_CLUE",
    agent_role = "observer",

    context_keys     = ["room_desc"],
    prev_output_keys = ["SELECT_PUZZLE"],

    prompt_template = """You are observing an escape room. Here is everything you can see:

{room_desc}

{prev_input_block}

Your task: find the ONE real clue that corresponds to the instrument type
specified by the planner above.

A REAL clue has both:
  1. A math problem (a question involving numbers and calculation)
  2. A hint that describes the specified instrument type
     and a unit of measurement for that instrument

Ignore clues that refer to other instrument types or locked ones.
Select the single real clue for the specified instrument.

Return a JSON object with exactly these two keys:

"message": "<Include the FULL TEXT of the math problem and the FULL TEXT of the hint. The next agent needs the exact problem to solve it.>",
"structured": {{
  "item_id": <integer, the item id of the clue>,
  "clue":    "<exact full text of the clue>"
}}
""",

    preamble = """\
The planner has selected a puzzle. Here is the planner's structured JSON:
---
{SELECT_PUZZLE}
---

The JSON contains:
  "puzzle_id"  — the puzzle to solve
  "item_type"  — the instrument type to look for (e.g. thermometer, compass, clock, scale)

Find the clue whose hint refers to this instrument type.""",

    output_schema = [
        OutputField("item_id", int, "the item id of the clue"),
        OutputField("clue",    str, "exact full text of the clue"),
    ],
)


# ---------------------------------------------------------------------------
# Action registry — look up Action objects by name string
# ---------------------------------------------------------------------------

ACTION_REGISTRY: Dict[str, Action] = {
    "OBSERVE_CLUE":         OBSERVE_CLUE,
    "SOLVE_CLUE":           SOLVE_CLUE,
    "OBSERVE_ITEM":         OBSERVE_ITEM,
    "APPLY_DELTA":          APPLY_DELTA,
    "OBSERVE_PUZZLE":       OBSERVE_PUZZLE,
    "SELECT_PUZZLE":        SELECT_PUZZLE,
    "OBSERVE_PLANNED_CLUE": OBSERVE_PLANNED_CLUE,
}


def actions_from_names(names: List[str]) -> List[Action]:
    """Resolve a list of action name strings to Action objects."""
    actions = []
    for name in names:
        if name not in ACTION_REGISTRY:
            raise ValueError(f"Unknown action: '{name}'. Available: {list(ACTION_REGISTRY.keys())}")
        actions.append(ACTION_REGISTRY[name])
    return actions



# ---------------------------------------------------------------------------
# ActionResult
# ---------------------------------------------------------------------------

@dataclass
class ActionResult:
    """
    Everything produced by one action execution.

    structured_output : verified fields (never shown to agents in natural mode;
                        serialised as JSON and forwarded in structured mode)
    message           : natural language (forwarded in natural mode; logged but
                        not forwarded in structured mode)
    schema_errors     : field-level validation errors (empty = clean)
    raw_response      : full LLM response for trace logging
    injected          : True if an anomaly was applied to structured_output
    """
    action_name:       str
    agent_role:        str
    structured_output: Dict[str, Any]
    message:           str
    schema_errors:     Dict[str, str]
    raw_response:      str  = ""
    confidence:        float = 1.0
    injected:          bool  = False

    @property
    def has_schema_errors(self) -> bool:
        return bool(self.schema_errors)

    def to_dict(self) -> dict:
        return {
            "action_name":       self.action_name,
            "agent_role":        self.agent_role,
            "message":           self.message,
            "structured_output": self.structured_output,
            "schema_errors":     self.schema_errors,
            "confidence":        self.confidence,
            "injected":          self.injected,
        }
