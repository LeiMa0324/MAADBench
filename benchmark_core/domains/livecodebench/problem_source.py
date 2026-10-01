"""LiveCodeBench execution problem source for EscapeRoom clues."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

# Default: the dependent 2-chain problems (executed ground truth).
DEFAULT_JSONL = (
    Path(__file__).resolve().parent
    / "data"
    / "code_chains.jsonl"
)

_PROBLEMS_CACHE: Optional[List[dict]] = None


# --- Amplification --------------------------------------------------------
# The executed result is transformed to a LARGE number (>= 6 digits), and that
# transformed value becomes the clue's ground-truth answer (delta). Because the
# verifier derives all ground truth from clue.answer, the amplified result is the
# new correct answer everywhere (SOLVE_CLUE, APPLY_DELTA).
#
# Transform: amplify(n) = n * AMPLIFY_FACTOR + AMPLIFY_OFFSET
# The additive OFFSET (not multiplication) is what guarantees a large result even
# when the raw output is 0 or negative — the LiveCodeBench outputs here range from
# -1 to 20, so a plain multiply would leave zeros at 0 and everything tiny.
AMPLIFY_FACTOR = 1            # optional scaling before the offset (1 = none)
AMPLIFY_OFFSET = 1_000_000   # 7-digit shift → every answer has >= 6 digits


def amplify(n):
    """Map any executed value to a large (>= 6-digit) ground-truth number."""
    return int(n * AMPLIFY_FACTOR + AMPLIFY_OFFSET)


def _amplify_instruction() -> str:
    """Human-readable rule that matches `amplify`, so the two never drift apart."""
    if AMPLIFY_FACTOR == 1:
        return f"Then add {AMPLIFY_OFFSET} to that value; the result is your final answer."
    return (f"Then multiply that value by {AMPLIFY_FACTOR} and add {AMPLIFY_OFFSET}; "
            "the result is your final answer.")


def format_chain_note(code_a: str, call_a: str, code_b: str,
                      call_b_template: str, slot: str) -> str:
    """Render a dependent 2-chain as clue text (shared by real clues and distractors).

    Keeping real and fake clues on the SAME layout means the only difference is
    whether the code is valid — no structural tell.
    """
    return (
        "This clue is a chain of two Python programs; run them in order.\n\n"
        "FIRST program — run it on its call and let a = the value it returns:\n\n"
        f"{(code_a or '').strip()}\n\n"
        f"Call: {(call_a or '').strip()}\n\n"
        f"SECOND program — replace the argument `{slot}` in its call with a (the value "
        "from the FIRST program), then run it. The value it returns is the clue's result:\n\n"
        f"{(code_b or '').strip()}\n\n"
        f"Call: {(call_b_template or '').strip()}"
    )


def _format_problem(rec: dict) -> str:
    """Render one chain record as the clue's problem text."""
    return format_chain_note(
        rec.get("code_a", ""), rec.get("call_a", ""),
        rec.get("code_b", ""), rec.get("call_b_template", ""), rec.get("slot", ""),
    )


def _read_problems(path: Path) -> List[dict]:
    """Load the 2-chain JSONL into the {'problem','answer'} shape Clue expects.

    The stored ``answer`` is amplify(chain gt), so it is the ground truth: the
    verifier derives SOLVE_CLUE and APPLY_DELTA ground truth from clue.answer.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Chain dataset not found: {path}\n"
            "Generate it first with domains/livecodebench/scripts/make_chains.py"
        )
    problems: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            gt = rec.get("gt")
            if gt is None:
                continue
            problems.append({
                "problem": _format_problem(rec),
                "answer": amplify(gt),
                "domain": "livecodebench",
                "problem_id": str(rec.get("problem_id", rec.get("id", len(problems)))),
                "metadata": {
                    "function_a": rec.get("function_a"),
                    "function_b": rec.get("function_b"),
                    "raw_ground_truth": gt,
                    "amplify_factor": AMPLIFY_FACTOR,
                    "amplify_offset": AMPLIFY_OFFSET,
                },
            })
    if not problems:
        raise ValueError(f"No usable chain problems found in {path}")
    return problems


def load_problems(jsonl_path: Optional[str | Path] = None) -> List[dict]:
    """Load and cache normalized LiveCodeBench clue records."""
    global _PROBLEMS_CACHE
    if _PROBLEMS_CACHE is None or jsonl_path is not None:
        path = Path(jsonl_path) if jsonl_path else DEFAULT_JSONL
        _PROBLEMS_CACHE = _read_problems(path)
    return _PROBLEMS_CACHE
