"""
FM_annotate.py — Two-stage MAST Failure Mode annotation for EscapeRoom traces.

Given a directory with summary.csv and trace JSONs:
  Stage 1 (programmatic): Deterministic rules from summary.csv columns only
  Stage 2 (trace):        Reads trace JSONs to check reasoning vs output numbers

Both stages are fully programmatic — no LLM calls needed.

Writes back to summary.csv with two columns:
  Failure_mode(MAST)  — the FM-ID, "cascade", or ""
  FM_annotator        — "programmatic", "trace", or ""

Usage:
  python FM_annotate.py <target_dir>

Examples:
  python FM_annotate.py ../../output/traces/selection/gpt-4.1
  python FM_annotate.py ../../output/traces/run_20260401_160803_gpt-4.1_0.0_nightmare
"""

import argparse
import csv
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Optional

# ── Column names ─────────────────────────────────────────────────────────────

FM_COL = "Failure_mode(MAST)"
ANNOTATOR_COL = "FM_annotator"

_MERGE = {"OBSERVE_PLANNED_CLUE": "OBSERVE_CLUE"}


def _failed_action(row: dict) -> str:
    """Return the failed action name for new and historical summaries."""
    if "action_status" in row:
        if row.get("action_status") != "wrong":
            return ""
        return row.get("action_name", "")
    return row.get("failed_action", "")


def _trace_message(entry: dict) -> str:
    """Read LLM output text from new or historical traces."""
    return entry.get("output_message", entry.get("message", ""))


def _trace_structured(entry: dict) -> dict:
    """Read structured LLM output from new or historical traces."""
    return entry.get("output_structured", entry.get("structured", {}))


def classify_unit_tool_call(error_type: str = "", desc: str = "") -> str:
    """Map audited unit-tool failures to MAST failure modes."""
    error_codes = {code.strip() for code in str(error_type or "").split("+") if code.strip()}
    desc = str(desc or "")

    if "TC_WRONG_ARGUMENTS" in error_codes or "TC_WRONG_TOOL" in error_codes:
        return "FM-2.5"
    if error_codes & {
        "TC_REQUIRED_NOT_CALLED",
        "TC_DIMENSION_MISMATCH",
        "TC_INVALID_REQUEST",
        "TC_UNKNOWN_TOOL",
        "TC_UNKNOWN_UNIT",
        "TC_CALL_LIMIT_EXCEEDED",
        "TC_EXECUTION_ERROR",
    }:
        return "FM-1.1"
    if error_codes == {"TC_MISSING_MESSAGE"}:
        return "FM-3.2"

    if "wrong_tool" in desc or "wrong_value" in desc or "wrong_from_unit" in desc or "wrong_to_unit" in desc:
        return "FM-2.5"
    if "required unit conversion was not called" in desc or "incompatible" in desc or "finite number" in desc:
        return "FM-1.1"
    if "missing_tool_message" in desc:
        return "FM-3.2"
    return ""


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1: Programmatic (summary.csv only)
# ══════════════════════════════════════════════════════════════════════════════

def _build_failure_index(rows: list[dict]) -> dict[tuple, set[str]]:
    """Build (room_id, temperature, puzzle_id) → set of failed actions."""
    idx: dict[tuple, set[str]] = defaultdict(set)
    for r in rows:
        fa = _failed_action(r)
        if fa:
            key = (r["room_id"], r["temperature"], r["puzzle_id"])
            idx[key].add(_MERGE.get(fa, fa))
    return idx


def classify_programmatic(row: dict, failure_index: dict[tuple, set[str]]) -> str:
    """Return FM-ID, 'cascade', or '' (needs trace analysis)."""
    fa = _failed_action(row)
    if not fa or fa == "API_TIMEOUT":
        return ""

    desc = row.get("desc", "")
    puzzle_key = (row["room_id"], row["temperature"], row["puzzle_id"])
    upstream = failure_index.get(puzzle_key, set())
    merged_fa = _MERGE.get(fa, fa)

    if merged_fa == "SELECT_PUZZLE":
        if "already_solved" in desc:
            return "FM-1.3"
        if "puzzle_locked" in desc:
            return "FM-1.1"
        if "leads_to_deadend" in desc:
            return "FM-2.3"
        if "puzzle_not_found" in desc:
            return "FM-2.6"
        return ""

    # OBSERVE_CLUE — ambiguous: FM-4.1 vs FM-2.6 (needs trace)
    if merged_fa == "OBSERVE_CLUE":
        if "clue_select_wrong" in desc:
            return ""  # Stage 2
        return ""

    # SOLVE_CLUE
    if merged_fa == "SOLVE_CLUE":
        if "OBSERVE_CLUE" in upstream:
            return "FM-4.3"  # error propagation from upstream
        if "math_unit_wrong" in desc and "math_value_wrong" not in desc:
            return "FM-1.1"  # value correct, unit wrong = disobey spec
        # math_value_wrong (own) → ambiguous: FM-4.2 vs FM-2.6 (needs trace)
        return ""

    # OBSERVE_ITEM
    if merged_fa == "OBSERVE_ITEM":
        if "item_type_wrong" in desc:
            return ""  # ambiguous: FM-4.1 vs FM-2.6 (needs trace)
        if "item_state_wrong" in desc:
            return "FM-2.5"  # type correct, state wrong = ignored input
        return ""

    # APPLY_DELTA
    if merged_fa == "APPLY_DELTA":
        if upstream & {"OBSERVE_CLUE", "SOLVE_CLUE", "OBSERVE_ITEM"}:
            return "FM-4.3"  # error propagation from upstream
        # own error → ambiguous: FM-4.2 vs FM-2.5 (needs trace)
        return ""

    if merged_fa == "UNIT_TOOL_CALL":
        return classify_unit_tool_call(row.get("error_type", ""), desc)

    if merged_fa == "OBSERVE_PUZZLE":
        if "observer_disagrees" in desc:
            return "FM-3.3"
        return ""

    return ""


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2: Trace analysis (reads trace JSONs, still deterministic)
# ══════════════════════════════════════════════════════════════════════════════

def _load_trace(target_dir: Path, trace_filename: str) -> Optional[dict]:
    path = target_dir / trace_filename
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_trace_entry(trace_data: dict, puzzle_id: str, action: str) -> Optional[dict]:
    for entry in trace_data.get("trace", []):
        pid = entry.get("puzzle_id", "")
        short_pid = pid.split("_")[-1] if "_P" in pid else pid
        if short_pid == puzzle_id and entry.get("action") == action:
            return entry
    return None


def _extract_numbers(text: str) -> list[float]:
    """Extract all numbers from text, handling commas and negatives."""
    # Match integers, decimals, negative numbers, scientific notation
    # Exclude numbers that are part of IDs like "room_0001"
    raw = re.findall(r'(?<![a-zA-Z_])(-?\d[\d,]*\.?\d*(?:[eE][+-]?\d+)?)', text)
    nums = []
    for s in raw:
        try:
            nums.append(float(s.replace(",", "")))
        except ValueError:
            continue
    return nums


def _number_close(a: float, b: float, rel_tol: float = 1e-3) -> bool:
    """Check if two numbers are approximately equal."""
    if a == 0 and b == 0:
        return True
    try:
        return math.isclose(a, b, rel_tol=rel_tol)
    except (TypeError, ValueError):
        return False


def _reasoning_contains_answer(reasoning: str, gt_answer) -> bool:
    """Check if the reasoning text contains a number close to gt_answer."""
    try:
        gt = float(gt_answer)
    except (TypeError, ValueError):
        return False

    nums = _extract_numbers(reasoning)
    return any(_number_close(n, gt) for n in nums)


def _reasoning_contains_value(reasoning: str, value) -> bool:
    """Check if reasoning text contains a number close to value."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    nums = _extract_numbers(reasoning)
    return any(_number_close(n, v) for n in nums)


def classify_trace_observe_clue(
    row: dict, trace_data: dict,
) -> Optional[str]:
    """Classify OBSERVE_CLUE clue_select_wrong as FM-4.1 or FM-2.6.

    FM-2.6: reasoning identified the correct clue but output the wrong item_id
    FM-4.1: reasoning itself was fooled by the fake clue
    """
    action = _failed_action(row)
    # Handle both OBSERVE_CLUE and OBSERVE_PLANNED_CLUE
    entry = _find_trace_entry(trace_data, row["puzzle_id"], action)
    if not entry:
        return None

    reasoning = _trace_message(entry)
    gt_id = row.get("gt", "")

    # Check if reasoning mentions the correct item_id
    if gt_id and str(gt_id) in reasoning:
        return "FM-2.6"  # reasoning found correct clue, output was wrong
    else:
        return "FM-4.1"  # reasoning was fooled by distraction


def _find_puzzle(trace_data: dict, puzzle_id: str) -> dict:
    """Find puzzle definition from room data."""
    for p in trace_data.get("room", {}).get("puzzles", []):
        if p["puzzle_id"] == puzzle_id:
            return p
    return {}


def classify_trace_solve_clue(
    row: dict, trace_data: dict,
) -> Optional[str]:
    """Classify SOLVE_CLUE math_value_wrong.

    Priority:
      1. If unit also wrong AND agent's unit appears in problem text → FM-1.1
         (confused problem's unit with clue's target unit)
      2. If reasoning contains correct answer but output differs → FM-2.6
      3. Otherwise → FM-4.2 (reasoning error)
    """
    entry = _find_trace_entry(trace_data, row["puzzle_id"], "SOLVE_CLUE")
    if not entry:
        return None

    desc = row.get("desc", "")
    reasoning = _trace_message(entry)
    gt_answer = row.get("gt", "")

    # Check: unit wrong AND agent's unit comes from problem text
    if "math_unit_wrong" in desc:
        agent_unit = _trace_structured(entry).get("unit", "")
        if agent_unit:
            puzzle = _find_puzzle(trace_data, row["puzzle_id"])
            problem_text = puzzle.get("clue", {}).get("problem", "")
            if agent_unit.lower() in problem_text.lower():
                return "FM-1.1"  # confused problem unit with target unit

    # Standard: check reasoning vs output
    if _reasoning_contains_answer(reasoning, gt_answer):
        return "FM-2.6"  # reasoning had it right, output was wrong
    else:
        return "FM-4.2"  # reasoning itself was wrong


def classify_trace_observe_item(
    row: dict, trace_data: dict,
) -> Optional[str]:
    """Classify OBSERVE_ITEM errors.

    item_type_wrong:
      FM-2.6: reasoning identified the correct item but output the wrong one
      FM-4.1: reasoning was fooled by fake item
    item_state_wrong:
      FM-2.6: reasoning read the correct state but output differs
      FM-4.2: reasoning itself misread the state
    """
    entry = _find_trace_entry(trace_data, row["puzzle_id"], "OBSERVE_ITEM")
    if not entry:
        return None

    reasoning = _trace_message(entry)
    desc = row.get("desc", "")

    if "item_type_wrong" in desc:
        gt_type = row.get("gt", "")
        # Check if reasoning mentions the correct item type
        if gt_type and gt_type.lower() in reasoning.lower():
            return "FM-2.6"
        else:
            return "FM-4.1"  # fooled by fake item

    if "item_state_wrong" in desc:
        gt_state = row.get("gt", "")
        # For state: check if correct state value appears in reasoning
        # gt field for OBSERVE_ITEM has the type, gt_state is in the desc
        # Extract gt state from desc: "item_state_wrong(X!=Y)" → Y is ground truth
        import re as _re
        m = _re.search(r"item_state_wrong\(.+?!=(.+?)\)", desc)
        if m:
            gt_state_val = m.group(1)
            if _reasoning_contains_value(reasoning, gt_state_val):
                return "FM-2.6"
        return "FM-4.2"  # misread the state

    return None


def classify_trace_apply_delta(
    row: dict, trace_data: dict,
) -> Optional[str]:
    """Classify APPLY_DELTA own error as FM-2.5 or FM-4.2.

    FM-2.5: agent used different values than upstream provided
    FM-4.2: agent used correct upstream values but computed wrong
    """
    entry = _find_trace_entry(trace_data, row["puzzle_id"], "APPLY_DELTA")
    if not entry:
        return None

    reasoning = _trace_message(entry)

    # Get upstream correct values
    solve_entry = _find_trace_entry(trace_data, row["puzzle_id"], "SOLVE_CLUE")
    observe_entry = _find_trace_entry(trace_data, row["puzzle_id"], "OBSERVE_ITEM")

    if not solve_entry or not observe_entry:
        return None

    solve_answer = _trace_structured(solve_entry).get("answer")
    item_state = _trace_structured(observe_entry).get("item_state")

    # Check if the agent's reasoning mentions the correct upstream values
    used_solve = _reasoning_contains_value(reasoning, solve_answer)
    used_item = _reasoning_contains_value(reasoning, item_state)

    if used_solve and used_item:
        return "FM-4.2"   # used correct values but computed wrong
    else:
        return "FM-2.5"   # ignored/changed upstream values


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 alt: LLM annotator (optional, use --use-llm)
# ══════════════════════════════════════════════════════════════════════════════

def _build_solve_clue_prompt(row: dict, trace_entry: dict, puzzle: dict) -> str:
    reasoning = _trace_message(trace_entry)
    structured = _trace_structured(trace_entry)
    gt_answer = row.get("gt", "")
    return f"""You are classifying a multi-agent system failure.

A clue_solver agent was given a math problem and produced a wrong answer.

## Problem
{puzzle.get('clue', {}).get('problem', 'N/A')}

## Agent's reasoning (message):
{reasoning}

## Agent's structured output:
answer: {structured.get('answer', 'N/A')}
unit: {structured.get('unit', 'N/A')}

## Ground truth:
answer: {gt_answer}

## Task
Determine which failure mode applies:

**FM-1.1 (Disobey task specification)**: The agent's reasoning itself contains mathematical errors — it attempted the problem but computed incorrectly.

**FM-2.6 (Reasoning-action mismatch)**: The agent's reasoning arrives at the correct (or near-correct) answer, but the final structured output contains a DIFFERENT number. The reasoning and action are inconsistent.

Respond with ONLY one line in this format:
FM-X.X: <one-sentence justification>
"""


def _build_apply_delta_prompt(
    row: dict, trace_entry: dict,
    solve_entry: Optional[dict], observe_entry: Optional[dict],
) -> str:
    reasoning = _trace_message(trace_entry)
    structured = _trace_structured(trace_entry)
    gt_answer = row.get("gt", "")
    solve_out = _trace_structured(solve_entry) if solve_entry else {}
    observe_out = _trace_structured(observe_entry) if observe_entry else {}
    return f"""You are classifying a multi-agent system failure.

An item_manager agent received correct inputs from upstream agents but computed a wrong final answer.

## Upstream inputs (both verified correct):
- clue_solver answer: {solve_out.get('answer', 'N/A')} {solve_out.get('unit', '')}
- observer item_state: {observe_out.get('item_state', 'N/A')} {observe_out.get('item_unit', '')}

## Agent's reasoning (message):
{reasoning}

## Agent's structured output:
answer: {structured.get('answer', 'N/A')}

## Ground truth answer: {gt_answer}

## Task
Determine which failure mode applies:

**FM-2.5 (Ignored other agent's input)**: The item_manager used values DIFFERENT from what upstream agents provided (e.g., used a different clue answer or item state than what was passed to it).

**FM-1.1 (Disobey task specification)**: The item_manager used the correct upstream values but made a computation error (e.g., wrong arithmetic applying the delta to the item state).

Respond with ONLY one line in this format:
FM-X.X: <one-sentence justification>
"""


def _parse_fm_response(response: str) -> Optional[str]:
    m = re.match(r"(FM-\d+\.\d+)", response.strip())
    return m.group(1) if m else None


def classify_llm_solve_clue(row, trace_data, llm_call) -> Optional[str]:
    entry = _find_trace_entry(trace_data, row["puzzle_id"], "SOLVE_CLUE")
    if not entry:
        return None
    puzzle = _find_puzzle(trace_data, row["puzzle_id"])
    prompt = _build_solve_clue_prompt(row, entry, puzzle)
    response = llm_call(prompt)
    fm = _parse_fm_response(response)
    return fm if fm in {"FM-1.1", "FM-2.6"} else None


def classify_llm_apply_delta(row, trace_data, llm_call) -> Optional[str]:
    entry = _find_trace_entry(trace_data, row["puzzle_id"], "APPLY_DELTA")
    if not entry:
        return None
    solve_e = _find_trace_entry(trace_data, row["puzzle_id"], "SOLVE_CLUE")
    observe_e = _find_trace_entry(trace_data, row["puzzle_id"], "OBSERVE_ITEM")
    prompt = _build_apply_delta_prompt(row, entry, solve_e, observe_e)
    response = llm_call(prompt)
    fm = _parse_fm_response(response)
    return fm if fm in {"FM-2.5", "FM-1.1"} else None


def _make_llm_call(model: str = "gpt-4.1", temperature: float = 0.0):
    """Create an LLM call function using OpenAI API."""
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def call(prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=200,
        )
        return response.choices[0].message.content or ""

    return call


# ══════════════════════════════════════════════════════════════════════════════
# Unified annotator
# ══════════════════════════════════════════════════════════════════════════════

def annotate(target_dir: Path, use_llm: bool = False, llm_model: str = "gpt-4.1") -> dict:
    """Run Stage 1 + Stage 2 on target_dir/summary.csv.

    Stage 2 uses trace number-matching by default, or LLM if use_llm=True.

    Returns dict with n_programmatic, n_cascade, n_trace, n_unclassified.
    """
    csv_path = target_dir / "summary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No summary.csv in {target_dir}")

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    for col in [FM_COL, ANNOTATOR_COL]:
        if col not in fieldnames:
            fieldnames.append(col)

    # ── Stage 1: Programmatic ────────────────────────────────────────────
    failure_index = _build_failure_index(rows)

    n_prog = 0
    for row in rows:
        fm = classify_programmatic(row, failure_index)
        row[FM_COL] = fm
        if fm:
            row[ANNOTATOR_COL] = "programmatic"
            n_prog += 1
        else:
            row[ANNOTATOR_COL] = ""

    print(f"  Stage 1: {n_prog} programmatic")

    # ── Stage 2: Trace analysis (heuristic or LLM) ─────────────────────
    # Find rows that need trace analysis (empty FM but have a failure)
    needs_trace = []
    for i, row in enumerate(rows):
        fa = _failed_action(row)
        if not fa or fa == "API_TIMEOUT" or row.get(FM_COL):
            continue
        merged_fa = _MERGE.get(fa, fa)
        desc = row.get("desc", "")
        if merged_fa == "OBSERVE_CLUE" and "clue_select_wrong" in desc:
            needs_trace.append((i, "OBSERVE_CLUE"))
        elif merged_fa == "SOLVE_CLUE" and "math_value_wrong" in desc:
            needs_trace.append((i, "SOLVE_CLUE"))
        elif merged_fa == "OBSERVE_ITEM" and "item_type_wrong" in desc:
            needs_trace.append((i, "OBSERVE_ITEM"))
        elif merged_fa == "APPLY_DELTA" and "apply_delta_wrong" in desc:
            needs_trace.append((i, "APPLY_DELTA"))

    llm_call = _make_llm_call(model=llm_model) if use_llm else None
    stage2_mode = "LLM" if use_llm else "heuristic"

    n_trace = 0
    n_unclassified = 0
    if needs_trace:
        by_type = Counter(t for _, t in needs_trace)
        detail = "  ".join(f"{k}:{v}" for k, v in sorted(by_type.items()))
        print(f"  Stage 2 ({stage2_mode}): {len(needs_trace)} rows  [{detail}]")

        # Cache loaded traces by filename
        trace_cache: dict[str, Optional[dict]] = {}

        for idx, amb_type in needs_trace:
            row = rows[idx]
            fn = row.get("trace_filename", "")

            if fn not in trace_cache:
                trace_cache[fn] = _load_trace(target_dir, fn)
            trace_data = trace_cache[fn]

            if not trace_data:
                n_unclassified += 1
                continue

            if use_llm:
                # LLM-based classification (only for SOLVE_CLUE and APPLY_DELTA)
                if amb_type == "SOLVE_CLUE":
                    fm = classify_llm_solve_clue(row, trace_data, llm_call)
                elif amb_type == "APPLY_DELTA":
                    fm = classify_llm_apply_delta(row, trace_data, llm_call)
                else:
                    # OBSERVE_CLUE and OBSERVE_ITEM always use heuristic
                    if amb_type == "OBSERVE_CLUE":
                        fm = classify_trace_observe_clue(row, trace_data)
                    elif amb_type == "OBSERVE_ITEM":
                        fm = classify_trace_observe_item(row, trace_data)
                    else:
                        fm = None
            else:
                # Heuristic: number/text matching in reasoning
                if amb_type == "OBSERVE_CLUE":
                    fm = classify_trace_observe_clue(row, trace_data)
                elif amb_type == "SOLVE_CLUE":
                    fm = classify_trace_solve_clue(row, trace_data)
                elif amb_type == "OBSERVE_ITEM":
                    fm = classify_trace_observe_item(row, trace_data)
                elif amb_type == "APPLY_DELTA":
                    fm = classify_trace_apply_delta(row, trace_data)
                else:
                    fm = None

            if fm:
                rows[idx][FM_COL] = fm
                rows[idx][ANNOTATOR_COL] = "LLM" if use_llm else "programmatic"
                n_trace += 1
            else:
                n_unclassified += 1

    # ── Write back ───────────────────────────────────────────────────────
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "n_programmatic": n_prog,
        "n_trace": n_trace,
        "n_unclassified": n_unclassified,
    }
    print(f"  Result: {n_prog} programmatic, "
          f"{n_trace} trace, {n_unclassified} unclassified")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Two-stage MAST FM annotation for EscapeRoom traces"
    )
    parser.add_argument("target_dir", type=str,
                        help="Directory with summary.csv and trace JSONs")
    parser.add_argument("--use-llm", action="store_true",
                        help="Use LLM for Stage 2 instead of trace heuristic")
    parser.add_argument("--llm-model", type=str, default="gpt-4.1",
                        help="LLM model for Stage 2 (default: gpt-4.1)")
    args = parser.parse_args()

    target = Path(args.target_dir)
    if not target.is_absolute():
        target = Path(__file__).resolve().parent / target

    print(f"FM Annotation: {target.name}")
    print(f"  Stage 2 mode: {'LLM (' + args.llm_model + ')' if args.use_llm else 'trace heuristic'}")
    print("=" * 60)

    annotate(target, use_llm=args.use_llm, llm_model=args.llm_model)

    # Print final distribution
    csv_path = target / "summary.csv"
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    fm_counts = Counter(r.get(FM_COL, "") for r in rows if r.get(FM_COL))
    ann_counts = Counter(r.get(ANNOTATOR_COL, "") for r in rows if r.get(ANNOTATOR_COL))

    print(f"\nFM distribution:")
    for k, v in sorted(fm_counts.items()):
        print(f"  {k}: {v}")
    print(f"\nAnnotator distribution:")
    for k, v in sorted(ann_counts.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
