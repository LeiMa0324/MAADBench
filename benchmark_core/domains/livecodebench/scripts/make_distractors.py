"""
make_distractors.py — build a pool of INVALID-Python distractor snippets.

The LiveCode variant's real clues are runnable Python functions ("what does this
return?"). To make the fake clues plausible in that setting, we sample real
functions from the dataset and corrupt each into code that does NOT compile
(broken syntax / indentation), so an Observer must distinguish a valid program
from a broken one.

Output: code_distractors.jsonl, one record per line:
    {"broken_code": <str>, "function_name": <str>, "call": <str>, "corruption": <str>}

Usage:
    python "benchmark_core/domains/livecodebench/scripts/make_distractors.py"
    python .../make_distractors.py --n 200 --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
SRC_JSONL = DATA_DIR / "execution_number_gt_medium_hard.jsonl"
OUT_JSONL = DATA_DIR / "code_distractors.jsonl"


def _is_invalid(code: str) -> bool:
    """True if the code fails to compile (SyntaxError/IndentationError/etc.)."""
    try:
        compile(code, "<distractor>", "exec")
        return False
    except Exception:
        return True


# --- corruption strategies (each returns corrupted code or None) ------------

def _drop_colon(code: str, rng: random.Random):
    lines = code.split("\n")
    idxs = [i for i, l in enumerate(lines) if l.rstrip().endswith(":")]
    if not idxs:
        return None
    i = rng.choice(idxs)
    lines[i] = lines[i].rstrip()[:-1]        # strip the trailing ':'
    return "\n".join(lines)


def _unbalance_bracket(code: str, rng: random.Random):
    positions = [m.start() for m in re.finditer(r"[)\]}]", code)]
    if not positions:
        return None
    p = rng.choice(positions)
    return code[:p] + code[p + 1:]           # delete one closing bracket


def _break_indent(code: str, rng: random.Random):
    lines = code.split("\n")
    idxs = [i for i, l in enumerate(lines) if l.startswith(("    ", "\t")) and l.strip()]
    if not idxs:
        return None
    i = rng.choice(idxs)
    lines[i] = lines[i].lstrip()             # dedent → IndentationError
    return "\n".join(lines)


def _stray_token(code: str, rng: random.Random):
    lines = code.split("\n")
    i = rng.randrange(len(lines))
    lines[i] = lines[i] + " $@?"             # illegal characters
    return "\n".join(lines)


def _mangle_keyword(code: str, rng: random.Random):
    for kw, bad in [("return ", "retunr "), ("def ", "fed "),
                    ("elif ", "eliff "), ("while ", "whille ")]:
        if kw in code:
            return code.replace(kw, bad, 1)
    return None


_STRATEGIES = [_drop_colon, _unbalance_bracket, _break_indent, _stray_token, _mangle_keyword]


def corrupt(code: str, rng: random.Random):
    """Apply strategies until one yields code that does NOT compile."""
    strats = _STRATEGIES[:]
    rng.shuffle(strats)
    for s in strats:
        out = s(code, rng)
        if out and out != code and _is_invalid(out):
            return out, s.__name__.lstrip("_")
    return None, None


def main():
    ap = argparse.ArgumentParser(description="Build invalid-Python distractor pool.")
    ap.add_argument("--src", type=str, default=str(SRC_JSONL))
    ap.add_argument("--out", type=str, default=str(OUT_JSONL))
    ap.add_argument("--n", type=int, default=0, help="Max distractors (0 = all source rows).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    src = Path(args.src)
    rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
    rng.shuffle(rows)
    if args.n:
        rows = rows[: args.n * 2]            # oversample; some corruptions may fail

    kept, skipped = 0, 0
    seen = set()
    with open(args.out, "w", encoding="utf-8") as f:
        for rec in rows:
            code = (rec.get("code") or "").strip()
            if not code or code in seen:
                continue
            broken, how = corrupt(code, rng)
            if broken is None:
                skipped += 1
                continue
            seen.add(code)
            f.write(json.dumps({
                "broken_code": broken,
                "function_name": rec.get("function_name", ""),
                "call": rec.get("input", ""),
                "corruption": how,
            }, ensure_ascii=False) + "\n")
            kept += 1
            if args.n and kept >= args.n:
                break

    print(f"[distractors] wrote {kept} invalid-python distractors -> {args.out} "
          f"({skipped} could not be corrupted)")


if __name__ == "__main__":
    main()
