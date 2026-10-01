"""
make_chains.py — build dependent 2-chain problems with executed ground truth.

Each chain links two dataset programs:

    STEP 1: run program A on its call  -> a
    STEP 2: in program B's call, replace one integer argument with a -> run B -> gt

Ground truth is obtained by ACTUALLY EXECUTING the chain (these are runnable
functions), in a sandboxed namespace with a timeout. A chain is kept only if:
  * B's call has an integer argument to substitute,
  * B runs cleanly on the substituted call and returns a number,
  * substituting `a` changes B's output vs the original (so step 1 truly matters).

Output: code_chains.jsonl, one record per line:
    {code_a, call_a, a, code_b, call_b_template, slot, gt, function_a, function_b}

Usage:
    python "benchmark_core/domains/livecodebench/scripts/make_chains.py" --n 200 --seed 0
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import random
import signal
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
SRC_JSONL = DATA_DIR / "execution_number_gt_medium_hard.jsonl"
OUT_JSONL = DATA_DIR / "code_chains.jsonl"
OUT_DISTRACTORS = DATA_DIR / "code_chain_distractors.jsonl"

# Imports most LiveCodeBench solutions rely on.
_PRELUDE = (
    "import math, collections, heapq, bisect, itertools, functools, re, string, operator\n"
    "from typing import *\n"
    "from collections import *\n"
    "from functools import *\n"
    "from itertools import *\n"
)


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def _run(code: str, call: str, timeout: int = 2):
    """Execute `code` then evaluate `call` in a fresh namespace, with a timeout."""
    ns: dict = {}
    exec(_PRELUDE, ns)
    exec(code, ns)
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(timeout)
    try:
        return eval(call, ns)
    finally:
        signal.alarm(0)


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _int_kwargs(call: str):
    """Return (Call node ok?, [(argname, intvalue), ...]) for integer kwargs."""
    try:
        tree = ast.parse(call, mode="eval")
    except SyntaxError:
        return []
    c = tree.body
    if not isinstance(c, ast.Call):
        return []
    slots = []
    for kw in c.keywords:
        if not kw.arg:
            continue
        v = kw.value
        val = None
        if isinstance(v, ast.Constant) and isinstance(v.value, int) and not isinstance(v.value, bool):
            val = v.value
        elif (isinstance(v, ast.UnaryOp) and isinstance(v.op, ast.USub)
              and isinstance(v.operand, ast.Constant) and isinstance(v.operand.value, int)):
            val = -v.operand.value
        if val is not None:
            slots.append((kw.arg, val))
    return slots


def _substitute(call: str, argname: str, newval):
    tree = ast.parse(call, mode="eval")
    for kw in tree.body.keywords:
        if kw.arg == argname:
            kw.value = ast.Constant(value=newval)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _template_call(call: str, argname: str) -> str:
    """Render B's call with the slot shown as the placeholder `a`."""
    tree = ast.parse(call, mode="eval")
    for kw in tree.body.keywords:
        if kw.arg == argname:
            kw.value = ast.Name(id="a")
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def build_chain(rec_a: dict, rec_b: dict, rng: random.Random):
    """Try to build one dependent chain from A (step 1) and B (step 2)."""
    a = rec_a.get("gt_number")
    if a is None:
        return None
    slots = _int_kwargs(rec_b.get("input", ""))
    if not slots:
        return None
    argname, orig = rng.choice(slots)
    if a == orig:
        return None                                   # substitution would be a no-op
    code_b = rec_b.get("code", "")
    call_b = rec_b.get("input", "")
    try:
        r_sub = _run(code_b, _substitute(call_b, argname, a))
        r_orig = _run(code_b, call_b)
    except Exception:
        return None
    if not (_is_number(r_sub) and _is_number(r_orig)):
        return None
    if r_sub == r_orig:
        return None                                   # arg doesn't affect output → step 1 irrelevant
    return {
        "code_a": rec_a.get("code", ""),
        "call_a": rec_a.get("input", ""),
        "a": a,
        "code_b": code_b,
        "call_b_template": _template_call(call_b, argname),
        "slot": argname,
        "gt": r_sub if isinstance(r_sub, int) else float(r_sub),
        "function_a": rec_a.get("function_name", ""),
        "function_b": rec_b.get("function_name", ""),
    }


def build_distractor_chain(rec_a: dict, rec_b: dict, rng: random.Random, corrupt_fn):
    """A chain-shaped fake: same two-program structure, but one program is broken."""
    slots = _int_kwargs(rec_b.get("input", ""))
    if not slots:
        return None
    argname, _ = rng.choice(slots)
    code_a, call_a = rec_a.get("code", ""), rec_a.get("input", "")
    code_b, call_b = rec_b.get("code", ""), rec_b.get("input", "")
    if not code_a or not code_b:
        return None
    which = rng.choice(["a", "b"])
    broken, how = corrupt_fn(code_a if which == "a" else code_b, rng)
    if broken is None:
        return None
    if which == "a":
        code_a = broken
    else:
        code_b = broken
    return {
        "code_a": code_a,
        "call_a": call_a,
        "code_b": code_b,
        "call_b_template": _template_call(call_b, argname),
        "slot": argname,
        "broken": which,
        "corruption": how,
    }


def main():
    ap = argparse.ArgumentParser(description="Build dependent 2-chain problems with executed gt.")
    ap.add_argument("--src", default=str(SRC_JSONL))
    ap.add_argument("--out", default=str(OUT_JSONL))
    ap.add_argument("--n", type=int, default=200, help="Target number of chains.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = [json.loads(l) for l in open(args.src, encoding="utf-8") if l.strip()]
    with_slot = [r for r in rows if _int_kwargs(r.get("input", ""))]
    print(f"[chains] {len(rows)} source rows; {len(with_slot)} have an integer arg to substitute")

    chains, attempts = [], 0
    # Try random (A, B) pairs until we hit the target or run out of budget.
    budget = args.n * 60
    while len(chains) < args.n and attempts < budget:
        attempts += 1
        rec_a = rng.choice(rows)
        rec_b = rng.choice(with_slot)
        ch = build_chain(rec_a, rec_b, rng)
        if ch:
            chains.append(ch)

    with open(args.out, "w", encoding="utf-8") as f:
        for ch in chains:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")
    gts = [c["gt"] for c in chains]
    rng_str = f"[{min(gts)}, {max(gts)}]" if gts else "n/a"
    print(f"[chains] wrote {len(chains)} chains -> {args.out}  (gt range {rng_str}, {attempts} attempts)")

    # --- chain-shaped distractors (same structure, one program broken) ------
    from make_distractors import corrupt
    dist, d_attempts = [], 0
    while len(dist) < args.n and d_attempts < budget:
        d_attempts += 1
        d = build_distractor_chain(rng.choice(rows), rng.choice(with_slot), rng, corrupt)
        if d:
            dist.append(d)
    with open(OUT_DISTRACTORS, "w", encoding="utf-8") as f:
        for d in dist:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"[chains] wrote {len(dist)} chain distractors -> {OUT_DISTRACTORS}")


if __name__ == "__main__":
    main()
