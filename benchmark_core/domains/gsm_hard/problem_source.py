"""GSM-hard problem source for EscapeRoom clues."""

from __future__ import annotations

from typing import Dict, List, Optional

_CACHE: Optional[List[Dict]] = None


def load_problems() -> List[Dict]:
    global _CACHE
    if _CACHE is None:
        from benchmark_core.domains.gsm_hard.dataloader import GSM_DataLoader
        dataset = GSM_DataLoader.load_from_huggingface(split="train")
        problems = []
        for index, item in enumerate(dataset):
            try:
                answer = float(str(item.answer).strip())
            except (TypeError, ValueError):
                continue
            problems.append({
                "problem_id": item.problem_id or f"gsmhard_train_{index:04d}",
                "problem": item.problem,
                "answer": answer,
                "domain": "gsm-hard",
                "metadata": {"split": "train", "source_index": index},
            })
        if not problems:
            raise ValueError("GSM-hard returned no problems with numeric answers")
        _CACHE = problems
    return _CACHE
