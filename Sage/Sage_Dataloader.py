"""
sage_dataloader.py — Data generation and loading for the SAGE benchmark.

Usage:
    generator = SAGEDataGenerator(seed=42)
    loader = generator.generate(easy=10, medium=5, hard=3)

    # Iterate
    for item in loader:
        print(item.e_id, item.level, item.ori_expression, item.ground_truth)

    # Filter by level
    easy_items = loader.by_level("easy")

    # Export
    loader.to_csv("sage_data.csv")

    # Reload from CSV
    loader2 = SAGEDataLoader.from_csv("sage_data.csv")
"""

import csv
import uuid
from dataclasses import dataclass, field, asdict
from typing import List, Optional


# ── Data Item ─────────────────────────────────────────────────────────────────

@dataclass
class SAGE_Expr:
    e_id: str                   # unique expression id, e.g. "easy_0000"
    level: str                  # "easy" | "medium" | "hard"
    ori_expression: str         # unsimplified expression string
    ground_truth: str           # fully simplified expression string
    k_steps: int                # number of simplification steps
    rules_used: List[str]       # list of rule names applied in ground-truth trace

    @staticmethod
    def _strip_outer_parens(s: str) -> str:
        s = s.strip()
        if s.startswith("(") and s.endswith(")"):
            return s[1:-1].strip()
        return s

    def simple_evaluate(self, prediction: str) -> bool:
        pred = self._strip_outer_parens(prediction)
        gt   = self._strip_outer_parens(self.ground_truth)
        return pred == gt

    def to_dict(self) -> dict:
        return {
            "e_id": self.e_id,
            "level": self.level,
            "ori_expression": self.ori_expression,
            "ground_truth": self.ground_truth,
            "k_steps": self.k_steps,
            "rules_used": "|".join(self.rules_used),  # pipe-separated for CSV
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SAGE_Expr":
        rules = d.get("rules_used", "")
        return cls(
            e_id=d["e_id"],
            level=d["level"],
            ori_expression=d["expression"],
            ground_truth=d["ground_truth"],
            k_steps=int(d.get("k_steps", 0)),
            rules_used=rules.split("|") if rules else [],
        )


# ── DataLoader ────────────────────────────────────────────────────────────────

class SAGEDataLoader:
    """
    Stores SAGEItem records and provides iteration, filtering, and CSV I/O.
    """

    CSV_FIELDS = ["e_id", "level", "ori_expression", "ground_truth", "k_steps", "rules_used"]

    def __init__(self, items: Optional[List[SAGE_Expr]] = None):
        self._items: List[SAGE_Expr] = items or []

    # ── Core access ────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __getitem__(self, idx):
        return self._items[idx]

    def add(self, item: SAGE_Expr) -> None:
        self._items.append(item)

    def by_level(self, level: str) -> List[SAGE_Expr]:
        """Return all items matching the given difficulty level."""
        level = level.lower()
        return [item for item in self._items if item.level == level]

    def get(self, e_id: str) -> Optional[SAGE_Expr]:
        """Return a single item by its e_id, or None if not found."""
        for item in self._items:
            if item.e_id == e_id:
                return item
        return None

    def summary(self) -> dict:
        """Return count breakdown by level."""
        counts = {"easy": 0, "medium": 0, "hard": 0, "total": len(self._items)}
        for item in self._items:
            if item.level in counts:
                counts[item.level] += 1
        return counts



    # ── CSV I/O ────────────────────────────────────────────────────────────────

    def to_csv(self, path: str) -> None:
        """Save all items to a CSV file."""
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDS)
            writer.writeheader()
            for item in self._items:
                writer.writerow(item.to_dict())
        print(f"Saved {len(self._items)} items → {path}")

    @classmethod
    def from_csv(cls, path: str) -> "SAGEDataLoader":
        """Load items from a previously saved CSV file."""
        items = []
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                items.append(SAGE_Expr.from_dict(row))
        loader = cls(items)
        print(f"Loaded {len(items)} items ← {path}")
        return loader

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"SAGEDataLoader("
            f"total={s['total']}, "
            f"easy={s['easy']}, "
            f"medium={s['medium']}, "
            f"hard={s['hard']})"
        )



# ── CLI / Demo ────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n--- Reload from CSV ---")
    loader2 = SAGEDataLoader.from_csv("tasks.csv")
    print(loader2)

    print("\n--- Easy items only ---")
    for item in loader2.by_level("easy"):
        print(f"  {item.e_id}: {item.ori_expression}")


