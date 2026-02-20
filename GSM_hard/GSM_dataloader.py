"""
GSM-hard Dataset Loader
Loads and processes problems from the GSM-hard dataset
HuggingFace: reasoning-machines/gsm-hard

Note:
- GSM-hard is NOT the Hendrycks MATH dataset.
- It typically provides fields like: problem/question, solution, answer.
- This loader maps GSM-hard into the existing GSKProblem schema for compatibility.
"""

import json
import re
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import os


@dataclass
class GSM_problem:
    """A single GSM-hard problem (kept compatible with the previous schema)."""
    problem_id: str
    problem: str
    solution: str
    answer: str  # Final answer (cleaned)
    level: int  # Difficulty 1-5
    type: str  # algebra, geometry, etc.

    def get_difficulty(self) -> str:
        """Convert level to difficulty label"""
        if self.level <= 2:
            return "easy"
        elif self.level <= 3:
            return "medium"
        else:
            return "hard"

class GSM_DataLoader:
    """Loader for GSM-hard dataset."""

    # Legacy topic list from the Hendrycks MATH loader (not used by GSM-hard)
    TYPES = ["algebra", "counting_and_probability", "geometry",
             "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]

    @staticmethod
    def load_from_huggingface(split: str = "test", max_samples: Optional[int] = None) -> List[GSM_problem]:
        """
        Load MATH dataset from Hugging Face

        Args:
            split: "train" or "test"
            max_samples: Maximum number of samples to load

        Returns:
            List of MathProblem objects
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")

        print(f"Loading GSM-hard dataset ({split} split)...")
        dataset = load_dataset("reasoning-machines/gsm-hard")
        ds = dataset["train"]  # 或 "train"
        # print(type(ds))
        # print("len:", len(ds))
        # print("columns:", ds.column_names)
        # import pandas as pd
        # sample = ds.select(range(5)).to_pandas()
        # print(sample)

        if isinstance(dataset, dict) and split in dataset:
            ds = dataset[split]
        else:
            # Fallback: many HF datasets expose splits as dataset["train"], dataset["test"], etc.
            try:
                ds = dataset[split]
            except Exception:
                # If split is missing, default to test
                ds = dataset["test"] if "test" in dataset else next(iter(dataset.values()))

        problems = []
        for idx, item in enumerate(ds):
            if max_samples and idx >= max_samples:
                break

            # GSM-hard commonly uses keys like: problem/question, solution, answer
            problem_text = item.get("input")
            solution_text = item.get("code")

            # Prefer the dataset-provided final answer if available
            raw_answer = item.get("target")
            if raw_answer is None:
                answer = GSM_DataLoader._extract_answer(solution_text)
            else:
                answer = str(raw_answer).strip()

            # GSM-hard does not provide MATH-style (level/type). Keep compatibility defaults.
            level = 5
            problem_type = "gsm-hard"

            problems.append(GSM_problem(
                problem_id=f"gsmhard_{split}_{idx:04d}",
                problem=problem_text,
                solution=solution_text,
                answer=answer,
                level=level,
                type=problem_type
            ))

        print(f"Loaded {len(problems)} problems")
        return problems

    @staticmethod
    def _extract_answer(solution: str) -> str:
        """
        Extract the final answer from solution text

        MATH dataset answers are in \\boxed{...} format
        """
        # 1) GSM8K-style: '#### 42'
        m = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", solution)
        if m:
            return m.group(1).strip()

        # 2) MATH-style boxed answer (sometimes appears in derived sets)
        m = re.search(r"\\boxed\{([^}]+)\}", solution)
        if m:
            return m.group(1).strip()

        # 3) Common explicit pattern: 'The answer is X'
        m = re.search(r"(?:answer|result|solution)\s+is:?\s*([^\n\.]+)", solution, re.IGNORECASE)
        if m:
            return m.group(1).strip()

        # 4) Fallback: last number in the solution text
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", solution)
        return nums[-1].strip() if nums else ""

    @staticmethod
    def filter_by_difficulty(problems: List[GSM_problem],
                             levels: List[int]) -> List[GSM_problem]:
        """
        Filter problems by difficulty level

        Args:
            problems: List of problems
            levels: List of levels to keep (1-5)

        Returns:
            Filtered list
        """
        return [p for p in problems if p.level in levels]

    @staticmethod
    def filter_by_type(problems: List[GSM_problem],
                       types: List[str]) -> List[GSM_problem]:
        """
        Filter problems by mathematical topic

        Args:
            problems: List of problems
            types: List of types to keep

        Returns:
            Filtered list
        """
        return [p for p in problems if p.type in types]

    @staticmethod
    def filter_numerical_answers(problems: List[GSM_problem]) -> List[GSM_problem]:
        """
        Keep only problems with numerical answers (easier to verify)

        Returns:
            Problems with numeric answers
        """
        filtered = []
        for p in problems:
            # Try to parse answer as number
            try:
                # Remove common LaTeX commands
                answer_cleaned = p.answer.replace('\\', '').replace('$', '').replace(',', '')
                # Try to evaluate
                value = eval(answer_cleaned)
                if isinstance(value, (int, float)):
                    filtered.append(p)
            except:
                # Skip problems with non-numerical answers
                continue

        return filtered

    @staticmethod
    def sample_stratified(problems: List[GSM_problem],
                          n: int,
                          by: str = "level") -> List[GSM_problem]:
        """
        Sample n problems stratified by difficulty or type

        Args:
            problems: List of problems
            n: Total number to sample
            by: "level" or "type"

        Returns:
            Stratified sample
        """
        import random
        random.seed(42)

        # Group by stratification key
        from collections import defaultdict
        groups = defaultdict(list)

        for p in problems:
            key = p.level if by == "level" else p.type
            groups[key].append(p)

        # Sample proportionally from each group
        sampled = []
        per_group = n // len(groups)

        for group_problems in groups.values():
            if len(group_problems) <= per_group:
                sampled.extend(group_problems)
            else:
                sampled.extend(random.sample(group_problems, per_group))

        # Fill remaining slots randomly
        remaining = n - len(sampled)
        if remaining > 0:
            available = [p for p in problems if p not in sampled]
            sampled.extend(random.sample(available, min(remaining, len(available))))

        return sampled[:n]


    @staticmethod
    def print_statistics(problems: List[GSM_problem]):
        """Print dataset statistics"""
        from collections import Counter

        print("\n" + "=" * 60)
        print("GSM-hard Dataset Statistics")
        print("=" * 60)

        print(f"\nTotal problems: {len(problems)}")

        # By difficulty level
        level_counts = Counter(p.level for p in problems)
        print("\nBy Level:")
        for level in sorted(level_counts.keys()):
            count = level_counts[level]
            pct = count / len(problems) * 100
            print(f"  Level {level}: {count} ({pct:.1f}%)")

        # By type
        type_counts = Counter(p.type for p in problems)
        print("\nBy Type:")
        for ptype in sorted(type_counts.keys()):
            count = type_counts[ptype]
            pct = count / len(problems) * 100
            print(f"  {ptype}: {count} ({pct:.1f}%)")

        # By difficulty label
        diff_counts = Counter(p.get_difficulty() for p in problems)
        print("\nBy Difficulty:")
        for diff in ["easy", "medium", "hard"]:
            count = diff_counts.get(diff, 0)
            pct = count / len(problems) * 100 if len(problems) > 0 else 0
            print(f"  {diff}: {count} ({pct:.1f}%)")


    @staticmethod
    def simplify_problem(problem: str) -> str:
        """
        Simplify LaTeX and make more agent-friendly

        Some MATH problems have complex LaTeX that confuses agents
        """
        # Remove excessive LaTeX formatting
        simplified = problem

        # Convert common LaTeX to text
        replacements = {
            r'\\frac\{(\d+)\}\{(\d+)\}': r'\1/\2',  # Fractions
            r'\\cdot': '*',  # Multiplication
            r'\\times': '*',
            r'\\div': '/',
            r'\\le': '<=',
            r'\\ge': '>=',
            r'\\ne': '!=',
            r'\\sqrt\{([^}]+)\}': r'sqrt(\1)',
        }

        for pattern, replacement in replacements.items():
            try:
                simplified = re.sub(pattern, replacement, simplified)
            except re.error as e:
                raise ValueError(f"Regex failed. pattern={pattern!r}, replacement={replacement!r}") from e

        # Remove dollar signs
        simplified = simplified.replace('$', '')

        return simplified

    @staticmethod
    def extract_numerical_answer(answer_str: str) -> Optional[float]:
        """
        Extract numerical value from answer string

        Returns None if answer is not numerical
        """
        try:
            # Clean LaTeX
            cleaned = answer_str.replace('\\', '').replace('$', '').replace(',', '').strip()

            # Handle fractions
            if '/' in cleaned:
                parts = cleaned.split('/')
                if len(parts) == 2:
                    return float(parts[0]) / float(parts[1])

            # Direct evaluation
            value = eval(cleaned)
            if isinstance(value, (int, float)):
                return float(value)
        except:
            pass

        return None

    @staticmethod
    def simple_evaluate( ground_truth, predicted):
        is_correct = False
        if predicted is not None and ground_truth is not None:
            try:
                pred_float = float(predicted)
                if abs(pred_float - ground_truth) < 0.01:
                    is_correct = True
            except (ValueError, TypeError):
                pass
        return is_correct

if __name__ == "__main__":
    # Example usage

    print("Testing GSM-hard Dataset Loader\n")

    # Load a small sample
    try:
        problems = GSM_DataLoader.load_from_huggingface(split="test")

        # Print statistics
        GSM_DataLoader.print_statistics(problems)

        # Filter to easier problems with numerical answers
        print("\n" + "=" * 60)
        print("Filtering for experiments")
        print("=" * 60)

        # Sample 30 problems
        sample = GSM_DataLoader.sample_stratified(problems, n=30, by="level")
        print(f"\nStratified sample: {len(sample)} problems")

        # Show examples
        print("\n" + "=" * 60)
        print("Example Problems")
        print("=" * 60)

        for i, p in enumerate(sample[:3]):
            print(f"\nProblem {i + 1} (Level {p.level}, {p.type}):")
            print(f"  Question: {p.problem[:100]}...")
            print(f"  Answer: {p.answer}")

        # Convert to TaskData format
        print("\n" + "=" * 60)
        print("Converting to TaskData format")
        print("=" * 60)


    except Exception as e:
        print(f"Error: {e}")
        print("\nTo use GSM-hard dataset, install:")
        print("  pip install datasets")