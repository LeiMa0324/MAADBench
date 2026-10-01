"""GSM-hard answer policy."""

import math
from typing import Any


def solve_answer_matches(predicted: Any, gold: Any) -> bool:
    try:
        return math.isclose(float(predicted), float(gold), rel_tol=1e-4)
    except (TypeError, ValueError):
        return False


ITEM_ABSOLUTE_TOLERANCE = None
