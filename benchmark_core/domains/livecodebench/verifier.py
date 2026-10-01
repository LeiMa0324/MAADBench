"""LiveCodeBench answer policy for amplified integer clue answers."""


def solve_answer_matches(predicted, gold) -> bool:
    try:
        return abs(float(predicted) - float(gold)) < 0.5
    except (TypeError, ValueError):
        return False


# The final reading may differ by one after unit-conversion rounding.
ITEM_ABSOLUTE_TOLERANCE = 1.0
