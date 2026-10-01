"""GSM-hard distractor factory."""

from __future__ import annotations

import random
from typing import Optional

FAKE_HINTS = [
    "Something that measures the passage of seasons awaits the answer, counted in moons.",
    "The answer belongs to an instrument that hears how loud the silence is, counted in whispers.",
    "Find what to add to a device that tastes the sweetness of the air, measured in grains.",
    "Something that weighs the brightness of a candle needs the answer, counted in lumens.",
    "The answer moves the needle of an instrument that tracks how fast the wind forgets, counted in sighs.",
    "An instrument that measures the depth of shadows depends on what you find, counted in shades.",
    "Something that counts how many times a door has been opened awaits the answer, measured in turns of the handle.",
    "Find what rises and falls with the mood of the room, measured in tremors.",
    "The answer tips the balance of something that judges how old the dust is, counted in layers.",
    "Something with a dial that points to yesterday needs the answer, counted in echoes.",
]


def create_fake_clue(clue_class, hint: Optional[str] = None):
    if hint is None:
        hint = "Hint: " + random.choice(FAKE_HINTS)
    return clue_class(is_fake=True, hint=hint, domain="gsm-hard")
