"""Utilities for assigning stable, room-scoped action identifiers."""

from __future__ import annotations

from typing import Any, Dict, List


def assign_action_ids(trace: List[Dict[str, Any]], room_id: str) -> None:
    """Append a unique numeric sequence to every action ID in trace order.

    Tool actions are materialized immediately before their APPLY_DELTA parent.
    Their ``parent_action_id`` is rewritten to the parent's indexed ID.
    """
    old_ids = [entry.get("action_id", "") for entry in trace]
    new_ids: List[str] = []

    for action_index, entry in enumerate(trace, start=1):
        puzzle_id = entry.get("puzzle_id", "")
        action_name = entry.get("action", "ACTION")
        base_id = old_ids[action_index - 1]
        if not base_id:
            parts = [part for part in (room_id, puzzle_id, action_name) if part]
            base_id = "_".join(parts)
        elif room_id and not base_id.startswith(f"{room_id}_"):
            base_id = f"{room_id}_{base_id}"

        entry["action_id"] = f"{base_id}_{action_index}"
        new_ids.append(entry["action_id"])

    for index, entry in enumerate(trace):
        if entry.get("action") != "UNIT_TOOL_CALL":
            continue
        old_parent_id = entry.get("parent_action_id", "")
        parent_index = next((
            candidate
            for candidate in range(index + 1, len(trace))
            if old_ids[candidate] == old_parent_id
            and trace[candidate].get("action") == "APPLY_DELTA"
            and trace[candidate].get("puzzle_id") == entry.get("puzzle_id")
            and trace[candidate].get("attempt") == entry.get("attempt")
        ), None)
        if parent_index is not None:
            entry["parent_action_id"] = new_ids[parent_index]
