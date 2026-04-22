from __future__ import annotations

import re
from typing import Any, Callable, Dict, List

from src.eval.common_types import ParsedTextCheckpoint


ACTION_PATTERN = re.compile(
    r"ACTION:\s*(.*?)(?=\n(?:STATE\s+[0-9]+(?:\.[0-9]+)?\s*:|ACTION:|Plan summary:)|\Z)",
    re.DOTALL,
)

STATE_PATTERN = re.compile(
    r"STATE\s+([0-9]+(?:\.[0-9]+)?)\s*:\s*(.*?)(?=\n(?:STATE\s+[0-9]+(?:\.[0-9]+)?\s*:|ACTION:|Plan summary:)|\Z)",
    re.DOTALL,
)


def extract_actions(raw_output: str) -> List[str]:
    """
    Extract AoT+ ACTION strings in order.
    """
    actions: List[str] = []
    for m in ACTION_PATTERN.finditer(raw_output):
        text = m.group(1).strip()
        if text:
            actions.append(text)
    return actions


def extract_state_entries(raw_output: str) -> List[tuple[str, str]]:
    """
    Extract AoT+ STATE entries in order.
    """
    entries: List[tuple[str, str]] = []
    for m in STATE_PATTERN.finditer(raw_output):
        state_id = m.group(1).strip()
        state_text = m.group(2).strip()
        if state_text:
            entries.append((state_id, state_text))
    return entries


def build_text_checkpoints(
    raw_output: str,
    parse_state_fn: Callable[[str], Dict[str, Any]],
) -> List[ParsedTextCheckpoint]:
    """
    Build parsed checkpoints from AoT+ STATE entries.

    Alignment:
        - STATE 0 -> action_index 0
        - later states align by textual order
    """
    entries = extract_state_entries(raw_output)
    checkpoints: List[ParsedTextCheckpoint] = []

    action_index = 0
    for i, (state_id, state_text) in enumerate(entries):
        if i == 0 and state_id == "0":
            current_index = 0
        else:
            action_index += 1
            current_index = action_index

        try:
            parsed = parse_state_fn(state_text)
            checkpoints.append(
                ParsedTextCheckpoint(
                    state_id=state_id,
                    state_text=state_text,
                    action_index=current_index,
                    parsed_state=parsed,
                    parse_error=None,
                )
            )
        except Exception as e:
            checkpoints.append(
                ParsedTextCheckpoint(
                    state_id=state_id,
                    state_text=state_text,
                    action_index=current_index,
                    parsed_state=None,
                    parse_error=str(e),
                )
            )

    return checkpoints