from __future__ import annotations

import re
from typing import Any, Callable, Dict, List

from src.eval.common_types import ParsedTextCheckpoint


# Matches ACTION: / STATE N: with optional markdown bold ** and optional leading - bullet
_SEP = r"\n[ \t]*[-*]*[ \t]*\*{0,2}(?:STATE\s+[0-9]+(?:\.[0-9]+)?\s*:|ACTION:|Plan summary:)"
ACTION_PATTERN = re.compile(
    r"\*{0,2}ACTION:\*{0,2}\s*(.*?)(?=" + _SEP + r"|\Z)",
    re.DOTALL,
)

STATE_PATTERN = re.compile(
    r"\*{0,2}STATE\s+([0-9]+(?:\.[0-9]+)?)\s*:\*{0,2}\s*(.*?)(?=" + _SEP + r"|\Z)",
    re.DOTALL,
)


def _first_line(text: str) -> str:
    """Return only the first non-empty line of text (strips multi-line bleed)."""
    for line in text.splitlines():
        line = line.strip().lstrip("-* ")
        if line:
            return line
    return text.strip()


def extract_actions(raw_output: str) -> List[str]:
    """
    Extract AoT+ ACTION strings in order.
    Only the first line of each match is kept — prevents STATE bleed from
    being included when the model uses non-standard formatting.
    """
    actions: List[str] = []
    for m in ACTION_PATTERN.finditer(raw_output):
        text = _first_line(m.group(1))
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