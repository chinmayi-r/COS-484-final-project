from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List

from src.eval.common_types import ParsedJsonCheckpoint


ACTION_PATTERN = re.compile(
    r"ACTION:\s*(.*?)(?=\n(?:REASONING:|STATE_JSON:|ACTION:)|\Z)",
    re.DOTALL,
)

STATE_JSON_PATTERN = re.compile(
    r"STATE_JSON:\s*```json\s*(.*?)\s*```",
    re.DOTALL | re.IGNORECASE,
)


def extract_actions(raw_output: str) -> List[str]:
    """
    Extract ACTION strings from JSON-style model output.
    """
    actions: List[str] = []
    for m in ACTION_PATTERN.finditer(raw_output):
        text = m.group(1).strip()
        if text:
            actions.append(text)
    return actions


def extract_state_json_blocks(raw_output: str) -> List[str]:
    """
    Extract fenced STATE_JSON blocks from model output.
    """
    blocks: List[str] = []
    for m in STATE_JSON_PATTERN.finditer(raw_output):
        text = m.group(1).strip()
        if text:
            blocks.append(text)
    return blocks


def build_json_checkpoints(
    raw_output: str,
    validate_state_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> List[ParsedJsonCheckpoint]:
    """
    Build parsed checkpoints from STATE_JSON blocks.

    Convention:
        first JSON block aligns to action_index = 1,
        second to action_index = 2, etc.
    """
    raw_blocks = extract_state_json_blocks(raw_output)
    checkpoints: List[ParsedJsonCheckpoint] = []

    for idx, raw_json in enumerate(raw_blocks, start=1):
        try:
            full_json = json.loads(raw_json)
            parsed_state = validate_state_fn(full_json)
            checkpoints.append(
                ParsedJsonCheckpoint(
                    action_index=idx,
                    raw_json_text=raw_json,
                    parsed_state=parsed_state,
                    parse_error=None,
                    raw_full_json=full_json,
                )
            )
        except Exception as e:
            checkpoints.append(
                ParsedJsonCheckpoint(
                    action_index=idx,
                    raw_json_text=raw_json,
                    parsed_state=None,
                    parse_error=str(e),
                    raw_full_json=None,
                )
            )

    return checkpoints