from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

Predicate = Tuple[Any, ...]


@dataclass
class ParsedTextCheckpoint:
    """
    Parsed checkpoint from text-based AoT+ output.

    Attributes:
        state_id: Textual state identifier, e.g. '0', '1', '2'.
        state_text: Raw state text.
        action_index: Checkpoint aligned to state after this many actions.
        parsed_state: Canonical parsed state if parsing succeeded.
        parse_error: Error string if parsing failed.
    """
    state_id: str
    state_text: str
    action_index: int
    parsed_state: Optional[Dict[str, Any]]
    parse_error: Optional[str]


@dataclass
class ParsedJsonCheckpoint:
    """
    Parsed checkpoint from JSON-based output.

    Attributes:
        action_index: Checkpoint aligned to state after this many actions.
        raw_json_text: Raw JSON block extracted from output.
        parsed_state: Canonical parsed state if parsing succeeded.
        parse_error: Error string if parsing or validation failed.
        raw_full_json: Original JSON object before any domain-specific filtering.
    """
    action_index: int
    raw_json_text: str
    parsed_state: Optional[Dict[str, Any]]
    parse_error: Optional[str]
    raw_full_json: Optional[Dict[str, Any]]


@dataclass
class EvaluationResult:
    """
    Final evaluation result for one model output trace.
    """
    actions: List[Any]
    ground_truth_states: List[Dict[str, Any]]
    step_errors: List[int]
    depth_errors: List[int]
    first_error_step: Optional[int]
    details: List[Dict[str, Any]]