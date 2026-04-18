from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.sim.blocksworld_sim import (
    ParsedAction,
    classify_error_type,
    diff_predicates,
    parse_action_text,
    rollout_actions,
    validate_json_state,
)


ACTION_PATTERN = re.compile(
    r"ACTION:\s*(.*?)(?=\n(?:REASONING:|STATE_JSON:|ACTION:)|\Z)",
    re.DOTALL,
)

STATE_JSON_PATTERN = re.compile(
    r"STATE_JSON:\s*```json\s*(.*?)\s*```",
    re.DOTALL | re.IGNORECASE,
)


@dataclass
class ParsedCheckpoint:
    action_index: int
    raw_json_text: str
    parsed_state: Optional[Dict[str, Any]]
    parse_error: Optional[str]


@dataclass
class EvaluationResult:
    actions: List[ParsedAction]
    checkpoints: List[ParsedCheckpoint]
    ground_truth_states: List[Dict[str, Any]]
    step_errors: List[int]
    depth_errors: List[int]
    first_error_step: Optional[int]
    details: List[Dict[str, Any]]


def extract_actions(raw_output: str) -> List[str]:
    """Extract JSON-method ACTION strings in order."""
    actions: List[str] = []
    for m in ACTION_PATTERN.finditer(raw_output):
        text = m.group(1).strip()
        if text:
            actions.append(text)
    return actions


def extract_state_json_blocks(raw_output: str) -> List[str]:
    """Extract fenced STATE_JSON blocks in order."""
    blocks: List[str] = []
    for m in STATE_JSON_PATTERN.finditer(raw_output):
        text = m.group(1).strip()
        if text:
            blocks.append(text)
    return blocks


def build_checkpoints(raw_output: str) -> List[ParsedCheckpoint]:
    """
    Build parsed checkpoints from STATE_JSON blocks.

    Following your doc's example, the first JSON block is aligned to
    action_index = 1, the second to action_index = 2, and so on.
    """
    raw_blocks = extract_state_json_blocks(raw_output)
    checkpoints: List[ParsedCheckpoint] = []

    for idx, raw_json in enumerate(raw_blocks, start=1):
        try:
            parsed = json.loads(raw_json)
            parsed = validate_json_state(parsed)
            checkpoints.append(
                ParsedCheckpoint(
                    action_index=idx,
                    raw_json_text=raw_json,
                    parsed_state=parsed,
                    parse_error=None,
                )
            )
        except Exception as e:
            checkpoints.append(
                ParsedCheckpoint(
                    action_index=idx,
                    raw_json_text=raw_json,
                    parsed_state=None,
                    parse_error=str(e),
                )
            )

    return checkpoints


def compute_binary_errors(matches: List[bool]) -> tuple[List[int], List[int]]:
    """Compute step-level and depth-level binary errors."""
    step_errors = [0 if m else 1 for m in matches]
    depth_errors: List[int] = []
    running = 0
    for e in step_errors:
        running = max(running, e)
        depth_errors.append(running)
    return step_errors, depth_errors


def first_error_step(step_errors: List[int]) -> Optional[int]:
    """Return the first 1-based checkpoint index with error, if any."""
    for i, e in enumerate(step_errors, start=1):
        if e == 1:
            return i
    return None


def evaluate_blocksworld_json_output(
    initial_state_json: Dict[str, Any],
    raw_output: str,
) -> EvaluationResult:
    """
    Evaluate Blocksworld JSON output against the shared ground-truth simulator.

    Handles both:
        - structured JSON
        - JSON + goal residual

    because goal-residual fields are treated as optional metadata rather than
    part of the core world state.
    """
    initial_state = validate_json_state(initial_state_json)

    raw_actions = extract_actions(raw_output)
    parsed_actions = [parse_action_text(a) for a in raw_actions]

    checkpoints = build_checkpoints(raw_output)
    ground_truth_states, invalid_action_index, invalid_action_text = rollout_actions(
        initial_state,
        parsed_actions,
    )

    matches: List[bool] = []
    details: List[Dict[str, Any]] = []

    for cp in checkpoints:
        if cp.action_index >= len(ground_truth_states):
            matches.append(False)
            details.append(
                {
                    "action_index": cp.action_index,
                    "match": False,
                    "parse_error": cp.parse_error,
                    "ground_truth_state": None,
                    "model_state": cp.parsed_state,
                    "missing_predicates": None,
                    "extra_predicates": None,
                    "error_type": "invalid_rollout_suffix",
                    "invalid_action_index": invalid_action_index,
                    "invalid_action_text": invalid_action_text,
                }
            )
            continue

        gt_state = ground_truth_states[cp.action_index]

        if cp.parsed_state is None:
            matches.append(False)
            details.append(
                {
                    "action_index": cp.action_index,
                    "match": False,
                    "parse_error": cp.parse_error,
                    "ground_truth_state": gt_state,
                    "model_state": None,
                    "missing_predicates": None,
                    "extra_predicates": None,
                    "error_type": "json_parse_failure",
                    "invalid_action_index": invalid_action_index,
                    "invalid_action_text": invalid_action_text,
                }
            )
            continue

        diff = diff_predicates(gt_state, cp.parsed_state)
        match = len(diff.missing_predicates) == 0 and len(diff.extra_predicates) == 0
        matches.append(match)

        details.append(
            {
                "action_index": cp.action_index,
                "match": match,
                "parse_error": None,
                "ground_truth_state": gt_state,
                "model_state": cp.parsed_state,
                "missing_predicates": sorted(diff.missing_predicates),
                "extra_predicates": sorted(diff.extra_predicates),
                "error_type": classify_error_type(diff),
                "invalid_action_index": invalid_action_index,
                "invalid_action_text": invalid_action_text,
            }
        )

    step_errors, depth_errors = compute_binary_errors(matches)

    return EvaluationResult(
        actions=parsed_actions,
        checkpoints=checkpoints,
        ground_truth_states=ground_truth_states,
        step_errors=step_errors,
        depth_errors=depth_errors,
        first_error_step=first_error_step(step_errors),
        details=details,
    )