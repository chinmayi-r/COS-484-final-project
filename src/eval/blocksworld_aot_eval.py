from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.sim.blocksworld_sim import (
    ParsedAction,
    classify_error_type,
    diff_predicates,
    nl_state_to_canonical,
    parse_action_text,
    rollout_actions,
)


ACTION_PATTERN = re.compile(
    r"ACTION:\s*(.*?)(?=\n(?:STATE\s+[0-9]+(?:\.[0-9]+)?\s*:|ACTION:|Plan summary:)|\Z)",
    re.DOTALL,
)

STATE_PATTERN = re.compile(
    r"STATE\s+([0-9]+(?:\.[0-9]+)?)\s*:\s*(.*?)(?=\n(?:STATE\s+[0-9]+(?:\.[0-9]+)?\s*:|ACTION:|Plan summary:)|\Z)",
    re.DOTALL,
)


@dataclass
class ParsedCheckpoint:
    state_id: str
    state_text: str
    action_index: int
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
    """Extract AoT+ ACTION strings in order."""
    actions: List[str] = []
    for m in ACTION_PATTERN.finditer(raw_output):
        text = m.group(1).strip()
        if text:
            actions.append(text)
    return actions


def extract_state_entries(raw_output: str) -> List[tuple[str, str]]:
    """Extract AoT+ STATE entries in order."""
    entries: List[tuple[str, str]] = []
    for m in STATE_PATTERN.finditer(raw_output):
        state_id = m.group(1).strip()
        state_text = m.group(2).strip()
        if state_text:
            entries.append((state_id, state_text))
    return entries


def build_checkpoints(raw_output: str) -> List[ParsedCheckpoint]:
    """
    Build parsed checkpoints from AoT+ STATE entries.

    Alignment rule:
        - STATE 0 -> action_index 0
        - subsequent states align by textual order
    """
    entries = extract_state_entries(raw_output)
    checkpoints: List[ParsedCheckpoint] = []

    action_index = 0
    for i, (state_id, state_text) in enumerate(entries):
        if i == 0 and state_id == "0":
            current_index = 0
        else:
            action_index += 1
            current_index = action_index

        try:
            parsed = nl_state_to_canonical(state_text)
            checkpoints.append(
                ParsedCheckpoint(
                    state_id=state_id,
                    state_text=state_text,
                    action_index=current_index,
                    parsed_state=parsed,
                    parse_error=None,
                )
            )
        except Exception as e:
            checkpoints.append(
                ParsedCheckpoint(
                    state_id=state_id,
                    state_text=state_text,
                    action_index=current_index,
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


def evaluate_blocksworld_aot_output(
    initial_state_text: str,
    raw_output: str,
) -> EvaluationResult:
    """
    Evaluate AoT+ Blocksworld output against a shared ground-truth simulator.
    """
    initial_state = nl_state_to_canonical(initial_state_text)

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
                    "state_id": cp.state_id,
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
                    "state_id": cp.state_id,
                    "action_index": cp.action_index,
                    "match": False,
                    "parse_error": cp.parse_error,
                    "ground_truth_state": gt_state,
                    "model_state": None,
                    "missing_predicates": None,
                    "extra_predicates": None,
                    "error_type": "state_parse_failure",
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
                "state_id": cp.state_id,
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