from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from src.eval.common_error_analysis import (
    analyze_checkpoint_transition,
    compute_binary_errors,
    finalize_accumulation_labels,
    first_error_step,
)
from src.eval.common_types import EvaluationResult


def require_goal_fields(raw_full_json: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    Check that goal-residual fields are present and well-typed.
    """
    if raw_full_json is None:
        return "raw_full_json_missing"

    required = {"goal_satisfied", "goal_remaining"}
    missing = required - set(raw_full_json.keys())
    if missing:
        return f"Missing goal fields: {sorted(missing)}"

    if not isinstance(raw_full_json["goal_satisfied"], list):
        return "'goal_satisfied' must be a list"

    if not isinstance(raw_full_json["goal_remaining"], list):
        return "'goal_remaining' must be a list"

    return None


def run_evaluation_from_checkpoints(
    *,
    variant_name: str,
    checkpoints: List[Any],
    ground_truth_states: List[Dict[str, Any]],
    invalid_action_index: Optional[int],
    invalid_action_text: Optional[str],
    state_to_predicates_fn: Callable[[Dict[str, Any]], set],
    require_goal_fields_flag: bool = False,
) -> tuple[List[bool], List[Dict[str, Any]]]:
    """
    Shared evaluation loop used by all per-domain evaluators.
    """
    matches: List[bool] = []
    details: List[Dict[str, Any]] = []

    for i, cp in enumerate(checkpoints):
        gt_state = (
            ground_truth_states[cp.action_index]
            if cp.action_index < len(ground_truth_states)
            else None
        )

        prev_gt_state = None
        if cp.action_index > 0 and cp.action_index - 1 < len(ground_truth_states):
            prev_gt_state = ground_truth_states[cp.action_index - 1]

        prev_model_state = None
        if i > 0:
            prev_model_state = checkpoints[i - 1].parsed_state

        goal_field_error = None
        if require_goal_fields_flag:
            goal_field_error = require_goal_fields(getattr(cp, "raw_full_json", None))

        analysis = analyze_checkpoint_transition(
            prev_gt_state=prev_gt_state,
            gt_state=gt_state,
            prev_model_state=prev_model_state,
            model_state=cp.parsed_state,
            state_to_predicates_fn=state_to_predicates_fn,
            parse_error=cp.parse_error,
            goal_field_error=goal_field_error,
        )

        detail = {
            "variant": variant_name,
            "action_index": cp.action_index,
            "ground_truth_state": gt_state,
            "model_state": cp.parsed_state,
            "invalid_action_index": invalid_action_index,
            "invalid_action_text": invalid_action_text,
            **analysis,
        }

        if hasattr(cp, "state_id"):
            detail["state_id"] = cp.state_id

        if hasattr(cp, "raw_full_json"):
            raw_full_json = cp.raw_full_json
            detail["goal_satisfied"] = None if raw_full_json is None else raw_full_json.get("goal_satisfied")
            detail["goal_remaining"] = None if raw_full_json is None else raw_full_json.get("goal_remaining")

        matches.append(analysis["match"])
        details.append(detail)

    finalize_accumulation_labels(details)
    return matches, details


def build_evaluation_result(
    *,
    actions: List[Any],
    ground_truth_states: List[Dict[str, Any]],
    matches: List[bool],
    details: List[Dict[str, Any]],
) -> EvaluationResult:
    """
    Assemble the final evaluation result.
    """
    step_errors, depth_errors = compute_binary_errors(matches)
    return EvaluationResult(
        actions=actions,
        ground_truth_states=ground_truth_states,
        step_errors=step_errors,
        depth_errors=depth_errors,
        first_error_step=first_error_step(step_errors),
        details=details,
    )