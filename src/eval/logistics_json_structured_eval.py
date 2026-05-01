from __future__ import annotations

from src.eval.common_eval_runner import (
    build_evaluation_result,
    run_evaluation_from_checkpoints,
)
from src.eval.common_json_utils import build_json_checkpoints, extract_actions
from src.sim.logistics_sim import (
    validate_city_map,
    parse_action_text,
    rollout_actions,
    state_to_predicates,
    validate_json_state,
)


def evaluate_logistics_json_structured_output(
    initial_state_json: dict,
    city_map: dict,
    raw_output: str,
):
    """
    Evaluate structured-JSON Logistics output.

    Args:
        initial_state_json: Initial dynamic state containing only 'at' and 'in'.
        city_map: Static location -> city metadata loaded once from the problem definition.
        raw_output: Model completion containing ACTION and STATE_JSON checkpoints.
    """
    meta = validate_city_map(city_map)
    initial_state = validate_json_state(initial_state_json)

    raw_actions = extract_actions(raw_output)
    parsed_actions = [parse_action_text(a) for a in raw_actions]

    checkpoints = build_json_checkpoints(
        raw_output=raw_output,
        validate_state_fn=validate_json_state,
    )

    ground_truth_states, invalid_action_index, invalid_action_text = rollout_actions(
        initial_state,
        parsed_actions,
        meta,
    )

    matches, details = run_evaluation_from_checkpoints(
        variant_name="json_structured",
        checkpoints=checkpoints,
        ground_truth_states=ground_truth_states,
        invalid_action_index=invalid_action_index,
        invalid_action_text=invalid_action_text,
        state_to_predicates_fn=state_to_predicates,
        require_goal_fields_flag=False,
    )

    return build_evaluation_result(
        actions=parsed_actions,
        ground_truth_states=ground_truth_states,
        matches=matches,
        details=details,
    )