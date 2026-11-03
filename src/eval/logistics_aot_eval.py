from __future__ import annotations

from src.eval.common_aot_utils import build_text_checkpoints, extract_actions
from src.eval.common_eval_runner import (
    build_evaluation_result,
    run_evaluation_from_checkpoints,
)
from src.sim.logistics_sim import (
    validate_city_map,
    nl_state_to_canonical,
    parse_action_text,
    rollout_actions,
    state_to_predicates,
)


def evaluate_logistics_aot_output(
    initial_state_text: str,
    city_map: dict,
    raw_output: str,
):
    """
    Evaluate AoT Logistics output.

    Args:
        initial_state_text: Initial state in the agreed AoT sentence format.
        city_map: Static location -> city metadata loaded once from the problem definition.
        raw_output: Model completion containing ACTION and STATE checkpoints.
    """
    meta = validate_city_map(city_map)
    initial_state = nl_state_to_canonical(initial_state_text)

    raw_actions = extract_actions(raw_output)
    parsed_actions = [parse_action_text(a) for a in raw_actions]

    checkpoints = build_text_checkpoints(
        raw_output=raw_output,
        parse_state_fn=nl_state_to_canonical,
    )

    ground_truth_states, invalid_action_index, invalid_action_text = rollout_actions(
        initial_state,
        parsed_actions,
        meta,
    )

    matches, details = run_evaluation_from_checkpoints(
        variant_name="aot_nl",
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