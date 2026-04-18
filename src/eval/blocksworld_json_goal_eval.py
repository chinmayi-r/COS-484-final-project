from __future__ import annotations

from src.eval.common_eval_runner import (
    build_evaluation_result,
    run_evaluation_from_checkpoints,
)
from src.eval.common_json_utils import build_json_checkpoints, extract_actions
from src.sim.blocksworld_sim import (
    parse_action_text,
    rollout_actions,
    state_to_predicates,
    validate_json_state,
)


def evaluate_blocksworld_json_goal_output(
    initial_state_json,
    raw_output: str,
):
    """
    Evaluate JSON + goal residual Blocksworld output.
    """
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
    )

    matches, details = run_evaluation_from_checkpoints(
        variant_name="json_goal_residual",
        checkpoints=checkpoints,
        ground_truth_states=ground_truth_states,
        invalid_action_index=invalid_action_index,
        invalid_action_text=invalid_action_text,
        state_to_predicates_fn=state_to_predicates,
        require_goal_fields_flag=True,
    )

    return build_evaluation_result(
        actions=parsed_actions,
        ground_truth_states=ground_truth_states,
        matches=matches,
        details=details,
    )