from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Set, Tuple

Predicate = Tuple[Any, ...]


def compute_binary_errors(matches: List[bool]) -> tuple[List[int], List[int]]:
    """
    Compute binary step-level and cumulative depth-level errors.

    step_errors[t] = 1 iff checkpoint t is incorrect
    depth_errors[t] = 1 iff any checkpoint up to t is incorrect
    """
    step_errors = [0 if m else 1 for m in matches]
    depth_errors: List[int] = []

    running = 0
    for e in step_errors:
        running = max(running, e)
        depth_errors.append(running)

    return step_errors, depth_errors


def first_error_step(step_errors: List[int]) -> Optional[int]:
    """
    Return the first 1-based checkpoint index with error, if any.
    """
    for i, e in enumerate(step_errors, start=1):
        if e == 1:
            return i
    return None


def predicate_delta(
    prev_state: Dict[str, Any],
    next_state: Dict[str, Any],
    state_to_predicates_fn: Callable[[Dict[str, Any]], Set[Predicate]],
) -> Set[Predicate]:
    """
    Return predicates whose truth value changed between two states.
    """
    prev_preds = state_to_predicates_fn(prev_state)
    next_preds = state_to_predicates_fn(next_state)
    return prev_preds ^ next_preds


def wrong_predicates(
    gt_state: Dict[str, Any],
    model_state: Dict[str, Any],
    state_to_predicates_fn: Callable[[Dict[str, Any]], Set[Predicate]],
) -> Set[Predicate]:
    """
    Return predicates involved in the mismatch between ground truth and model state.
    """
    gt_preds = state_to_predicates_fn(gt_state)
    model_preds = state_to_predicates_fn(model_state)
    return gt_preds ^ model_preds


def analyze_checkpoint_transition(
    *,
    prev_gt_state: Optional[Dict[str, Any]],
    gt_state: Optional[Dict[str, Any]],
    prev_model_state: Optional[Dict[str, Any]],
    model_state: Optional[Dict[str, Any]],
    state_to_predicates_fn: Callable[[Dict[str, Any]], Set[Predicate]],
    parse_error: Optional[str] = None,
    goal_field_error: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze one checkpoint using local transition-aware signals.

    Labels produced here:
        - none
        - state_parse_failure
        - goal_field_failure
        - omission
        - hallucination
        - propagation
        - mixed
        - invalid_rollout_suffix

    Strict accumulation is added later in finalize_accumulation_labels().
    """
    if gt_state is None:
        return {
            "match": False,
            "error_types": {"invalid_rollout_suffix"},
            "missing_predicates": [],
            "extra_predicates": [],
            "required_changed_predicates": [],
            "model_changed_predicates": [],
            "omitted_updates": [],
            "hallucinated_updates": [],
            "persistent_wrong_predicates": [],
            "newly_wrong_predicates": [],
            "all_wrong_predicates": [],
            "accumulation_predicates": [],
            "parse_error": parse_error,
            "goal_field_error": goal_field_error,
        }

    if parse_error is not None or model_state is None:
        labels = {"state_parse_failure"}
        if goal_field_error is not None:
            labels.add("goal_field_failure")

        return {
            "match": False,
            "error_types": labels,
            "missing_predicates": [],
            "extra_predicates": [],
            "required_changed_predicates": [],
            "model_changed_predicates": [],
            "omitted_updates": [],
            "hallucinated_updates": [],
            "persistent_wrong_predicates": [],
            "newly_wrong_predicates": [],
            "all_wrong_predicates": [],
            "accumulation_predicates": [],
            "parse_error": parse_error,
            "goal_field_error": goal_field_error,
        }

    gt_preds = state_to_predicates_fn(gt_state)
    model_preds = state_to_predicates_fn(model_state)

    missing_predicates = gt_preds - model_preds
    extra_predicates = model_preds - gt_preds
    all_wrong_now = gt_preds ^ model_preds

    labels: Set[str] = set()
    if not missing_predicates and not extra_predicates and goal_field_error is None:
        labels.add("none")

    if prev_gt_state is not None:
        required_changed = predicate_delta(
            prev_gt_state, gt_state, state_to_predicates_fn
        )
    else:
        required_changed = set()

    if prev_model_state is not None:
        model_changed = predicate_delta(
            prev_model_state, model_state, state_to_predicates_fn
        )
    else:
        model_changed = set()

    omitted_updates = required_changed & all_wrong_now
    hallucinated_updates = model_changed - required_changed

    if omitted_updates:
        labels.add("omission")
    if hallucinated_updates:
        labels.add("hallucination")

    if prev_gt_state is not None and prev_model_state is not None:
        prev_wrong = wrong_predicates(
            prev_gt_state, prev_model_state, state_to_predicates_fn
        )
        persistent_wrong = prev_wrong & all_wrong_now
        newly_wrong = all_wrong_now - prev_wrong

        if not prev_wrong and all_wrong_now:
            labels.add("propagation")
    else:
        persistent_wrong = set()
        newly_wrong = set(all_wrong_now)

    if goal_field_error is not None:
        labels.add("goal_field_failure")

    if not labels:
        labels.add("mixed")

    return {
        "match": len(all_wrong_now) == 0 and goal_field_error is None,
        "error_types": labels,
        "missing_predicates": sorted(missing_predicates),
        "extra_predicates": sorted(extra_predicates),
        "required_changed_predicates": sorted(required_changed),
        "model_changed_predicates": sorted(model_changed),
        "omitted_updates": sorted(omitted_updates),
        "hallucinated_updates": sorted(hallucinated_updates),
        "persistent_wrong_predicates": sorted(persistent_wrong),
        "newly_wrong_predicates": sorted(newly_wrong),
        "all_wrong_predicates": sorted(all_wrong_now),
        "accumulation_predicates": [],
        "parse_error": parse_error,
        "goal_field_error": goal_field_error,
    }


def finalize_accumulation_labels(details: List[Dict[str, Any]]) -> None:
    """
    Add strict accumulation labels in place.

    Strict definition:
        A predicate mismatch p is an accumulation error from step k onward iff:
        1. p is wrong at step k
        2. p was not wrong at step k-1 (or k is the first checkpoint)
        3. p remains wrong at every subsequent checkpoint until the end
    """
    wrong_sets: List[Set[Predicate]] = [
        set(d.get("all_wrong_predicates", [])) for d in details
    ]

    n = len(wrong_sets)
    if n == 0:
        return

    suffix_intersection: List[Set[Predicate]] = [set() for _ in range(n)]
    suffix_intersection[-1] = set(wrong_sets[-1])

    for i in range(n - 2, -1, -1):
        suffix_intersection[i] = wrong_sets[i] & suffix_intersection[i + 1]

    prev_wrong: Set[Predicate] = set()

    for i in range(n):
        first_appearing_here = wrong_sets[i] - prev_wrong
        strict_accumulation_preds = first_appearing_here & suffix_intersection[i]

        if strict_accumulation_preds:
            details[i]["error_types"] = set(details[i]["error_types"])
            details[i]["error_types"].add("accumulation")

        existing_here = set(details[i].get("accumulation_predicates", []))
        details[i]["accumulation_predicates"] = sorted(
            existing_here | strict_accumulation_preds
        )

        carried = set(strict_accumulation_preds)
        j = i + 1
        while carried and j < n:
            carried = carried & wrong_sets[j]
            if carried:
                details[j]["error_types"] = set(details[j]["error_types"])
                details[j]["error_types"].add("accumulation")

                existing = set(details[j].get("accumulation_predicates", []))
                details[j]["accumulation_predicates"] = sorted(existing | carried)
            j += 1

        prev_wrong = set(wrong_sets[i])