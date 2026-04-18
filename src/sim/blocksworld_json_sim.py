from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from src.domains.blocksworld import BlocksWorldDomain


@dataclass
class ParsedAction:
    """
    Structured Blocksworld action.

    Attributes:
        raw_text: Original action string.
        action_type: 'move', 'unstack', 'stack', 'pickup', or 'putdown'.
        args: Action arguments in order.
    """
    raw_text: str
    action_type: str
    args: Tuple[str, ...]


@dataclass
class PredicateDiff:
    """
    Predicate-level difference between a ground-truth state and a predicted state.
    """
    missing_predicates: Set[Tuple[Any, ...]]
    extra_predicates: Set[Tuple[Any, ...]]


# AoT+ / repo action format:
# "Move Block B from Block A to the table"
# "Move Block A from the table to Block C"
MOVE_PATTERN = re.compile(
    r"Move\s+Block\s+([A-Za-z0-9]+)\s+from\s+(.+?)\s+to\s+(?:Block\s+([A-Za-z0-9]+)|the\s+table|table)",
    re.IGNORECASE,
)

# JSON-method / PDDL-style surface forms implied by your doc:
UNSTACK_PATTERN = re.compile(r"Unstack\s+([A-Za-z0-9]+)\s+from\s+([A-Za-z0-9]+)", re.IGNORECASE)
STACK_PATTERN = re.compile(r"Stack\s+([A-Za-z0-9]+)\s+on\s+([A-Za-z0-9]+)", re.IGNORECASE)
PICKUP_PATTERN = re.compile(r"Pick\s*up\s+([A-Za-z0-9]+)", re.IGNORECASE)
PUTDOWN_PATTERN = re.compile(r"Put\s*down\s+([A-Za-z0-9]+)", re.IGNORECASE)


def canonicalize_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a Blocksworld state into canonical dictionary form.

    Args:
        state: dict with keys 'on', 'clear', 'holding', 'handempty'.

    Returns:
        Canonicalized state with sorted 'on' mapping and sorted unique 'clear' list.
    """
    return {
        "on": dict(sorted(state["on"].items())),
        "clear": sorted(set(state["clear"])),
        "holding": state["holding"],
        "handempty": bool(state["handempty"]),
    }


def clone_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create an independent copy of a canonical Blocksworld state.
    """
    return {
        "on": dict(state["on"]),
        "clear": list(state["clear"]),
        "holding": state["holding"],
        "handempty": state["handempty"],
    }


def validate_json_state(state_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and canonicalize a Blocksworld JSON state.

    Required keys follow your doc:
        on, clear, holding, handempty

    Optional metadata keys:
        goal_satisfied, goal_remaining

    Args:
        state_json: Parsed JSON object.

    Returns:
        Canonicalized state.

    Raises:
        ValueError: If required fields are missing or malformed.
    """
    required = {"on", "clear", "holding", "handempty"}
    missing = required - set(state_json.keys())
    if missing:
        raise ValueError(f"Missing required state keys: {sorted(missing)}")

    if not isinstance(state_json["on"], dict):
        raise ValueError("'on' must be a dictionary")
    if not isinstance(state_json["clear"], list):
        raise ValueError("'clear' must be a list")
    if state_json["holding"] is not None and not isinstance(state_json["holding"], str):
        raise ValueError("'holding' must be a string or null")
    if not isinstance(state_json["handempty"], bool):
        raise ValueError("'handempty' must be a boolean")

    return canonicalize_state(
        {
            "on": dict(state_json["on"]),
            "clear": list(state_json["clear"]),
            "holding": state_json["holding"],
            "handempty": state_json["handempty"],
        }
    )


def nl_state_to_canonical(state_text: str) -> Dict[str, Any]:
    """
    Parse a natural-language Blocksworld state into canonical form.

    Uses the repo's domain parser for AoT+ state text.

    Args:
        state_text: Natural-language state description.

    Returns:
        Canonical Blocksworld state.
    """
    parsed = BlocksWorldDomain.parse_state_description(state_text)
    return canonicalize_state(
        {
            "on": dict(parsed.on),
            "clear": list(parsed.clear),
            "holding": None,
            "handempty": True,
        }
    )


def state_to_predicates(state: Dict[str, Any]) -> Set[Tuple[Any, ...]]:
    """
    Convert a canonical Blocksworld state into a set of predicates.

    Predicates:
        on(X, Y), clear(X), holding(X), handempty
    """
    predicates: Set[Tuple[Any, ...]] = set()

    for block, dest in state["on"].items():
        predicates.add(("on", block, dest))

    for block in state["clear"]:
        predicates.add(("clear", block))

    if state["holding"] is not None:
        predicates.add(("holding", state["holding"]))

    if state["handempty"]:
        predicates.add(("handempty",))

    return predicates


def diff_predicates(
    ground_truth_state: Dict[str, Any],
    predicted_state: Dict[str, Any],
) -> PredicateDiff:
    """
    Compare two states at the predicate level.

    Returns:
        PredicateDiff with missing and extra predicates.
    """
    gt = state_to_predicates(ground_truth_state)
    pred = state_to_predicates(predicted_state)
    return PredicateDiff(
        missing_predicates=gt - pred,
        extra_predicates=pred - gt,
    )


def classify_error_type(diff: PredicateDiff) -> str:
    """
    Assign a coarse first-pass error label from predicate differences.

    Returns:
        'none', 'omission', 'hallucination', or 'mixed'
    """
    missing = len(diff.missing_predicates)
    extra = len(diff.extra_predicates)

    if missing == 0 and extra == 0:
        return "none"
    if missing > 0 and extra == 0:
        return "omission"
    if missing == 0 and extra > 0:
        return "hallucination"
    return "mixed"


def parse_action_text(action_text: str) -> ParsedAction:
    """
    Parse either AoT+ natural-language Blocksworld actions or JSON-method
    PDDL-style surface actions.

    Supported formats:
        - Move Block X from ... to Block Y / the table
        - Unstack X from Y
        - Stack X on Y
        - Pick up X
        - Put down X

    Args:
        action_text: Raw action string.

    Returns:
        ParsedAction.

    Raises:
        ValueError: If the action cannot be parsed.
    """
    text = action_text.strip()

    m = MOVE_PATTERN.fullmatch(text)
    if m:
        block = m.group(1)
        dest_block = m.group(3)
        destination = dest_block if dest_block is not None else "table"
        return ParsedAction(text, "move", (block, destination))

    m = UNSTACK_PATTERN.fullmatch(text)
    if m:
        return ParsedAction(text, "unstack", (m.group(1), m.group(2)))

    m = STACK_PATTERN.fullmatch(text)
    if m:
        return ParsedAction(text, "stack", (m.group(1), m.group(2)))

    m = PICKUP_PATTERN.fullmatch(text)
    if m:
        return ParsedAction(text, "pickup", (m.group(1),))

    m = PUTDOWN_PATTERN.fullmatch(text)
    if m:
        return ParsedAction(text, "putdown", (m.group(1),))

    raise ValueError(f"Could not parse Blocksworld action: {action_text}")


def apply_action(state: Dict[str, Any], action: ParsedAction) -> bool:
    """
    Apply one action to a mutable canonical Blocksworld state.

    Args:
        state: Mutable canonical state.
        action: Parsed action.

    Returns:
        True if the action is valid and applied, else False.
    """
    clear = set(state["clear"])
    on = dict(state["on"])
    holding = state["holding"]
    handempty = state["handempty"]

    if action.action_type == "move":
        block, destination = action.args
        if holding is not None or not handempty:
            return False
        if block not in clear:
            return False
        if destination == block:
            return False
        if destination != "table" and destination not in on:
            return False
        if destination != "table" and destination not in clear:
            return False

        current_location = on[block]
        if current_location != "table":
            clear.add(current_location)

        on[block] = destination

        if destination != "table":
            clear.remove(destination)

    elif action.action_type == "unstack":
        x, y = action.args
        if not handempty or holding is not None:
            return False
        if on.get(x) != y:
            return False
        if x not in clear:
            return False

        del on[x]
        clear.remove(x)
        clear.add(y)
        holding = x
        handempty = False

    elif action.action_type == "stack":
        x, y = action.args
        if handempty or holding != x:
            return False
        if y not in clear:
            return False

        on[x] = y
        clear.add(x)
        clear.remove(y)
        holding = None
        handempty = True

    elif action.action_type == "pickup":
        (x,) = action.args
        if not handempty or holding is not None:
            return False
        if on.get(x) != "table":
            return False
        if x not in clear:
            return False

        del on[x]
        clear.remove(x)
        holding = x
        handempty = False

    elif action.action_type == "putdown":
        (x,) = action.args
        if handempty or holding != x:
            return False

        on[x] = "table"
        clear.add(x)
        holding = None
        handempty = True

    else:
        return False

    state["on"] = on
    state["clear"] = sorted(clear)
    state["holding"] = holding
    state["handempty"] = handempty
    return True


def rollout_actions(
    initial_state: Dict[str, Any],
    actions: List[ParsedAction],
) -> Tuple[List[Dict[str, Any]], Optional[int], Optional[str]]:
    """
    Roll out a sequence of actions from an initial state.

    Convention:
        states[t] = state after t actions

    Args:
        initial_state: Canonical initial state.
        actions: Parsed action sequence.

    Returns:
        states: List of canonical states.
        invalid_action_index: 1-based first invalid action index, if any.
        invalid_action_text: Raw text of first invalid action, if any.
    """
    cur = clone_state(initial_state)
    states = [clone_state(cur)]

    for idx, action in enumerate(actions, start=1):
        nxt = clone_state(cur)
        ok = apply_action(nxt, action)
        if not ok:
            return states, idx, action.raw_text

        states.append(clone_state(nxt))
        cur = nxt

    return states, None, None