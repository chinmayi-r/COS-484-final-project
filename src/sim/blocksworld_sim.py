from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from src.domains.blocksworld import BlocksWorldDomain


Predicate = Tuple[Any, ...]


@dataclass
class ParsedAction:
    """
    Structured Blocksworld action.
    """
    raw_text: str
    action_type: str
    args: Tuple[str, ...]


MOVE_PATTERN = re.compile(
    r"Move\s+Block\s+([A-Za-z0-9]+)\s+from\s+(.+?)\s+to\s+(?:Block\s+([A-Za-z0-9]+)|the\s+table|table)",
    re.IGNORECASE,
)
UNSTACK_PATTERN = re.compile(
    r"Unstack\s+([A-Za-z0-9]+)\s+from\s+([A-Za-z0-9]+)",
    re.IGNORECASE,
)
STACK_PATTERN = re.compile(
    r"Stack\s+([A-Za-z0-9]+)\s+on\s+([A-Za-z0-9]+)",
    re.IGNORECASE,
)
PICKUP_PATTERN = re.compile(
    r"Pick\s*up\s+([A-Za-z0-9]+)",
    re.IGNORECASE,
)
PUTDOWN_PATTERN = re.compile(
    r"Put\s*down\s+([A-Za-z0-9]+)",
    re.IGNORECASE,
)


def canonicalize_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a Blocksworld state into canonical dictionary form.
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
    Validate and canonicalize a base Blocksworld JSON state.
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


def state_to_predicates(state: Dict[str, Any]) -> Set[Predicate]:
    """
    Convert a canonical Blocksworld state into a set of predicates.
    """
    predicates: Set[Predicate] = set()

    for block, dest in state["on"].items():
        predicates.add(("on", block, dest))

    for block in state["clear"]:
        predicates.add(("clear", block))

    if state["holding"] is not None:
        predicates.add(("holding", state["holding"]))

    if state["handempty"]:
        predicates.add(("handempty",))

    return predicates


def parse_action_text(action_text: str) -> ParsedAction:
    """
    Parse Blocksworld actions.
    """
    text = action_text.strip().rstrip(".,;!?")

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
        if block not in on:
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