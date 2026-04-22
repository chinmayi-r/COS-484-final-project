"""
Logistics domain simulator — mirrors blocksworld_sim.py in structure.

State canonical form:
  {
    "at":  {entity: location_or_null},   # null when package is inside a vehicle
    "in":  {package: vehicle_or_null},
    "city_map": {location: city}         # static per problem
  }

Airports are identified by name prefix "Airport".
Trucks drive within a city; airplanes fly between airports.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from src.domains.logistics import LogisticsDomain


Predicate = Tuple[Any, ...]


@dataclass
class ParsedAction:
    raw_text: str
    action_type: str   # drive_truck | load_truck | unload_truck | fly_airplane | load_airplane | unload_airplane
    args: Tuple[str, ...]


# ---------------------------------------------------------------------------
# Action regex patterns (case-insensitive, flexible spacing)
# ---------------------------------------------------------------------------

DRIVE_PATTERN = re.compile(
    r"Drive\s+Truck\s+(\w+)\s+from\s+(\w+)\s+to\s+(\w+)", re.IGNORECASE
)
LOAD_TRUCK_PATTERN = re.compile(
    r"Load\s+Package\s+(\w+)\s+onto\s+Truck\s+(\w+)", re.IGNORECASE
)
UNLOAD_TRUCK_PATTERN = re.compile(
    r"Unload\s+Package\s+(\w+)\s+from\s+Truck\s+(\w+)\s+at\s+(\w+)", re.IGNORECASE
)
FLY_PATTERN = re.compile(
    r"Fly\s+Airplane\s+(\w+)\s+from\s+(\w+)\s+to\s+(\w+)", re.IGNORECASE
)
LOAD_AIRPLANE_PATTERN = re.compile(
    r"Load\s+Package\s+(\w+)\s+onto\s+Airplane\s+(\w+)", re.IGNORECASE
)
UNLOAD_AIRPLANE_PATTERN = re.compile(
    r"Unload\s+Package\s+(\w+)\s+from\s+Airplane\s+(\w+)\s+at\s+(\w+)", re.IGNORECASE
)


def _is_airport(location: str) -> bool:
    return location.lower().startswith("airport")


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def canonicalize_state(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "at":       {k: v for k, v in sorted(state["at"].items())},
        "in":       {k: v for k, v in sorted(state["in"].items())},
        "city_map": {k: v for k, v in sorted(state["city_map"].items())},
    }


def clone_state(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "at":       dict(state["at"]),
        "in":       dict(state["in"]),
        "city_map": dict(state["city_map"]),
    }


def validate_json_state(state_json: Dict[str, Any]) -> Dict[str, Any]:
    required = {"at", "in", "city_map"}
    missing = required - set(state_json.keys())
    if missing:
        raise ValueError(f"Missing required state keys: {sorted(missing)}")
    if not isinstance(state_json["at"], dict):
        raise ValueError("'at' must be a dict")
    if not isinstance(state_json["in"], dict):
        raise ValueError("'in' must be a dict")
    if not isinstance(state_json["city_map"], dict):
        raise ValueError("'city_map' must be a dict")
    return canonicalize_state(state_json)


def nl_state_to_canonical(state_text: str, city_map: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Parse a natural-language Logistics state into canonical dict form.
    Uses LogisticsDomain.parse_state_description for entity extraction,
    then converts to the {at, in, city_map} schema.
    """
    parsed = LogisticsDomain.parse_state_description(state_text)

    default_city_map = {
        "Airport1": "City1", "Location1": "City1", "Location2": "City1",
        "Airport2": "City2", "Location3": "City2", "Location4": "City2",
    }
    cmap = city_map or default_city_map

    at: Dict[str, Optional[str]] = {}
    in_: Dict[str, Optional[str]] = {}

    for pkg, loc in parsed.package_locations.items():
        if loc in parsed.vehicle_locations:
            at[pkg] = None
            in_[pkg] = loc
        else:
            at[pkg] = loc
            in_[pkg] = None

    for vehicle, loc in parsed.vehicle_locations.items():
        at[vehicle] = loc

    return canonicalize_state({"at": at, "in": in_, "city_map": cmap})


def state_to_predicates(state: Dict[str, Any]) -> Set[Predicate]:
    predicates: Set[Predicate] = set()
    for entity, loc in state["at"].items():
        if loc is not None:
            predicates.add(("at", entity, loc))
    for pkg, vehicle in state["in"].items():
        if vehicle is not None:
            predicates.add(("in", pkg, vehicle))
        else:
            predicates.add(("not-in", pkg))
    return predicates


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------

def parse_action_text(action_text: str) -> ParsedAction:
    text = action_text.strip()

    m = DRIVE_PATTERN.fullmatch(text)
    if m:
        return ParsedAction(text, "drive_truck", (m.group(1), m.group(2), m.group(3)))

    m = LOAD_TRUCK_PATTERN.fullmatch(text)
    if m:
        return ParsedAction(text, "load_truck", (m.group(1), m.group(2)))

    m = UNLOAD_TRUCK_PATTERN.fullmatch(text)
    if m:
        return ParsedAction(text, "unload_truck", (m.group(1), m.group(2), m.group(3)))

    m = FLY_PATTERN.fullmatch(text)
    if m:
        return ParsedAction(text, "fly_airplane", (m.group(1), m.group(2), m.group(3)))

    m = LOAD_AIRPLANE_PATTERN.fullmatch(text)
    if m:
        return ParsedAction(text, "load_airplane", (m.group(1), m.group(2)))

    m = UNLOAD_AIRPLANE_PATTERN.fullmatch(text)
    if m:
        return ParsedAction(text, "unload_airplane", (m.group(1), m.group(2), m.group(3)))

    raise ValueError(f"Could not parse Logistics action: {action_text!r}")


# ---------------------------------------------------------------------------
# Action application
# ---------------------------------------------------------------------------

def apply_action(state: Dict[str, Any], action: ParsedAction) -> bool:
    at = state["at"]
    in_ = state["in"]
    city_map = state["city_map"]

    if action.action_type == "drive_truck":
        truck, frm, to = action.args
        if at.get(truck) != frm:
            return False
        if city_map.get(frm) != city_map.get(to):
            return False
        at[truck] = to
        # move any packages that are in the truck
        for pkg, vehicle in in_.items():
            if vehicle == truck:
                at[pkg] = None  # still in truck, just moved

    elif action.action_type == "load_truck":
        pkg, truck = action.args
        if in_.get(pkg) is not None:
            return False  # already in a vehicle
        if at.get(pkg) != at.get(truck):
            return False  # not at the same location
        if at.get(truck) is None:
            return False  # truck has no location?
        at[pkg] = None
        in_[pkg] = truck

    elif action.action_type == "unload_truck":
        pkg, truck, loc = action.args
        if in_.get(pkg) != truck:
            return False
        if at.get(truck) != loc:
            return False
        in_[pkg] = None
        at[pkg] = loc

    elif action.action_type == "fly_airplane":
        airplane, frm, to = action.args
        if at.get(airplane) != frm:
            return False
        if not _is_airport(frm) or not _is_airport(to):
            return False
        at[airplane] = to
        for pkg, vehicle in in_.items():
            if vehicle == airplane:
                at[pkg] = None  # still in airplane

    elif action.action_type == "load_airplane":
        pkg, airplane = action.args
        if in_.get(pkg) is not None:
            return False
        airplane_loc = at.get(airplane)
        if airplane_loc is None or not _is_airport(airplane_loc):
            return False
        if at.get(pkg) != airplane_loc:
            return False
        at[pkg] = None
        in_[pkg] = airplane

    elif action.action_type == "unload_airplane":
        pkg, airplane, loc = action.args
        if in_.get(pkg) != airplane:
            return False
        if at.get(airplane) != loc:
            return False
        in_[pkg] = None
        at[pkg] = loc

    else:
        return False

    return True


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def rollout_actions(
    initial_state: Dict[str, Any],
    actions: List[ParsedAction],
) -> Tuple[List[Dict[str, Any]], Optional[int], Optional[str]]:
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
