from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple


Predicate = Tuple[Any, ...]


@dataclass(frozen=True)
class LogisticsMeta:
    """
    Static metadata for a Logistics problem instance.

    Attributes:
        city_map: Maps each location to its city.
    """
    city_map: Dict[str, str]

    def city_of(self, location: str) -> str:
        if location not in self.city_map:
            raise KeyError(f"Unknown location: {location}")
        return self.city_map[location]

    def is_airport(self, location: str) -> bool:
        return location.endswith("-0")


@dataclass
class ParsedAction:
    """
    Structured Logistics action.

    Supported action types:
        - drive-truck
        - load-truck
        - unload-truck
        - fly-airplane
        - load-airplane
        - unload-airplane
    """
    raw_text: str
    action_type: str
    args: Tuple[str, ...]


# Allow hyphenated locations like L0-0
ID = r"[A-Za-z0-9]+"
LOC = r"[A-Za-z0-9-]+"
CITY = r"[A-Za-z0-9]+"

DRIVE_TRUCK_PATTERN = re.compile(
    rf"drive-truck\s+({ID})\s+({LOC})\s+({LOC})\s+({CITY})",
    re.IGNORECASE,
)
LOAD_TRUCK_PATTERN = re.compile(
    rf"load-truck\s+({ID})\s+({ID})\s+({LOC})",
    re.IGNORECASE,
)
UNLOAD_TRUCK_PATTERN = re.compile(
    rf"unload-truck\s+({ID})\s+({ID})\s+({LOC})",
    re.IGNORECASE,
)
FLY_AIRPLANE_PATTERN = re.compile(
    rf"fly-airplane\s+({ID})\s+({LOC})\s+({LOC})",
    re.IGNORECASE,
)
LOAD_AIRPLANE_PATTERN = re.compile(
    rf"load-airplane\s+({ID})\s+({ID})\s+({LOC})",
    re.IGNORECASE,
)
UNLOAD_AIRPLANE_PATTERN = re.compile(
    rf"unload-airplane\s+({ID})\s+({ID})\s+({LOC})",
    re.IGNORECASE,
)

PACKAGE_AT_PATTERN = re.compile(
    rf"Package\s+({ID})\s+is\s+at\s+({LOC})\.?",
    re.IGNORECASE,
)
PACKAGE_IN_TRUCK_PATTERN = re.compile(
    rf"Package\s+({ID})\s+is\s+in\s+Truck\s+({ID})\.?",
    re.IGNORECASE,
)
PACKAGE_IN_AIRPLANE_PATTERN = re.compile(
    rf"Package\s+({ID})\s+is\s+in\s+Airplane\s+({ID})\.?",
    re.IGNORECASE,
)
TRUCK_AT_PATTERN = re.compile(
    rf"Truck\s+({ID})\s+is\s+at\s+({LOC})\.?",
    re.IGNORECASE,
)
AIRPLANE_AT_PATTERN = re.compile(
    rf"Airplane\s+({ID})\s+is\s+at\s+({LOC})\.?",
    re.IGNORECASE,
)


def canonicalize_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a Logistics state into canonical dictionary form.

    Required keys:
        at, in
    """
    return {
        "at": dict(sorted(state["at"].items())),
        "in": dict(sorted(state["in"].items())),
    }


def clone_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create an independent copy of a canonical Logistics state.
    """
    return {
        "at": dict(state["at"]),
        "in": dict(state["in"]),
    }


def validate_city_map(city_map: Dict[str, Any]) -> LogisticsMeta:
    """
    Validate static city metadata and return a LogisticsMeta object.
    """
    if not isinstance(city_map, dict):
        raise ValueError("'city_map' must be a dictionary")

    if not city_map:
        raise ValueError("'city_map' must be non-empty")

    for loc, city in city_map.items():
        if not isinstance(loc, str) or not isinstance(city, str):
            raise ValueError("city_map must map string location -> string city")

    return LogisticsMeta(city_map=dict(city_map))


def validate_json_state(state_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and canonicalize a base Logistics JSON state.

    Required keys:
        at, in

    Static metadata like city_map is intentionally NOT required here because it
    is loaded separately from the problem definition / prompt header.
    """
    required = {"at", "in"}
    missing = required - set(state_json.keys())
    if missing:
        raise ValueError(f"Missing required state keys: {sorted(missing)}")

    if not isinstance(state_json["at"], dict):
        raise ValueError("'at' must be a dictionary")
    if not isinstance(state_json["in"], dict):
        raise ValueError("'in' must be a dictionary")

    at: Dict[str, Optional[str]] = {}
    inside: Dict[str, Optional[str]] = {}

    for entity, location in state_json["at"].items():
        if not isinstance(entity, str):
            raise ValueError("All 'at' keys must be strings")
        if location is not None and not isinstance(location, str):
            raise ValueError("All 'at' values must be strings or null")
        at[entity] = location

    for package, vehicle in state_json["in"].items():
        if not isinstance(package, str):
            raise ValueError("All 'in' keys must be strings")
        if vehicle is not None and not isinstance(vehicle, str):
            raise ValueError("All 'in' values must be strings or null")
        inside[package] = vehicle

    # Require every package that appears in one map to appear in both maps.
    package_keys = {k for k in at if k.startswith("P")} | set(inside.keys())
    for pkg in package_keys:
        if pkg not in at:
            raise ValueError(f"Package {pkg} missing from 'at'")
        if pkg not in inside:
            raise ValueError(f"Package {pkg} missing from 'in'")

        at_val = at[pkg]
        in_val = inside[pkg]

        # Canonical containment rule from the agreed contract:
        # if a package is in a vehicle => at[pkg] must be null
        # else                        => at[pkg] must be a location
        if in_val is not None and at_val is not None:
            raise ValueError(
                f"Package {pkg} cannot have both at={at_val} and in={in_val}"
            )
        if in_val is None and at_val is None:
            raise ValueError(
                f"Package {pkg} must be either at a location or in a vehicle"
            )

    return canonicalize_state({"at": at, "in": inside})


def nl_state_to_canonical(state_text: str) -> Dict[str, Any]:
    """
    Parse the agreed AoT natural-language state format into canonical form.

    Supported sentence templates:
        - Package P0 is at L0-1.
        - Package P1 is in Truck T0.
        - Package P2 is in Airplane A0.
        - Truck T0 is at L0-1.
        - Airplane A0 is at L1-0.
    """
    at: Dict[str, Optional[str]] = {}
    inside: Dict[str, Optional[str]] = {}

    for pkg, loc in PACKAGE_AT_PATTERN.findall(state_text):
        at[pkg] = loc
        inside.setdefault(pkg, None)

    for pkg, truck in PACKAGE_IN_TRUCK_PATTERN.findall(state_text):
        at[pkg] = None
        inside[pkg] = truck

    for pkg, airplane in PACKAGE_IN_AIRPLANE_PATTERN.findall(state_text):
        at[pkg] = None
        inside[pkg] = airplane

    for truck, loc in TRUCK_AT_PATTERN.findall(state_text):
        at[truck] = loc

    for airplane, loc in AIRPLANE_AT_PATTERN.findall(state_text):
        at[airplane] = loc

    if not at and not inside:
        raise ValueError("Could not parse any Logistics state facts")

    return validate_json_state({"at": at, "in": inside})


def state_to_predicates(state: Dict[str, Any]) -> Set[Predicate]:
    """
    Convert a canonical Logistics state into a set of dynamic predicates.

    Dynamic predicates:
        at(X, L)
        in(P, V)

    We intentionally do NOT include static city metadata in the predicate set,
    because city_map is external metadata, not part of the predicted dynamic state.
    """
    predicates: Set[Predicate] = set()

    for entity, location in state["at"].items():
        if location is not None:
            predicates.add(("at", entity, location))

    for package, vehicle in state["in"].items():
        if vehicle is not None:
            predicates.add(("in", package, vehicle))

    return predicates


def parse_action_text(action_text: str) -> ParsedAction:
    """
    Parse agreed PDDL-style Logistics action strings.

    Supported forms:
        drive-truck T1 L1-0 L1-1 C1
        load-truck P2 T1 L1-1
        unload-truck P2 T1 L1-0
        fly-airplane A0 L1-0 L2-0
        load-airplane P1 A0 L2-0
        unload-airplane P1 A0 L0-0
    """
    text = action_text.strip()

    m = DRIVE_TRUCK_PATTERN.fullmatch(text)
    if m:
        return ParsedAction(text, "drive-truck", m.groups())

    m = LOAD_TRUCK_PATTERN.fullmatch(text)
    if m:
        return ParsedAction(text, "load-truck", m.groups())

    m = UNLOAD_TRUCK_PATTERN.fullmatch(text)
    if m:
        return ParsedAction(text, "unload-truck", m.groups())

    m = FLY_AIRPLANE_PATTERN.fullmatch(text)
    if m:
        return ParsedAction(text, "fly-airplane", m.groups())

    m = LOAD_AIRPLANE_PATTERN.fullmatch(text)
    if m:
        return ParsedAction(text, "load-airplane", m.groups())

    m = UNLOAD_AIRPLANE_PATTERN.fullmatch(text)
    if m:
        return ParsedAction(text, "unload-airplane", m.groups())

    raise ValueError(f"Could not parse Logistics action: {action_text}")


def _vehicle_type(vehicle: str) -> str:
    """
    Infer vehicle type from naming convention (case-insensitive).
    PlanBench uses lowercase names (t0, a0); our examples use uppercase (T0, A0).
    """
    v = vehicle.lower()
    if v.startswith("t"):
        return "truck"
    if v.startswith("a"):
        return "airplane"
    raise ValueError(f"Unknown vehicle type for id: {vehicle}")


def _same_city(loc1: str, loc2: str, meta: LogisticsMeta) -> bool:
    return meta.city_of(loc1) == meta.city_of(loc2)


def apply_action(
    state: Dict[str, Any],
    action: ParsedAction,
    meta: LogisticsMeta,
) -> bool:
    """
    Apply one action to a mutable canonical Logistics state.

    Returns:
        True if the action is valid and applied, else False.
    """
    at = dict(state["at"])
    inside = dict(state["in"])

    if action.action_type == "drive-truck":
        truck, src, dst, city = action.args

        if _vehicle_type(truck) != "truck":
            return False
        if truck not in at or at[truck] != src:
            return False
        if src not in meta.city_map or dst not in meta.city_map:
            return False
        if meta.city_of(src) != city or meta.city_of(dst) != city:
            return False
        if not _same_city(src, dst, meta):
            return False

        at[truck] = dst

    elif action.action_type == "load-truck":
        package, truck, loc = action.args

        if _vehicle_type(truck) != "truck":
            return False
        if package not in inside or package not in at:
            return False
        if truck not in at:
            return False
        if at[truck] != loc:
            return False
        if inside[package] is not None:
            return False
        if at[package] != loc:
            return False

        at[package] = None
        inside[package] = truck

    elif action.action_type == "unload-truck":
        package, truck, loc = action.args

        if _vehicle_type(truck) != "truck":
            return False
        if package not in inside or package not in at:
            return False
        if truck not in at:
            return False
        if inside[package] != truck:
            return False
        if at[truck] != loc:
            return False

        at[package] = loc
        inside[package] = None

    elif action.action_type == "fly-airplane":
        airplane, src, dst = action.args

        if _vehicle_type(airplane) != "airplane":
            return False
        if airplane not in at or at[airplane] != src:
            return False
        if src not in meta.city_map or dst not in meta.city_map:
            return False
        if not meta.is_airport(src) or not meta.is_airport(dst):
            return False
        if _same_city(src, dst, meta):
            return False

        at[airplane] = dst

    elif action.action_type == "load-airplane":
        package, airplane, loc = action.args

        if _vehicle_type(airplane) != "airplane":
            return False
        if package not in inside or package not in at:
            return False
        if airplane not in at:
            return False
        if at[airplane] != loc:
            return False
        if not meta.is_airport(loc):
            return False
        if inside[package] is not None:
            return False
        if at[package] != loc:
            return False

        at[package] = None
        inside[package] = airplane

    elif action.action_type == "unload-airplane":
        package, airplane, loc = action.args

        if _vehicle_type(airplane) != "airplane":
            return False
        if package not in inside or package not in at:
            return False
        if airplane not in at:
            return False
        if inside[package] != airplane:
            return False
        if at[airplane] != loc:
            return False
        if not meta.is_airport(loc):
            return False

        at[package] = loc
        inside[package] = None

    else:
        return False

    state["at"] = at
    state["in"] = inside
    return True


def rollout_actions(
    initial_state: Dict[str, Any],
    actions: List[ParsedAction],
    meta: LogisticsMeta,
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
        ok = apply_action(nxt, action, meta)
        if not ok:
            return states, idx, action.raw_text

        states.append(clone_state(nxt))
        cur = nxt

    return states, None, None