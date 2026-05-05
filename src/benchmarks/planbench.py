"""
Download and parse PlanBench PDDL instances from the karthikv792/LLMs-Planning
GitHub repository, then convert them to the natural-language format expected by
run_aot_mini.py.

Public API
----------
get_blocksworld_problems(n, seed, cache_dir) -> List[Dict]
    Each dict has keys: instance_id, initial_state, goal_state

get_logistics_problems(n, seed, cache_dir) -> (List[Dict], city_map)
    Each dict has keys: instance_id, initial_state, goal_state
    city_map maps location -> city (all lowercase)

Instances are sourced from:
  plan-bench/instances/blocksworld/generated/instance-N.pddl
  plan-bench/instances/logistics/generated/instance-N.pddl
"""
from __future__ import annotations

import os
import random
import re
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

_GITHUB_RAW = "https://raw.githubusercontent.com/karthikv792/LLMs-Planning/main"
_BW_PATH = "plan-bench/instances/blocksworld/generated/instance-{n}.pddl"
_LOG_PATH = "plan-bench/instances/logistics/generated/instance-{n}.pddl"

# Indices 1..189 are available for both domains.
_MAX_INSTANCE = 189


# ---------------------------------------------------------------------------
# PDDL fetch + cache
# ---------------------------------------------------------------------------

def _fetch_pddl(url: str, cache_path: Path) -> str:
    if cache_path.exists():
        return cache_path.read_text()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=15) as r:
        text = r.read().decode()
    cache_path.write_text(text)
    return text


# ---------------------------------------------------------------------------
# Shared s-expression parser
# ---------------------------------------------------------------------------

def _strip_comments(text: str) -> str:
    return re.sub(r";[^\n]*", "", text)


def _tokenize(text: str) -> List[str]:
    text = _strip_comments(text)
    text = text.replace("(", " ( ").replace(")", " ) ")
    return text.split()


def _parse_sexp(tokens: List[str], pos: int) -> Tuple[object, int]:
    if tokens[pos] != "(":
        raise ValueError(f"Expected '(' at token {pos}: {tokens[pos]!r}")
    pos += 1
    result = []
    while pos < len(tokens) and tokens[pos] != ")":
        if tokens[pos] == "(":
            sub, pos = _parse_sexp(tokens, pos)
            result.append(sub)
        else:
            result.append(tokens[pos])
            pos += 1
    return result, pos + 1


# ---------------------------------------------------------------------------
# Blocksworld parser
# ---------------------------------------------------------------------------

def _parse_bw_pddl(text: str) -> Dict[str, Any]:
    """Parse a blocksworld-4ops PDDL instance."""
    tokens = _tokenize(text)
    sexp, _ = _parse_sexp(tokens, 0)
    # sexp[0]='define', sexp[1]=['problem', name], sexp[2:]= sections
    sections = {s[0].lower(): s[1:] for s in sexp[2:] if isinstance(s, list) and s}

    blocks: List[str] = [
        t for t in sections[":objects"]
        if t.lower() not in ("-", "object", "block")
    ]

    on_table: Set[str] = set()
    on: Dict[str, str] = {}  # x -> y (x is on y)
    clear: Set[str] = set()
    hand_empty = False

    for pred in sections[":init"]:
        if not isinstance(pred, list) or not pred:
            continue
        pname = pred[0].lower()
        if pname == "ontable":
            on_table.add(pred[1].lower())
        elif pname == "on":
            on[pred[1].lower()] = pred[2].lower()
        elif pname == "clear":
            clear.add(pred[1].lower())
        elif pname in ("handempty", "arm-empty"):
            hand_empty = True

    # :goal — (and ...)
    goal_on: Dict[str, str] = {}
    goal_on_table: Set[str] = set()
    goal_body = sections[":goal"][0]
    goal_preds = goal_body[1:] if goal_body[0].lower() == "and" else [goal_body]
    for pred in goal_preds:
        if not isinstance(pred, list) or not pred:
            continue
        pname = pred[0].lower()
        if pname == "on":
            goal_on[pred[1].lower()] = pred[2].lower()
        elif pname == "ontable":
            goal_on_table.add(pred[1].lower())

    return {
        "blocks": [b.lower() for b in blocks],
        "on_table": on_table,
        "on": on,
        "clear": clear,
        "hand_empty": hand_empty,
        "goal_on": goal_on,
        "goal_on_table": goal_on_table,
    }


def _bw_to_nl(data: Dict[str, Any]) -> Tuple[str, str]:
    """Convert parsed BW PDDL data to (initial_state_nl, goal_state_nl)."""
    # Map single lowercase letter -> uppercase "Block X" prefix
    def blk(b: str) -> str:
        return f"Block {b.upper()}"

    parts: List[str] = []
    for b in sorted(data["blocks"]):
        if b in data["on"]:
            parts.append(f"{blk(b)} is on {blk(data['on'][b])}")
        elif b in data["on_table"]:
            parts.append(f"{blk(b)} is on the table")
    for b in sorted(data["clear"]):
        parts.append(f"{blk(b)} is clear")
    if data["hand_empty"]:
        parts.append("Hand is empty")
    initial_nl = ". ".join(parts) + "."

    goal_parts: List[str] = []
    for b in sorted(set(list(data["goal_on"].keys()) + list(data["goal_on_table"]))):
        if b in data["goal_on"]:
            goal_parts.append(f"{blk(b)} is on {blk(data['goal_on'][b])}")
        elif b in data["goal_on_table"]:
            goal_parts.append(f"{blk(b)} is on the table")
    goal_nl = ". ".join(goal_parts) + "."

    return initial_nl, goal_nl


# ---------------------------------------------------------------------------
# Logistics parser
# ---------------------------------------------------------------------------

def _parse_log_pddl(text: str) -> Dict[str, Any]:
    """Parse a logistics-strips PDDL instance."""
    tokens = _tokenize(text)
    sexp, _ = _parse_sexp(tokens, 0)
    sections = {s[0].lower(): s[1:] for s in sexp[2:] if isinstance(s, list) and s}

    # Typed object sets discovered from predicates
    cities: Set[str] = set()
    trucks: Set[str] = set()
    airplanes: Set[str] = set()
    locations: Set[str] = set()
    packages: Set[str] = set()
    airports: Set[str] = set()

    at: Dict[str, str] = {}     # entity -> location
    in_vehicle: Dict[str, str] = {}  # package -> vehicle
    city_of: Dict[str, str] = {}  # location -> city

    for pred in sections[":init"]:
        if not isinstance(pred, list) or not pred:
            continue
        pname = pred[0].lower()
        if pname == "city":
            cities.add(pred[1].lower())
        elif pname == "truck":
            trucks.add(pred[1].lower())
        elif pname == "airplane":
            airplanes.add(pred[1].lower())
        elif pname == "location":
            locations.add(pred[1].lower())
        elif pname == "obj":
            packages.add(pred[1].lower())
        elif pname == "airport":
            airports.add(pred[1].lower())
        elif pname == "at":
            at[pred[1].lower()] = pred[2].lower()
        elif pname == "in":
            in_vehicle[pred[1].lower()] = pred[2].lower()
        elif pname == "in-city":
            city_of[pred[1].lower()] = pred[2].lower()

    # :goal
    goal_at: Dict[str, str] = {}
    goal_in: Dict[str, str] = {}
    goal_body = sections[":goal"][0]
    goal_preds = goal_body[1:] if goal_body[0].lower() == "and" else [goal_body]
    for pred in goal_preds:
        if not isinstance(pred, list) or not pred:
            continue
        pname = pred[0].lower()
        if pname == "at":
            goal_at[pred[1].lower()] = pred[2].lower()
        elif pname == "in":
            goal_in[pred[1].lower()] = pred[2].lower()

    data = {
        "trucks": trucks,
        "airplanes": airplanes,
        "packages": packages,
        "airports": airports,
        "at": at,
        "in_vehicle": in_vehicle,
        "city_of": city_of,
        "goal_at": goal_at,
        "goal_in": goal_in,
    }
    return _normalize_log_locations(data)


def _normalize_log_locations(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rename locations so that each city's airport becomes l{city_idx}-0 and
    the remaining locations are numbered l{city_idx}-1, l{city_idx}-2, etc.

    This is required because PlanBench instances place airports at arbitrary
    suffixes (e.g. l0-2, l1-3), while our simulator's is_airport() method
    identifies airports by the '-0' suffix.
    """
    city_of: Dict[str, str] = data["city_of"]
    airports: set = data["airports"]

    # Group locations by city
    from collections import defaultdict
    city_locs: Dict[str, List[str]] = defaultdict(list)
    for loc, city in city_of.items():
        city_locs[city].append(loc)

    # Build location rename map
    rename: Dict[str, str] = {}
    for city in sorted(city_locs):
        locs = city_locs[city]
        airport_locs = [l for l in locs if l in airports]
        non_airport_locs = sorted(l for l in locs if l not in airports)

        # Extract city index from city name (c0->0, c1->1, etc.)
        city_idx = re.sub(r"[^0-9]", "", city) or "0"

        # Airport -> l{idx}-0, others -> l{idx}-1, l{idx}-2, ...
        airport = airport_locs[0] if airport_locs else None
        if airport:
            rename[airport] = f"l{city_idx}-0"
        for i, loc in enumerate(non_airport_locs, start=1):
            rename[loc] = f"l{city_idx}-{i}"

    if not rename or rename == {loc: loc for loc in rename}:
        return data  # no renaming needed

    def r(loc: str) -> str:
        return rename.get(loc, loc)

    new_at = {entity: r(loc) for entity, loc in data["at"].items()}
    new_city_of = {r(loc): city for loc, city in city_of.items()}
    new_airports = {r(a) for a in airports}
    new_goal_at = {pkg: r(loc) for pkg, loc in data["goal_at"].items()}

    return {
        "trucks": data["trucks"],
        "airplanes": data["airplanes"],
        "packages": data["packages"],
        "airports": new_airports,
        "at": new_at,
        "in_vehicle": data["in_vehicle"],
        "city_of": new_city_of,
        "goal_at": new_goal_at,
        "goal_in": data["goal_in"],
    }


def _log_to_nl(data: Dict[str, Any]) -> Tuple[str, str, Dict[str, str]]:
    """Convert parsed logistics PDDL to (initial_nl, goal_nl, city_map)."""
    def fmt_loc(loc: str) -> str:
        return loc.upper()

    def fmt_pkg(p: str) -> str:
        # p0 -> P0
        return p[0].upper() + p[1:]

    def fmt_truck(t: str) -> str:
        return t[0].upper() + t[1:]

    def fmt_plane(a: str) -> str:
        return a[0].upper() + a[1:]

    parts: List[str] = []

    # Packages: at location or in vehicle
    for pkg in sorted(data["packages"]):
        if pkg in data["in_vehicle"]:
            v = data["in_vehicle"][pkg]
            if v in data["trucks"]:
                parts.append(f"Package {fmt_pkg(pkg)} is in Truck {fmt_truck(v)}")
            else:
                parts.append(f"Package {fmt_pkg(pkg)} is in Airplane {fmt_plane(v)}")
        elif pkg in data["at"]:
            parts.append(f"Package {fmt_pkg(pkg)} is at {fmt_loc(data['at'][pkg])}")

    # Trucks
    for t in sorted(data["trucks"]):
        if t in data["at"]:
            parts.append(f"Truck {fmt_truck(t)} is at {fmt_loc(data['at'][t])}")

    # Airplanes
    for a in sorted(data["airplanes"]):
        if a in data["at"]:
            parts.append(f"Airplane {fmt_plane(a)} is at {fmt_loc(data['at'][a])}")

    initial_nl = ". ".join(parts) + "."

    # Goal (only packages typically)
    goal_parts: List[str] = []
    for pkg in sorted(data["goal_at"]):
        goal_parts.append(f"Package {fmt_pkg(pkg)} is at {fmt_loc(data['goal_at'][pkg])}")
    for pkg, v in sorted(data["goal_in"].items()):
        if v in data["trucks"]:
            goal_parts.append(f"Package {fmt_pkg(pkg)} is in Truck {fmt_truck(v)}")
        else:
            goal_parts.append(f"Package {fmt_pkg(pkg)} is in Airplane {fmt_plane(v)}")
    goal_nl = ". ".join(goal_parts) + "."

    def fmt_city(c: str) -> str:
        return c[0].upper() + c[1:]  # c0 -> C0

    # city_map: location (uppercase) -> city (uppercase)
    city_map = {fmt_loc(loc): fmt_city(city) for loc, city in data["city_of"].items()}

    return initial_nl, goal_nl, city_map


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_blocksworld_problems(
    n: int,
    seed: int = 42,
    cache_dir: str = ".planbench_cache",
) -> List[Dict[str, Any]]:
    """
    Download n random PlanBench Blocksworld instances.

    Returns a list of dicts with keys: instance_id, initial_state, goal_state.
    """
    rng = random.Random(seed)
    indices = rng.sample(range(1, _MAX_INSTANCE + 1), min(n, _MAX_INSTANCE))

    results = []
    for idx in indices:
        url = f"{_GITHUB_RAW}/{_BW_PATH.format(n=idx)}"
        cache_path = Path(cache_dir) / "blocksworld" / f"instance-{idx}.pddl"
        text = _fetch_pddl(url, cache_path)
        data = _parse_bw_pddl(text)
        init_nl, goal_nl = _bw_to_nl(data)
        results.append({
            "instance_id": f"bw-{idx}",
            "initial_state": init_nl,
            "goal_state": goal_nl,
        })

    return results


def get_logistics_problems(
    n: int,
    seed: int = 42,
    cache_dir: str = ".planbench_cache",
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Download n random PlanBench Logistics instances.

    Returns (list_of_dicts, city_map).
    Each dict has keys: instance_id, initial_state, goal_state.
    city_map is a union of all city_maps from selected instances (they share
    the same location naming convention so this is safe).
    """
    rng = random.Random(seed)
    indices = rng.sample(range(1, _MAX_INSTANCE + 1), min(n, _MAX_INSTANCE))

    results = []
    combined_city_map: Dict[str, str] = {}

    for idx in indices:
        url = f"{_GITHUB_RAW}/{_LOG_PATH.format(n=idx)}"
        cache_path = Path(cache_dir) / "logistics" / f"instance-{idx}.pddl"
        text = _fetch_pddl(url, cache_path)
        data = _parse_log_pddl(text)
        init_nl, goal_nl, city_map = _log_to_nl(data)
        combined_city_map.update(city_map)
        results.append({
            "instance_id": f"log-{idx}",
            "initial_state": init_nl,
            "goal_state": goal_nl,
            "city_map": city_map,
        })

    return results, combined_city_map
