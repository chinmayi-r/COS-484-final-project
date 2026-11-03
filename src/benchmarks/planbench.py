"""
PlanBench instance downloader and PDDL parser.

Downloads Blocksworld and Logistics problem instances from:
  https://github.com/karthikv792/LLMs-Planning

Parses PDDL into the NL format expected by our evaluators:

Blocksworld NL state:
    "Block A is on the table. Block B is on Block A. Block A is clear. ..."

Logistics NL state (same PDDL-style location names as our sim):
    "Package p0 is at l0-2. Truck t0 is at l0-1. Airplane a0 is at l0-0."

Usage:
    from src.benchmarks.planbench import get_blocksworld_problems, get_logistics_problems

    bw_problems = get_blocksworld_problems(n=20, seed=42, cache_dir=".planbench_cache")
    # returns list of {"initial_state": str, "goal_state": str}

    log_problems, city_map = get_logistics_problems(n=15, seed=42, cache_dir=".planbench_cache")
    # returns (list of {"initial_state": str, "goal_state": str}, city_map dict)
"""
from __future__ import annotations

import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import requests

# ---------------------------------------------------------------------------
# Raw GitHub base URLs
# ---------------------------------------------------------------------------
_BASE = "https://raw.githubusercontent.com/karthikv792/LLMs-Planning/main/plan-bench/instances"
BW_URL   = _BASE + "/blocksworld/generated_basic/instance-{n}.pddl"
LOG_URL  = _BASE + "/logistics/generated_basic/instance-{n}.pddl"
BW_TOTAL  = 189   # files go 1..189
LOG_TOTAL = 285   # files go 1..285


# ---------------------------------------------------------------------------
# Fetch / cache
# ---------------------------------------------------------------------------

def _fetch(url: str, cache_dir: str) -> str:
    """Return file contents, using a local cache to avoid repeat downloads."""
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split("/")[-1] + "_" + url.split("/")[-2])
    if os.path.exists(fname):
        with open(fname) as f:
            return f.read()
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    text = resp.text
    with open(fname, "w") as f:
        f.write(text)
    return text


# ---------------------------------------------------------------------------
# PDDL parsing helpers
# ---------------------------------------------------------------------------

def _strip_comments(pddl: str) -> str:
    return re.sub(r";[^\n]*", "", pddl)


def _extract_section(pddl: str, keyword: str) -> str:
    """Extract the content of a top-level PDDL section like :init or :goal.

    Uses parenthesis counting so nested structures are handled correctly.
    keyword should include the leading colon, e.g. ':init' or ':goal'.
    """
    # Normalise: strip leading colon if caller passes plain name
    kw = keyword.lstrip(":").lower()
    needle = f"(:{kw}"
    idx = pddl.lower().find(needle)
    if idx == -1:
        return ""
    # Walk forward counting parens; return everything inside the outer parens
    depth = 0
    content_start = idx + len(needle)  # first char after '(:keyword'
    for i in range(idx, len(pddl)):
        if pddl[i] == "(":
            depth += 1
        elif pddl[i] == ")":
            depth -= 1
            if depth == 0:
                return pddl[content_start:i]
    return ""


def _predicates(section: str) -> List[List[str]]:
    """Parse a PDDL section into a list of predicate token lists."""
    results = []
    for m in re.finditer(r"\(([^()]+)\)", section):
        tokens = m.group(1).split()
        if tokens:
            results.append(tokens)
    return results


# ---------------------------------------------------------------------------
# Blocksworld PDDL → NL
# ---------------------------------------------------------------------------

def _parse_blocksworld_pddl(pddl: str) -> Tuple[str, str]:
    """
    Parse a Blocksworld PDDL file into (initial_state_nl, goal_state_nl).

    Block names from PDDL (a, b, c …) are upper-cased (A, B, C …) to match
    the NL format our blocksworld sim expects.
    """
    pddl = _strip_comments(pddl)

    init_section = _extract_section(pddl, ":init")
    goal_section = _extract_section(pddl, ":goal")

    def nl_from_predicates(preds: List[List[str]]) -> str:
        parts = []
        for tokens in preds:
            name = tokens[0].lower()
            args = [t.upper() for t in tokens[1:]]
            if name == "ontable":
                parts.append(f"Block {args[0]} is on the table")
            elif name == "on":
                parts.append(f"Block {args[0]} is on Block {args[1]}")
            elif name == "clear":
                parts.append(f"Block {args[0]} is clear")
            elif name == "handempty":
                parts.append("The hand is empty")
            # skip holding — shouldn't appear in initial/goal
        return ". ".join(parts) + ("." if parts else "")

    init_nl = nl_from_predicates(_predicates(init_section))
    goal_nl = nl_from_predicates(_predicates(goal_section))
    return init_nl, goal_nl


# ---------------------------------------------------------------------------
# Logistics PDDL → NL + city_map
# ---------------------------------------------------------------------------

def _parse_logistics_pddl(pddl: str) -> Tuple[str, str, Dict[str, str]]:
    """
    Parse a Logistics PDDL file into (initial_state_nl, goal_state_nl, city_map).

    city_map: {location: city}  (e.g. {"l0-0": "c0", "l0-1": "c0", ...})
    Airports are locations declared with (AIRPORT loc).

    NL format mirrors what our logistics sim's nl_state_to_canonical expects:
        "Package p0 is at l0-2. Truck t0 is at l1-1. Airplane a0 is at l0-0."
    """
    pddl = _strip_comments(pddl)

    init_section = _extract_section(pddl, ":init")
    goal_section = _extract_section(pddl, ":goal")

    init_preds = _predicates(init_section)
    goal_preds = _predicates(goal_section)

    # Collect type info from :init
    airplanes: set = set()
    trucks: set    = set()
    packages: set  = set()
    locations: set = set()
    airports: set  = set()
    city_map: Dict[str, str] = {}

    at_init: Dict[str, str] = {}    # entity → location
    in_init: Dict[str, str] = {}    # package → vehicle

    for tokens in init_preds:
        name = tokens[0].upper()
        args = [t.lower() for t in tokens[1:]]
        if name == "AIRPLANE":
            airplanes.add(args[0])
        elif name == "TRUCK":
            trucks.add(args[0])
        elif name == "OBJ":
            packages.add(args[0])
        elif name == "LOCATION":
            locations.add(args[0])
        elif name == "AIRPORT":
            airports.add(args[0])
        elif name == "IN-CITY" and len(args) == 2:
            city_map[args[0]] = args[1]
        elif name == "AT" and len(args) == 2:
            at_init[args[0]] = args[1]
        elif name == "IN" and len(args) == 2:
            in_init[args[0]] = args[1]

    def _entity_type(e: str) -> str:
        if e in airplanes:
            return "Airplane"
        if e in trucks:
            return "Truck"
        if e in packages:
            return "Package"
        return "Unknown"

    # Build initial state NL
    init_parts = []
    for entity, loc in sorted(at_init.items()):
        etype = _entity_type(entity)
        init_parts.append(f"{etype} {entity} is at {loc}")
    for pkg, vehicle in sorted(in_init.items()):
        vtype = _entity_type(vehicle)
        init_parts.append(f"Package {pkg} is in {vtype} {vehicle}")
    init_nl = ". ".join(init_parts) + ("." if init_parts else "")

    # Build goal state NL (usually just package locations)
    goal_parts = []
    for tokens in goal_preds:
        name = tokens[0].upper()
        args = [t.lower() for t in tokens[1:]]
        if name == "AT" and len(args) == 2:
            etype = _entity_type(args[0])
            goal_parts.append(f"{etype} {args[0]} is at {args[1]}")
        elif name == "IN" and len(args) == 2:
            vtype = _entity_type(args[1])
            goal_parts.append(f"Package {args[0]} is in {vtype} {args[1]}")
    goal_nl = ". ".join(goal_parts) + ("." if goal_parts else "")

    return init_nl, goal_nl, city_map


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_blocksworld_problems(
    n: int = 20,
    seed: int = 42,
    instance_ids: Optional[List[int]] = None,
    cache_dir: str = ".planbench_cache",
) -> List[Dict[str, str]]:
    """
    Download and parse n Blocksworld PlanBench instances.

    Args:
        n: number of problems to return
        seed: random seed for instance selection (ignored if instance_ids given)
        instance_ids: explicit list of 1-based instance indices to use
        cache_dir: local directory for caching downloaded PDDL files

    Returns:
        List of dicts with keys "initial_state" and "goal_state" (NL strings).
    """
    if instance_ids is None:
        rng = random.Random(seed)
        instance_ids = rng.sample(range(1, BW_TOTAL + 1), min(n, BW_TOTAL))
    instance_ids = instance_ids[:n]

    problems = []
    for idx in instance_ids:
        url = BW_URL.format(n=idx)
        pddl = _fetch(url, os.path.join(cache_dir, "blocksworld"))
        init_nl, goal_nl = _parse_blocksworld_pddl(pddl)
        problems.append({
            "initial_state": init_nl,
            "goal_state":    goal_nl,
            "instance_id":   idx,
        })
    return problems


def get_logistics_problems(
    n: int = 15,
    seed: int = 42,
    instance_ids: Optional[List[int]] = None,
    cache_dir: str = ".planbench_cache",
) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
    """
    Download and parse n Logistics PlanBench instances.

    Returns:
        (problems, city_map) where city_map is merged across all instances
        (safe because all instances share the same location→city mapping).
    """
    if instance_ids is None:
        rng = random.Random(seed)
        instance_ids = rng.sample(range(1, LOG_TOTAL + 1), min(n, LOG_TOTAL))
    instance_ids = instance_ids[:n]

    problems = []
    merged_city_map: Dict[str, str] = {}

    for idx in instance_ids:
        url = LOG_URL.format(n=idx)
        pddl = _fetch(url, os.path.join(cache_dir, "logistics"))
        init_nl, goal_nl, city_map = _parse_logistics_pddl(pddl)
        merged_city_map.update(city_map)
        problems.append({
            "initial_state": init_nl,
            "goal_state":    goal_nl,
            "instance_id":   idx,
        })
    return problems, merged_city_map
