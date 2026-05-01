"""Logistics benchmark problem generator.

Difficulty tiers:
  easy   — 1 city, 2 packages, 1 truck  (short drives only)
  medium — 2 cities, 3 packages, 2 trucks, 1 airplane
  hard   — 3 cities, 4 packages, 3 trucks, 2 airplanes
"""

import random
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Minimal state + plan representation
# ---------------------------------------------------------------------------

def _pkg_desc(packages: Dict[str, str]) -> str:
    return ", ".join(f"Package {p} is at {loc}" for p, loc in sorted(packages.items()))


def _vehicle_desc(vehicles: Dict[str, str]) -> str:
    return ", ".join(f"{'Airplane' if v.startswith('A') else 'Truck'} {v} is at {loc}"
                     for v, loc in sorted(vehicles.items()))


def _full_state_desc(packages: Dict[str, str], vehicles: Dict[str, str]) -> str:
    parts = [_pkg_desc(packages)]
    vdesc = _vehicle_desc(vehicles)
    if vdesc:
        parts.append(vdesc)
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Problem generation helpers
# ---------------------------------------------------------------------------

def _city_of(loc: str, city_locations: Dict[str, List[str]]) -> str:
    for city, locs in city_locations.items():
        if loc in locs:
            return city
    return loc  # loc is already a city (airport)


def _airport_of(city: str) -> str:
    return f"Airport_{city}"


def _generate_easy_problem(rng: random.Random) -> dict:
    """1 city, 2 packages, 1 truck, drive actions only."""
    city = "CityA"
    locations = ["Loc_A1", "Loc_A2", "Loc_A3"]
    airport = _airport_of(city)
    all_locs = locations + [airport]
    city_locations = {city: all_locs}

    pkg_names = ["P1", "P2"]
    truck = "T1"

    def rand_locs(n):
        return [rng.choice(all_locs) for _ in range(n)]

    # Initial placement
    pkg_init = {p: rng.choice(all_locs) for p in pkg_names}
    truck_init = {truck: rng.choice(all_locs)}

    # Goal: deliver packages to different locations
    pkg_goal = {}
    for p in pkg_names:
        candidates = [l for l in all_locs if l != pkg_init[p]]
        pkg_goal[p] = rng.choice(candidates)

    plan = _plan_easy(pkg_init, pkg_goal, truck_init, truck, all_locs, city_locations)

    initial_state = _full_state_desc(pkg_init, truck_init)
    goal_state = _pkg_desc(pkg_goal)

    return {
        "initial_state": initial_state,
        "goal_state": goal_state,
        "optimal_plan": plan,
        "optimal_length": len(plan),
        "tier": "easy",
        "city_locations": city_locations,
        "pkg_init": pkg_init,
        "pkg_goal": pkg_goal,
        "truck_init": truck_init,
        "trucks": [truck],
        "airplanes": [],
    }


def _plan_easy(pkg_init, pkg_goal, truck_init, truck, all_locs, city_locs):
    """Greedy plan for single-city logistics."""
    steps = []
    pkg_loc = dict(pkg_init)
    truck_loc = dict(truck_init)
    truck_cargo = {truck: None}

    for pkg, goal_loc in pkg_goal.items():
        if pkg_loc[pkg] == goal_loc:
            continue
        # Drive truck to package
        if truck_loc[truck] != pkg_loc[pkg]:
            steps.append(f"Drive Truck {truck} from {truck_loc[truck]} to {pkg_loc[pkg]}")
            truck_loc[truck] = pkg_loc[pkg]
        # Load
        steps.append(f"Load Package {pkg} onto Truck {truck} at {truck_loc[truck]}")
        truck_cargo[truck] = pkg
        # Drive to goal
        steps.append(f"Drive Truck {truck} from {truck_loc[truck]} to {goal_loc}")
        truck_loc[truck] = goal_loc
        # Unload
        steps.append(f"Unload Package {pkg} from Truck {truck} at {goal_loc}")
        pkg_loc[pkg] = goal_loc
        truck_cargo[truck] = None

    return steps


def _generate_medium_problem(rng: random.Random) -> dict:
    """2 cities, 3 packages, 2 trucks, 1 airplane."""
    cities = ["CityA", "CityB"]
    city_locations = {
        "CityA": ["Loc_A1", "Loc_A2", _airport_of("CityA")],
        "CityB": ["Loc_B1", "Loc_B2", _airport_of("CityB")],
    }
    all_locs = [l for locs in city_locations.values() for l in locs]
    trucks = ["T1", "T2"]
    airplane = "Plane1"

    pkg_names = ["P1", "P2", "P3"]
    pkg_init = {p: rng.choice(all_locs) for p in pkg_names}
    pkg_goal = {}
    for p in pkg_names:
        candidates = [l for l in all_locs if l != pkg_init[p]]
        pkg_goal[p] = rng.choice(candidates)

    truck_init = {
        "T1": rng.choice(city_locations["CityA"]),
        "T2": rng.choice(city_locations["CityB"]),
    }
    plane_init = {airplane: rng.choice([_airport_of(c) for c in cities])}
    vehicle_init = {**truck_init, **plane_init}

    plan = _plan_multi_city(pkg_init, pkg_goal, truck_init, plane_init,
                            city_locations, cities)
    initial_state = _full_state_desc(pkg_init, vehicle_init)
    goal_state = _pkg_desc(pkg_goal)

    return {
        "initial_state": initial_state,
        "goal_state": goal_state,
        "optimal_plan": plan,
        "optimal_length": len(plan),
        "tier": "medium",
        "city_locations": city_locations,
        "pkg_init": pkg_init,
        "pkg_goal": pkg_goal,
        "truck_init": truck_init,
        "plane_init": plane_init,
        "trucks": trucks,
        "airplanes": [airplane],
    }


def _generate_hard_problem(rng: random.Random) -> dict:
    """3 cities, 4 packages, 3 trucks, 2 airplanes."""
    cities = ["CityA", "CityB", "CityC"]
    city_locations = {
        "CityA": ["Loc_A1", "Loc_A2", _airport_of("CityA")],
        "CityB": ["Loc_B1", "Loc_B2", _airport_of("CityB")],
        "CityC": ["Loc_C1", "Loc_C2", _airport_of("CityC")],
    }
    all_locs = [l for locs in city_locations.values() for l in locs]
    trucks = ["T1", "T2", "T3"]
    airplanes = ["Plane1", "Plane2"]

    pkg_names = ["P1", "P2", "P3", "P4"]
    pkg_init = {p: rng.choice(all_locs) for p in pkg_names}
    pkg_goal = {}
    for p in pkg_names:
        candidates = [l for l in all_locs if l != pkg_init[p]]
        pkg_goal[p] = rng.choice(candidates)

    truck_city_map = {"T1": "CityA", "T2": "CityB", "T3": "CityC"}
    truck_init = {t: rng.choice(city_locations[truck_city_map[t]]) for t in trucks}
    plane_init = {pl: _airport_of(rng.choice(cities)) for pl in airplanes}
    vehicle_init = {**truck_init, **plane_init}

    plan = _plan_multi_city(pkg_init, pkg_goal, truck_init, plane_init,
                            city_locations, cities)
    initial_state = _full_state_desc(pkg_init, vehicle_init)
    goal_state = _pkg_desc(pkg_goal)

    return {
        "initial_state": initial_state,
        "goal_state": goal_state,
        "optimal_plan": plan,
        "optimal_length": len(plan),
        "tier": "hard",
        "city_locations": city_locations,
        "pkg_init": pkg_init,
        "pkg_goal": pkg_goal,
        "truck_init": truck_init,
        "plane_init": plane_init,
        "trucks": trucks,
        "airplanes": airplanes,
    }


def _city_of_loc(loc: str, city_locations: Dict[str, List[str]]) -> str:
    for city, locs in city_locations.items():
        if loc in locs:
            return city
    raise ValueError(f"Location {loc!r} not found in any city")


def _plan_multi_city(pkg_init, pkg_goal, truck_init, plane_init, city_locations, cities):
    """Greedy sequential plan: handle each package one at a time."""
    steps = []
    pkg_loc = dict(pkg_init)
    truck_loc = dict(truck_init)
    plane_loc = dict(plane_init)

    planes = list(plane_init.keys())

    for pkg, goal_loc in pkg_goal.items():
        if pkg_loc[pkg] == goal_loc:
            continue

        src_city = _city_of_loc(pkg_loc[pkg], city_locations)
        dst_city = _city_of_loc(goal_loc, city_locations)

        if src_city == dst_city:
            # Same city: just use a truck
            truck = _closest_truck(truck_loc, pkg_loc[pkg], city_locations, src_city)
            if truck is None:
                continue
            steps += _truck_deliver(pkg, truck, pkg_loc, truck_loc, goal_loc)
        else:
            # Cross-city: truck → airport → plane → airport → truck
            src_airport = _airport_of(src_city)
            dst_airport = _airport_of(dst_city)

            # Truck in source city to pick up and bring to airport
            src_truck = _closest_truck(truck_loc, pkg_loc[pkg], city_locations, src_city)
            if src_truck:
                steps += _truck_deliver(pkg, src_truck, pkg_loc, truck_loc, src_airport)

            # Fly to destination city
            plane = _pick_plane(plane_loc, src_airport, planes)
            if plane_loc[plane] != src_airport:
                steps.append(f"Fly Airplane {plane} from {plane_loc[plane]} to {src_airport}")
                plane_loc[plane] = src_airport
            steps.append(f"Load Package {pkg} onto Airplane {plane} at {src_airport}")
            steps.append(f"Fly Airplane {plane} from {src_airport} to {dst_airport}")
            plane_loc[plane] = dst_airport
            steps.append(f"Unload Package {pkg} from Airplane {plane} at {dst_airport}")
            pkg_loc[pkg] = dst_airport

            # Truck in destination city to deliver from airport
            dst_truck = _closest_truck(truck_loc, dst_airport, city_locations, dst_city)
            if dst_truck and goal_loc != dst_airport:
                steps += _truck_deliver(pkg, dst_truck, pkg_loc, truck_loc, goal_loc)

    return steps


def _closest_truck(truck_loc, target_loc, city_locations, city):
    city_locs = set(city_locations[city])
    for truck, loc in truck_loc.items():
        if loc in city_locs:
            return truck
    return None


def _pick_plane(plane_loc, preferred_airport, planes):
    for plane in planes:
        if plane_loc[plane] == preferred_airport:
            return plane
    return planes[0]


def _truck_deliver(pkg, truck, pkg_loc, truck_loc, goal_loc):
    steps = []
    if truck_loc[truck] != pkg_loc[pkg]:
        steps.append(f"Drive Truck {truck} from {truck_loc[truck]} to {pkg_loc[pkg]}")
        truck_loc[truck] = pkg_loc[pkg]
    steps.append(f"Load Package {pkg} onto Truck {truck} at {truck_loc[truck]}")
    if truck_loc[truck] != goal_loc:
        steps.append(f"Drive Truck {truck} from {truck_loc[truck]} to {goal_loc}")
        truck_loc[truck] = goal_loc
    steps.append(f"Unload Package {pkg} from Truck {truck} at {goal_loc}")
    pkg_loc[pkg] = goal_loc
    return steps


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_logistics_problems(
    tier: str,
    n_problems: int = 20,
    seed: int = 42,
) -> List[dict]:
    """Generate Logistics problems for a difficulty tier."""
    generators = {
        "easy":   _generate_easy_problem,
        "medium": _generate_medium_problem,
        "hard":   _generate_hard_problem,
    }
    if tier not in generators:
        raise ValueError(f"tier must be 'easy', 'medium', or 'hard', got {tier!r}")

    rng = random.Random(seed)
    return [generators[tier](rng) for _ in range(n_problems)]


def get_all_logistics_problems(n_per_tier: int = 20, seed: int = 42) -> Dict[str, List[dict]]:
    return {
        "easy":   generate_logistics_problems("easy",   n_per_tier, seed),
        "medium": generate_logistics_problems("medium", n_per_tier, seed + 1),
        "hard":   generate_logistics_problems("hard",   n_per_tier, seed + 2),
    }
