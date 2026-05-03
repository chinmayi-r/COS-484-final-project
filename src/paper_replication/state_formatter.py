"""Convert PlanBench NL state descriptions to the compact notation used in the AoT+ paper.

Blocksworld format:
  Initial: "D on A on C on B on table, E on table."
  Goal:    "A on D on B."  (no trailing 'table')

Logistics format:
  Initial: "Airplanes: A0 at L1-0\nTrucks: T0 at L0-1\nPackages: P0 at L2-0, ..."
  Goal:    "P0 at L2-0, P1 at L0-1, ..."
"""
from __future__ import annotations
import re
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Blocksworld
# ---------------------------------------------------------------------------

def _parse_bw_nl(nl: str) -> Tuple[Dict[str, str], List[str]]:
    """Parse NL like 'Block A is on Block B. Block B is on the table. Block A is clear.'
    Returns (on_dict, clear_list) with uppercase block names."""
    blocks: set[str] = set()
    for m in re.finditer(r"Block ([A-Za-z0-9]+)", nl, re.IGNORECASE):
        blocks.add(m.group(1).upper())

    on: Dict[str, str] = {b: "table" for b in blocks}
    for m in re.finditer(
        r"Block ([A-Za-z0-9]+) is on (?:Block ([A-Za-z0-9]+)|the table|table)",
        nl, re.IGNORECASE,
    ):
        block = m.group(1).upper()
        dest = m.group(2).upper() if m.group(2) else "table"
        on[block] = dest

    occupied = {v for v in on.values() if v != "table"}
    clear = sorted(b for b in blocks if b not in occupied)
    return on, clear


def bw_nl_to_compact_initial(nl: str) -> str:
    """Convert NL initial-state description to compact paper format.

    Each stack is written top-to-bottom ending with 'table'.
    Multiple stacks are comma-separated.
    Example: 'D on A on C on B on table, E on table.'
    """
    on, clear = _parse_bw_nl(nl)

    chains: List[str] = []
    for top in sorted(clear):
        parts = [top]
        cur = on[top]
        while cur != "table":
            parts.append(cur)
            cur = on.get(cur, "table")
        parts.append("table")
        chains.append(" on ".join(parts))

    return ", ".join(chains) + "."


def bw_nl_to_compact_goal(nl: str) -> str:
    """Convert NL goal description to compact paper format.

    Same chain tracing but 'table' is omitted at the end (paper convention).
    Multiple chains are comma-separated.
    Example: 'A on D on B.'
    """
    on, clear = _parse_bw_nl(nl)

    chains: List[str] = []
    for top in sorted(clear):
        parts = [top]
        cur = on[top]
        while cur != "table":
            parts.append(cur)
            cur = on.get(cur, "table")
        chains.append(" on ".join(parts))

    return ", ".join(chains) + "."


# ---------------------------------------------------------------------------
# Logistics
# ---------------------------------------------------------------------------

def log_nl_to_compact(init_nl: str) -> str:
    """Convert PlanBench logistics NL initial state to the paper's block format.

    Returns a multi-line string:
        Airplanes: A0 at L1-0, A1 at L2-0
        Trucks: T0 at L0-1, T1 at L1-0
        Packages: P0 at L2-0, P1 at L0-1

    Preserves original capitalisation of names from the NL text.
    """
    airplanes: List[Tuple[str, str]] = []
    trucks: List[Tuple[str, str]] = []
    packages: List[Tuple[str, str]] = []

    for m in re.finditer(r"Airplane\s+(\S+?)\s+is\s+at\s+(\S+?)\.?(?:\s|$)", init_nl, re.IGNORECASE):
        airplanes.append((m.group(1), m.group(2).rstrip(".")))
    for m in re.finditer(r"Truck\s+(\S+?)\s+is\s+at\s+(\S+?)\.?(?:\s|$)", init_nl, re.IGNORECASE):
        trucks.append((m.group(1), m.group(2).rstrip(".")))
    for m in re.finditer(r"Package\s+(\S+?)\s+is\s+at\s+(\S+?)\.?(?:\s|$)", init_nl, re.IGNORECASE):
        packages.append((m.group(1), m.group(2).rstrip(".")))

    airplanes.sort(key=lambda x: x[0].upper())
    trucks.sort(key=lambda x: x[0].upper())
    packages.sort(key=lambda x: x[0].upper())

    lines: List[str] = []
    if airplanes:
        lines.append("Airplanes: " + ", ".join(f"{n} at {l}" for n, l in airplanes))
    if trucks:
        lines.append("Trucks: "    + ", ".join(f"{n} at {l}" for n, l in trucks))
    if packages:
        lines.append("Packages: "  + ", ".join(f"{n} at {l}" for n, l in packages))

    return "\n".join(lines)


def log_goal_nl_to_compact(goal_nl: str) -> str:
    """Convert PlanBench logistics goal NL to the paper's goal string.

    Returns: 'P0 at L2-0, P1 at L0-1, ...'
    """
    parts: List[Tuple[str, str]] = []
    for m in re.finditer(r"Package\s+(\S+?)\s+is\s+at\s+(\S+?)\.?(?:\s|$)", goal_nl, re.IGNORECASE):
        parts.append((m.group(1), m.group(2).rstrip(".")))
    parts.sort(key=lambda x: x[0].upper())
    return ", ".join(f"{n} at {l}" for n, l in parts)


def derive_city_map(init_nl: str, goal_nl: str = "") -> Dict[str, str]:
    """Build a city_map dict from all Lx-y location names in init+goal NL.

    Returns lowercase keys/values: {'l0-0': 'c0', 'l0-1': 'c0', 'l1-0': 'c1', ...}
    """
    combined = init_nl + " " + goal_nl
    city_map: Dict[str, str] = {}
    for m in re.finditer(r"L(\d+)-\d+", combined, re.IGNORECASE):
        loc = m.group(0).lower()      # e.g. "l0-1"
        city = f"c{m.group(1)}"       # e.g. "c0"
        city_map[loc] = city
    return city_map



def _parse_bw_nl(nl: str) -> Tuple[Dict[str, str], List[str]]:
    """Parse NL like 'Block A is on Block B. Block B is on the table. Block A is clear.'
    Returns (on_dict, clear_list) with uppercase block names."""
    blocks: set[str] = set()
    for m in re.finditer(r"Block ([A-Za-z0-9]+)", nl, re.IGNORECASE):
        blocks.add(m.group(1).upper())

    on: Dict[str, str] = {b: "table" for b in blocks}
    for m in re.finditer(
        r"Block ([A-Za-z0-9]+) is on (?:Block ([A-Za-z0-9]+)|the table|table)",
        nl, re.IGNORECASE,
    ):
        block = m.group(1).upper()
        dest = m.group(2).upper() if m.group(2) else "table"
        on[block] = dest

    occupied = {v for v in on.values() if v != "table"}
    clear = sorted(b for b in blocks if b not in occupied)
    return on, clear


def bw_nl_to_compact_initial(nl: str) -> str:
    """Convert NL initial-state description to compact paper format.

    Each stack is written top-to-bottom ending with 'table'.
    Multiple stacks are comma-separated.
    Example: 'D on A on C on B on table, E on table.'
    """
    on, clear = _parse_bw_nl(nl)

    chains: List[str] = []
    for top in sorted(clear):
        parts = [top]
        cur = on[top]
        while cur != "table":
            parts.append(cur)
            cur = on.get(cur, "table")
        parts.append("table")
        chains.append(" on ".join(parts))

    return ", ".join(chains) + "."


def bw_nl_to_compact_goal(nl: str) -> str:
    """Convert NL goal description to compact paper format.

    Same chain tracing but 'table' is omitted at the end (paper convention).
    Multiple chains are comma-separated.
    Example: 'A on D on B.'
    """
    on, clear = _parse_bw_nl(nl)

    chains: List[str] = []
    for top in sorted(clear):
        parts = [top]
        cur = on[top]
        while cur != "table":
            parts.append(cur)
            cur = on.get(cur, "table")
        chains.append(" on ".join(parts))

    return ", ".join(chains) + "."
