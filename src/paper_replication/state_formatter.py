"""Convert PlanBench NL state descriptions to the compact notation used in the AoT+ paper.

Paper format:
  Initial: "D on A on C on B on table, E on table."
  Goal:    "A on D on B."  (no trailing 'table')
"""
from __future__ import annotations
import re
from typing import Dict, List, Tuple


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
