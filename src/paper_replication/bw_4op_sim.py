"""Blocksworld 4-operator simulator and PDDL plan extractor for paper replication.

Operators: pick-up, put-down, unstack, stack  (exactly as in the AoT+ paper).
State dict schema:
  {
    "on":       {block: "table" | block_name},   # block that is held has on[X]="held"
    "clear":    [block, ...],
    "holding":  None | block_name,
    "handempty": bool
  }
"""
from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# State construction
# ---------------------------------------------------------------------------

def nl_to_4op_state(nl: str) -> Dict[str, Any]:
    """Parse NL blocksworld description ('Block A is on Block B...') into 4-op state dict."""
    blocks: list[str] = sorted(
        {m.group(1).upper() for m in re.finditer(r"Block ([A-Za-z0-9]+)", nl, re.IGNORECASE)}
    )
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

    return {"on": on, "clear": clear, "holding": None, "handempty": True}


# ---------------------------------------------------------------------------
# PDDL action parsing
# ---------------------------------------------------------------------------

def parse_pddl_bw_action(line: str) -> Tuple[str, List[str]]:
    """Parse one PDDL action line like '(unstack D A)' → ('unstack', ['D', 'A']).

    Case-insensitive; args are uppercased. Raises ValueError on failure.
    """
    line = line.strip()
    m = re.match(r"^\(\s*([a-zA-Z\-]+)((?:\s+\S+)*)\s*\)$", line)
    if not m:
        raise ValueError(f"Cannot parse PDDL action: {line!r}")
    op = m.group(1).lower()
    args = [a.upper() for a in m.group(2).split() if a]
    return op, args


def extract_pddl_plan(text: str) -> List[str]:
    """Return the raw PDDL lines appearing after 'The PDDL version:' in *text*.

    Stops at a blank line or end of text. Returns empty list if marker absent.
    """
    marker = re.search(r"The PDDL version\s*:", text, re.IGNORECASE)
    if not marker:
        return []

    after = text[marker.end():].lstrip("\n")
    lines: List[str] = []
    for raw in after.split("\n"):
        stripped = raw.strip()
        if not stripped:
            break  # stop at first blank line
        if re.match(r"^\(", stripped):
            lines.append(stripped)
        else:
            break  # stop at non-PDDL content
    return lines


# ---------------------------------------------------------------------------
# 4-op action application
# ---------------------------------------------------------------------------

def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def apply_4op_action(state: Dict[str, Any], op: str, args: List[str]) -> Dict[str, Any]:
    """Apply one 4-op action.  Returns a new state dict or raises ValueError."""
    s = copy.deepcopy(state)
    clear_set: set[str] = set(s["clear"])

    if op == "pick-up":
        _check(len(args) == 1, f"pick-up needs 1 arg, got {args}")
        X = args[0]
        _check(s["handempty"], f"pick-up {X}: hand not empty (holding {s['holding']})")
        _check(s["on"].get(X) == "table", f"pick-up {X}: {X} is not on the table")
        _check(X in clear_set, f"pick-up {X}: {X} is not clear")
        clear_set.discard(X)
        s["on"][X] = "held"
        s["holding"] = X
        s["handempty"] = False

    elif op == "put-down":
        _check(len(args) == 1, f"put-down needs 1 arg, got {args}")
        X = args[0]
        _check(s["holding"] == X, f"put-down {X}: not holding {X} (holding {s['holding']})")
        s["on"][X] = "table"
        clear_set.add(X)
        s["holding"] = None
        s["handempty"] = True

    elif op == "unstack":
        _check(len(args) == 2, f"unstack needs 2 args, got {args}")
        X, Y = args[0], args[1]
        _check(s["handempty"], f"unstack {X} {Y}: hand not empty")
        _check(s["on"].get(X) == Y, f"unstack {X} {Y}: {X} is not on {Y} (on {s['on'].get(X)})")
        _check(X in clear_set, f"unstack {X} {Y}: {X} is not clear")
        clear_set.discard(X)
        clear_set.add(Y)
        s["on"][X] = "held"
        s["holding"] = X
        s["handempty"] = False

    elif op == "stack":
        _check(len(args) == 2, f"stack needs 2 args, got {args}")
        X, Y = args[0], args[1]
        _check(s["holding"] == X, f"stack {X} {Y}: not holding {X} (holding {s['holding']})")
        _check(Y in clear_set, f"stack {X} {Y}: {Y} is not clear")
        s["on"][X] = Y
        clear_set.discard(Y)
        clear_set.add(X)
        s["holding"] = None
        s["handempty"] = True

    else:
        raise ValueError(f"Unknown 4-op: {op!r}")

    s["clear"] = sorted(clear_set)
    return s


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def rollout_4op(
    initial: Dict[str, Any],
    actions: List[Tuple[str, List[str]]],
) -> Tuple[List[Dict[str, Any]], Optional[int], Optional[str]]:
    """Execute all 4-op actions sequentially.

    Returns (states, bad_idx, bad_msg).
    states[0] is the initial state; states[i+1] is after actions[i].
    bad_idx is None if every action succeeded.
    """
    states = [initial]
    for i, (op, args) in enumerate(actions):
        try:
            states.append(apply_4op_action(states[-1], op, args))
        except ValueError as e:
            return states, i, str(e)
    return states, None, None


# ---------------------------------------------------------------------------
# Goal checking
# ---------------------------------------------------------------------------

def bw_4op_goal_satisfied(final_state: Dict[str, Any], goal_nl: str) -> bool:
    """Return True iff final_state satisfies all on/clear predicates in goal_nl."""
    on = final_state["on"]

    for m in re.finditer(r"Block\s+(\w+)\s+is\s+on\s+Block\s+(\w+)", goal_nl, re.IGNORECASE):
        blk, dest = m.group(1).upper(), m.group(2).upper()
        if on.get(blk) != dest:
            return False

    for m in re.finditer(r"Block\s+(\w+)\s+is\s+on\s+the\s+table", goal_nl, re.IGNORECASE):
        blk = m.group(1).upper()
        if on.get(blk) != "table":
            return False

    clear_set = set(final_state["clear"])
    for m in re.finditer(r"Block\s+(\w+)\s+is\s+clear", goal_nl, re.IGNORECASE):
        blk = m.group(1).upper()
        if blk not in clear_set:
            return False

    return True
