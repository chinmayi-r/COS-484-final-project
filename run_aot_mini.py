"""
Run AoT+ with gpt-4o-mini on Blocksworld and Logistics.

Usage:
    python run_aot_mini.py

Requires a .env file (or env vars) with:
    OPENAI_API_KEY=sk-...
    OPENAI_MODEL_NAME=gpt-4o-mini

Outputs per-problem results and a final accuracy summary matching the
paper's table format (accuracy = % of problems where the extracted plan
reaches the goal state without any illegal actions).
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Ensure repo root is on the path
sys.path.insert(0, os.path.dirname(__file__))

import copy

from src.aot_plus.core import AoTPlus
from src.aot_plus.blocksworld_examples import (
    get_blocksworld_json_examples,
    get_blocksworld_json_goal_examples,
)
from src.aot_plus.logistics_examples import (
    get_logistics_json_examples,
    get_logistics_json_goal_examples,
)
from src.domains.blocksworld import BlocksWorldDomain
from src.domains.logistics import LogisticsDomain
from src.sim.logistics_sim import (
    LogisticsMeta,
    nl_state_to_canonical as log_nl_to_canonical,
    parse_action_text as log_parse_action,
    rollout_actions as log_rollout,
    validate_city_map,
)
from src.eval.common_aot_utils import extract_actions  # used as fallback in extract_plan_summary
from src.utils.config import Config


# ---------------------------------------------------------------------------
# 4-op Blocksworld simulator (local copy — no cross-import from paper_replication)
# Operators: pick-up, put-down, unstack, stack
# ---------------------------------------------------------------------------

def _nl_to_4op_state(nl: str) -> Dict[str, Any]:
    """Parse NL BW description into 4-op state dict."""
    blocks = sorted(
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


def _parse_4op_action(text: str) -> Tuple[str, List[str]]:
    """Parse a 4-op action string.

    Accepted forms:
      (pick-up X)           (put-down X)
      (unstack X from Y)    (stack X onto Y)
      (unstack X Y)         (stack X Y)       ← bare PDDL fallback
      pick-up X             unstack X from Y  ← no-paren fallback
    """
    text = text.strip().rstrip(".")

    # Strip optional parens
    inner = text
    if inner.startswith("(") and inner.endswith(")"):
        inner = inner[1:-1].strip()

    tl = inner.lower()

    if tl.startswith("pick-up "):
        return "pick-up", [inner.split(None, 1)[1].strip().upper()]
    if tl.startswith("put-down "):
        return "put-down", [inner.split(None, 1)[1].strip().upper()]

    m = re.match(r"unstack\s+(\w+)\s+(?:from\s+)?(\w+)", inner, re.IGNORECASE)
    if m:
        return "unstack", [m.group(1).upper(), m.group(2).upper()]

    m = re.match(r"stack\s+(\w+)\s+(?:on(?:to)?\s+)?(\w+)", inner, re.IGNORECASE)
    if m:
        return "stack", [m.group(1).upper(), m.group(2).upper()]

    raise ValueError(f"Cannot parse action: {text!r}")


def _apply_4op(state: Dict[str, Any], op: str, args: List[str]) -> Dict[str, Any]:
    """Apply one 4-op BW action. Returns new state or raises ValueError."""
    s = copy.deepcopy(state)
    cl: set = set(s["clear"])

    if op == "pick-up":
        X = args[0]
        if not s["handempty"]:
            raise ValueError(f"pick-up {X}: hand not empty")
        if s["on"].get(X) != "table":
            raise ValueError(f"pick-up {X}: not on table")
        if X not in cl:
            raise ValueError(f"pick-up {X}: not clear")
        cl.discard(X)
        s["on"][X] = "held"
        s["holding"] = X
        s["handempty"] = False

    elif op == "put-down":
        X = args[0]
        if s["holding"] != X:
            raise ValueError(f"put-down {X}: not holding {X}")
        s["on"][X] = "table"
        cl.add(X)
        s["holding"] = None
        s["handempty"] = True

    elif op == "unstack":
        X, Y = args[0], args[1]
        if not s["handempty"]:
            raise ValueError(f"unstack {X} {Y}: hand not empty")
        if s["on"].get(X) != Y:
            raise ValueError(f"unstack {X} {Y}: {X} not on {Y}")
        if X not in cl:
            raise ValueError(f"unstack {X} {Y}: {X} not clear")
        cl.discard(X)
        cl.add(Y)
        s["on"][X] = "held"
        s["holding"] = X
        s["handempty"] = False

    elif op == "stack":
        X, Y = args[0], args[1]
        if s["holding"] != X:
            raise ValueError(f"stack {X} {Y}: not holding {X}")
        if Y not in cl:
            raise ValueError(f"stack {X} {Y}: {Y} not clear")
        s["on"][X] = Y
        cl.discard(Y)
        cl.add(X)
        s["holding"] = None
        s["handempty"] = True

    else:
        raise ValueError(f"Unknown 4-op: {op!r}")

    s["clear"] = sorted(cl)
    return s


def _rollout_4op(
    initial: Dict[str, Any],
    actions: List[Tuple[str, List[str]]],
) -> Tuple[List[Dict[str, Any]], Optional[int], Optional[str]]:
    states = [initial]
    for i, (op, args) in enumerate(actions):
        try:
            states.append(_apply_4op(states[-1], op, args))
        except ValueError as e:
            return states, i, str(e)
    return states, None, None


# ---------------------------------------------------------------------------
# Plan extraction — only from the committed "Plan summary:" section
# ---------------------------------------------------------------------------

_PLAN_SUMMARY_RE = re.compile(
    r"Plan\s+summary\s*:?\s*\n(.*?)(?:\Z)",
    re.IGNORECASE | re.DOTALL,
)
_NUMBERED_ACTION_RE = re.compile(r"^\s*\d+\.\s+(.+)", re.MULTILINE)


def extract_plan_summary(raw_output: str) -> List[str]:
    """Extract actions from the numbered 'Plan summary:' list at the end of the response.

    Falls back to ACTION: lines only if no Plan summary section exists, so
    partial responses still produce something instead of silently failing.
    """
    m = _PLAN_SUMMARY_RE.search(raw_output)
    if m:
        section = m.group(1)
        actions = []
        for line_m in _NUMBERED_ACTION_RE.finditer(section):
            text = line_m.group(1).strip().rstrip(".,;")
            if text:
                actions.append(text)
        if actions:
            return actions
    # fallback: if no Plan summary section, use all ACTION: lines (least-bad option)
    return extract_actions(raw_output)


# ---------------------------------------------------------------------------
# Problem banks
# ---------------------------------------------------------------------------

# 20 Blocksworld problems of increasing difficulty.
# Format: (initial_state_nl, goal_state_nl)
BLOCKSWORLD_PROBLEMS: List[Tuple[str, str]] = [
    # --- 3-block problems (easy) ---
    (
        "Block A is on the table. Block B is on Block A. Block C is on the table. "
        "Block B is clear. Block C is clear.",
        "Block A is on Block B. Block B is on Block C. Block C is on the table.",
    ),
    (
        "Block A is on Block B. Block B is on Block C. Block C is on the table. "
        "Block A is clear.",
        "Block C is on Block B. Block B is on Block A. Block A is on the table.",
    ),
    (
        "Block A is on the table. Block B is on the table. Block C is on the table. "
        "Block A is clear. Block B is clear. Block C is clear.",
        "Block C is on Block B. Block B is on Block A. Block A is on the table.",
    ),
    (
        "Block A is on Block B. Block B is on the table. Block C is on the table. "
        "Block A is clear. Block C is clear.",
        "Block B is on Block A. Block A is on the table. Block C is on the table.",
    ),
    (
        "Block A is on the table. Block B is on Block C. Block C is on the table. "
        "Block A is clear. Block B is clear.",
        "Block A is on Block B. Block B is on the table. Block C is on Block A.",
    ),
    # --- 4-block problems (medium) ---
    (
        "Block A is on the table. Block B is on Block A. Block C is on Block B. Block D is on the table. "
        "Block C is clear. Block D is clear.",
        "Block D is on Block C. Block C is on Block B. Block B is on Block A. Block A is on the table.",
    ),
    (
        "Block A is on Block B. Block B is on Block C. Block C is on Block D. Block D is on the table. "
        "Block A is clear.",
        "Block D is on Block A. Block A is on the table. Block B is on the table. Block C is on the table.",
    ),
    (
        "Block A is on the table. Block B is on the table. Block C is on the table. Block D is on the table. "
        "Block A is clear. Block B is clear. Block C is clear. Block D is clear.",
        "Block A is on Block B. Block B is on Block C. Block C is on Block D. Block D is on the table.",
    ),
    (
        "Block A is on Block B. Block B is on the table. Block C is on Block D. Block D is on the table. "
        "Block A is clear. Block C is clear.",
        "Block C is on Block A. Block A is on Block D. Block D is on Block B. Block B is on the table.",
    ),
    (
        "Block A is on the table. Block B is on Block A. Block C is on the table. Block D is on Block C. "
        "Block B is clear. Block D is clear.",
        "Block B is on Block D. Block D is on Block C. Block C is on Block A. Block A is on the table.",
    ),
    # --- 5-block problems (hard) ---
    (
        "Block A is on the table. Block B is on Block A. Block C is on Block B. "
        "Block D is on the table. Block E is on Block D. "
        "Block C is clear. Block E is clear.",
        "Block E is on Block D. Block D is on Block C. Block C is on Block B. "
        "Block B is on Block A. Block A is on the table.",
    ),
    (
        "Block A is on Block B. Block B is on Block C. Block C is on Block D. "
        "Block D is on Block E. Block E is on the table. Block A is clear.",
        "Block E is on Block A. Block A is on the table. Block B is on the table. "
        "Block C is on the table. Block D is on the table.",
    ),
    (
        "Block A is on the table. Block B is on the table. Block C is on the table. "
        "Block D is on the table. Block E is on the table. "
        "Block A is clear. Block B is clear. Block C is clear. Block D is clear. Block E is clear.",
        "Block A is on Block B. Block B is on Block C. Block C is on Block D. "
        "Block D is on Block E. Block E is on the table.",
    ),
    (
        "Block A is on Block C. Block B is on Block D. Block C is on the table. "
        "Block D is on the table. Block E is on the table. "
        "Block A is clear. Block B is clear. Block E is clear.",
        "Block E is on Block A. Block A is on Block B. Block B is on Block C. "
        "Block C is on Block D. Block D is on the table.",
    ),
    (
        "Block A is on the table. Block B is on Block A. Block C is on Block B. "
        "Block D is on Block C. Block E is on the table. "
        "Block D is clear. Block E is clear.",
        "Block E is on Block D. Block D is on Block A. Block A is on the table. "
        "Block B is on the table. Block C is on the table.",
    ),
    (
        "Block A is on Block B. Block B is on the table. Block C is on Block D. "
        "Block D is on Block E. Block E is on the table. "
        "Block A is clear. Block C is clear.",
        "Block C is on Block A. Block A is on Block E. Block E is on Block B. "
        "Block B is on the table. Block D is on the table.",
    ),
    (
        "Block A is on the table. Block B is on the table. Block C is on Block A. "
        "Block D is on Block B. Block E is on Block C. "
        "Block D is clear. Block E is clear.",
        "Block A is on Block D. Block D is on Block E. Block E is on Block B. "
        "Block B is on Block C. Block C is on the table.",
    ),
    (
        "Block A is on Block B. Block B is on Block C. Block C is on the table. "
        "Block D is on the table. Block E is on Block D. "
        "Block A is clear. Block E is clear.",
        "Block D is on Block C. Block C is on Block B. Block B is on Block A. "
        "Block A is on Block E. Block E is on the table.",
    ),
    (
        "Block A is on the table. Block B is on Block A. Block C is on the table. "
        "Block D is on Block C. Block E is on Block D. "
        "Block B is clear. Block E is clear.",
        "Block A is on Block E. Block E is on Block D. Block D is on Block C. "
        "Block C is on Block B. Block B is on the table.",
    ),
    (
        "Block A is on Block B. Block B is on Block C. Block C is on the table. "
        "Block D is on Block E. Block E is on the table. "
        "Block A is clear. Block D is clear.",
        "Block B is on Block D. Block D is on Block E. Block E is on Block C. "
        "Block C is on Block A. Block A is on the table.",
    ),
]

# 15 Logistics problems in PDDL-style format.
# City map: locations ending in "-0" are airports.
#   C0: L0-0 (airport), L0-1, L0-2
#   C1: L1-0 (airport), L1-1, L1-2
#   C2: L2-0 (airport), L2-1, L2-2
LOGISTICS_CITY_MAP = {
    "L0-0": "C0", "L0-1": "C0", "L0-2": "C0",
    "L1-0": "C1", "L1-1": "C1", "L1-2": "C1",
    "L2-0": "C2", "L2-1": "C2", "L2-2": "C2",
}

LOGISTICS_PROBLEMS: List[Tuple[str, str]] = [
    # --- Single package, intra-city ---
    (
        "Package P0 is at L0-2. Truck T0 is at L0-1. Airplane A0 is at L0-0.",
        "Package P0 is at L0-1.",
    ),
    (
        "Package P0 is at L0-1. Truck T0 is at L0-0. Airplane A0 is at L0-0.",
        "Package P0 is at L0-2.",
    ),
    (
        "Package P0 is at L1-2. Truck T1 is at L1-1. Airplane A0 is at L0-0.",
        "Package P0 is at L1-0.",
    ),
    # --- Single package, cross-city ---
    (
        "Package P0 is at L0-1. Truck T0 is at L0-2. Truck T1 is at L1-2. Airplane A0 is at L0-0.",
        "Package P0 is at L1-1.",
    ),
    (
        "Package P0 is at L0-2. Truck T0 is at L0-1. Truck T1 is at L1-1. Airplane A0 is at L1-0.",
        "Package P0 is at L1-2.",
    ),
    (
        "Package P0 is at L1-1. Truck T0 is at L0-0. Truck T1 is at L1-0. Airplane A0 is at L0-0.",
        "Package P0 is at L0-2.",
    ),
    # --- Two packages, intra-city ---
    (
        "Package P0 is at L0-1. Package P1 is at L0-2. Truck T0 is at L0-0. Airplane A0 is at L0-0.",
        "Package P0 is at L0-2. Package P1 is at L0-1.",
    ),
    (
        "Package P0 is at L0-0. Package P1 is at L0-2. Truck T0 is at L0-1. Airplane A0 is at L0-0.",
        "Package P0 is at L0-2. Package P1 is at L0-0.",
    ),
    # --- Two packages, cross-city ---
    (
        "Package P0 is at L0-1. Package P1 is at L1-1. "
        "Truck T0 is at L0-1. Truck T1 is at L1-1. Airplane A0 is at L0-0.",
        "Package P0 is at L1-1. Package P1 is at L0-1.",
    ),
    (
        "Package P0 is at L0-2. Package P1 is at L1-2. "
        "Truck T0 is at L0-0. Truck T1 is at L1-0. Airplane A0 is at L0-0.",
        "Package P0 is at L1-1. Package P1 is at L0-1.",
    ),
    # --- Three packages ---
    (
        "Package P0 is at L0-1. Package P1 is at L0-2. Package P2 is at L1-1. "
        "Truck T0 is at L0-0. Truck T1 is at L1-0. Airplane A0 is at L0-0.",
        "Package P0 is at L1-1. Package P1 is at L0-1. Package P2 is at L0-2.",
    ),
    (
        "Package P0 is at L1-2. Package P1 is at L0-1. Package P2 is at L1-1. "
        "Truck T0 is at L0-2. Truck T1 is at L1-2. Airplane A0 is at L1-0.",
        "Package P0 is at L0-1. Package P1 is at L1-2. Package P2 is at L0-2.",
    ),
    # --- Three-city problems ---
    (
        "Package P0 is at L0-1. Truck T0 is at L0-2. Truck T1 is at L1-1. "
        "Truck T2 is at L2-1. Airplane A0 is at L0-0. Airplane A1 is at L2-0.",
        "Package P0 is at L2-2.",
    ),
    (
        "Package P0 is at L2-1. Package P1 is at L0-2. "
        "Truck T0 is at L0-1. Truck T1 is at L1-2. Truck T2 is at L2-0. "
        "Airplane A0 is at L1-0. Airplane A1 is at L2-0.",
        "Package P0 is at L0-1. Package P1 is at L2-2.",
    ),
    (
        "Package P0 is at L0-2. Package P1 is at L1-1. Package P2 is at L2-2. "
        "Truck T0 is at L0-1. Truck T1 is at L1-0. Truck T2 is at L2-1. "
        "Airplane A0 is at L0-0. Airplane A1 is at L1-0.",
        "Package P0 is at L2-1. Package P1 is at L0-2. Package P2 is at L1-2.",
    ),
]


# ---------------------------------------------------------------------------
# Goal checking + goal-tracking helpers
# ---------------------------------------------------------------------------

def _bw_goal_satisfied(final_state: Dict[str, Any], goal_nl: str) -> bool:
    """Check if the final 4-op BW state satisfies the NL goal."""
    for m in re.finditer(r"Block\s+(\w+)\s+is\s+on\s+Block\s+(\w+)", goal_nl, re.IGNORECASE):
        block, dest = m.group(1).upper(), m.group(2).upper()
        if final_state["on"].get(block) != dest:
            return False
    for m in re.finditer(r"Block\s+(\w+)\s+is\s+on\s+the\s+table", goal_nl, re.IGNORECASE):
        block = m.group(1).upper()
        if final_state["on"].get(block) != "table":
            return False
    for m in re.finditer(r"Block\s+(\w+)\s+is\s+clear", goal_nl, re.IGNORECASE):
        block = m.group(1).upper()
        if block not in final_state["clear"]:
            return False
    return True


def _bw_goal_predicates(goal_state: Dict[str, Any]) -> List[str]:
    """Return goal predicates derived from a parsed goal state dict."""
    preds = []
    for block, loc in sorted(goal_state["on"].items()):
        if loc == "table":
            preds.append(f"{block} on table")
        elif loc not in ("held", None):
            preds.append(f"{block} on {loc}")
    for block in sorted(goal_state.get("clear", [])):
        preds.append(f"{block} clear")
    return preds


def _bw_check_predicates(
    state: Dict[str, Any], predicates: List[str]
) -> Tuple[List[str], List[str]]:
    """Split predicates into (satisfied, remaining) for a given state."""
    satisfied, remaining = [], []
    for pred in predicates:
        m_on = re.match(r"(\w+) on (\w+)", pred)
        m_cl = re.match(r"(\w+) clear", pred)
        if m_on:
            blk, loc = m_on.group(1), m_on.group(2)
            (satisfied if state["on"].get(blk) == loc else remaining).append(pred)
        elif m_cl:
            blk = m_cl.group(1)
            (satisfied if blk in state.get("clear", []) else remaining).append(pred)
        else:
            remaining.append(pred)
    return satisfied, remaining


def _augment_with_goal_tracking(
    state: Dict[str, Any], predicates: List[str]
) -> Dict[str, Any]:
    """Add goal_satisfied/goal_remaining fields to a state dict (in-place copy)."""
    s = dict(state)
    satisfied, remaining = _bw_check_predicates(s, predicates)
    s["goal_satisfied"] = satisfied
    s["goal_remaining"] = remaining
    return s


# ---------------------------------------------------------------------------
# Error taxonomy (Conditions B and C — JSON state is machine-readable)
# ---------------------------------------------------------------------------

_ACTION_LINE_RE = re.compile(r"^ACTION:\s*(.+)$", re.MULTILINE)
_STATE_JSON_RE = re.compile(r"STATE_JSON:\s*```json\s*(.*?)\s*```", re.DOTALL)
_SEGMENT_RE = re.compile(
    r"((?:ACTION:\s*.+\n?)+?)\s*STATE_JSON:\s*```json\s*(.*?)\s*```",
    re.DOTALL,
)


def _compute_error_taxonomy_bw(
    raw_output: str,
    initial_nl: str,
    goal_nl: str,
) -> Dict[str, Any]:
    """
    Compare the model's final STATE_JSON against 4-op simulator ground truth.

    Approach:
      1. Extract the committed plan from "Plan summary:"
      2. Simulate it to get the ground truth state (stops at first illegal action)
      3. Extract the last STATE_JSON block from the body
      4. Diff model state vs ground truth, classify each wrong predicate:
           omission    — predicate should have changed from initial but model left it unchanged
           hallucination — predicate should have stayed same as initial but model changed it
           wrong_value — both sim and model differ from initial, but to different values
      5. Propagation flag: the first illegal action references a block with a state error

    Returns:
      n_state_blocks    : number of STATE_JSON blocks found in body
      state_parseable   : whether the final STATE_JSON parsed successfully
      first_illegal_step: 0-indexed step of first illegal plan action (None if all legal)
      error_counts      : {"omission": N, "hallucination": N, "wrong_value": N}
      propagation       : True if first illegal action involves a block with a state error
      block_errors      : {block: error_type} for each wrong predicate
    """
    empty = {
        "n_state_blocks": 0, "state_parseable": False,
        "first_illegal_step": None,
        "error_counts": {"omission": 0, "hallucination": 0, "wrong_value": 0},
        "propagation": False, "block_errors": {},
    }

    # Parse initial state
    try:
        initial = _nl_to_4op_state(initial_nl)
    except Exception:
        return empty

    # Simulate the committed plan to get ground truth final state
    plan_raw = extract_plan_summary(raw_output)
    parsed_plan = []
    for a in plan_raw:
        try:
            parsed_plan.append(_parse_4op_action(a))
        except ValueError:
            pass
    sim_states, bad_idx, _ = _rollout_4op(initial, parsed_plan)
    sim_final = sim_states[-1]  # last valid state (may be mid-plan if illegal action occurred)

    # Extract the last STATE_JSON from the body (before Plan summary)
    body = raw_output.split("Plan summary:")[0] if "Plan summary:" in raw_output else raw_output
    state_blocks = _STATE_JSON_RE.findall(body)
    n_state_blocks = len(state_blocks)

    if not state_blocks:
        return {**empty, "first_illegal_step": bad_idx}

    try:
        model_final = json.loads(state_blocks[-1].strip())
    except json.JSONDecodeError:
        return {**empty, "n_state_blocks": n_state_blocks,
                "first_illegal_step": bad_idx}

    # Diff model["on"] vs sim_final["on"] using initial["on"] as baseline
    model_on = model_final.get("on", {})
    sim_on   = sim_final.get("on", {})
    init_on  = initial.get("on", {})

    block_errors: Dict[str, str] = {}
    omission = hallucination = wrong_value = 0

    for block in set(list(sim_on.keys()) + list(model_on.keys())):
        sim_val  = sim_on.get(block)
        mod_val  = model_on.get(block)
        init_val = init_on.get(block)

        if sim_val == mod_val:
            continue  # correct

        if mod_val == init_val:
            # Should have changed from initial, but model left it at initial value
            block_errors[block] = "omission"
            omission += 1
        elif sim_val == init_val:
            # Should have stayed at initial value, but model changed it
            block_errors[block] = "hallucination"
            hallucination += 1
        else:
            # Both sim and model differ from initial, but disagree on the new value
            block_errors[block] = "wrong_value"
            wrong_value += 1

    # Propagation: first illegal action references a block with a state error
    propagation = False
    if bad_idx is not None and bad_idx < len(parsed_plan):
        _, bad_args = parsed_plan[bad_idx]
        if any(a in block_errors for a in bad_args):
            propagation = True

    return {
        "n_state_blocks":     n_state_blocks,
        "state_parseable":    True,
        "first_illegal_step": bad_idx,
        "error_counts": {
            "omission":      omission,
            "hallucination": hallucination,
            "wrong_value":   wrong_value,
        },
        "propagation":  propagation,
        "block_errors": block_errors,
    }



def _log_goal_satisfied(final_state: Dict[str, Any], goal_nl: str) -> bool:
    """Check if the final logistics state satisfies the NL goal (package locations only)."""
    goal_state = log_nl_to_canonical(goal_nl)
    for entity, loc in goal_state["at"].items():
        if loc is None:
            continue
        if final_state["at"].get(entity) != loc:
            return False
    return True


# ---------------------------------------------------------------------------
# Plan verifiers
# ---------------------------------------------------------------------------

def verify_blocksworld(initial_nl: str, goal_nl: str, raw_output: str) -> Tuple[bool, int, str]:
    """
    Returns (goal_achieved, n_actions_extracted, error_message).
    Uses 4-op PDDL verifier (pick-up/put-down/unstack/stack).
    """
    try:
        initial = _nl_to_4op_state(initial_nl)
    except Exception as e:
        return False, 0, f"Failed to parse initial state: {e}"

    raw_actions = extract_plan_summary(raw_output)
    if not raw_actions:
        return False, 0, "No actions extracted"

    parsed = []
    for a in raw_actions:
        try:
            parsed.append(_parse_4op_action(a))
        except ValueError:
            return False, len(parsed), f"Unparseable action: {a!r}"

    states, bad_idx, bad_text = _rollout_4op(initial, parsed)
    if bad_idx is not None:
        return False, len(parsed), f"Illegal action at step {bad_idx}: {bad_text!r}"

    final = states[-1]
    if _bw_goal_satisfied(final, goal_nl):
        return True, len(parsed), ""
    return False, len(parsed), "Goal not reached"


def verify_logistics(
    initial_nl: str,
    goal_nl: str,
    city_map: Dict[str, str],
    raw_output: str,
) -> Tuple[bool, int, str]:
    """
    Returns (goal_achieved, n_actions_extracted, error_message).
    """
    try:
        initial = log_nl_to_canonical(initial_nl)
    except Exception as e:
        return False, 0, f"Failed to parse initial state: {e}"

    meta = validate_city_map(city_map)
    raw_actions = extract_plan_summary(raw_output)
    if not raw_actions:
        return False, 0, "No actions extracted"

    parsed = []
    for a in raw_actions:
        try:
            parsed.append(log_parse_action(a))
        except ValueError:
            return False, len(parsed), f"Unparseable action: {a!r}"

    states, bad_idx, bad_text = log_rollout(initial, parsed, meta)
    if bad_idx is not None:
        return False, len(parsed), f"Illegal action at step {bad_idx}: {bad_text!r}"

    final = states[-1]
    if _log_goal_satisfied(final, goal_nl):
        return True, len(parsed), ""
    return False, len(parsed), "Goal not reached"


# ---------------------------------------------------------------------------
# Solver factory
# ---------------------------------------------------------------------------

def make_aot_plus_solver(
    domain: str,
    provider: str,
    condition: str = "B",
    checkpoint_interval: int = 3,
) -> AoTPlus:
    """Build an AoT+ solver for the given condition (B or C)."""
    goal_tracking = (condition == "C")
    if domain == "blocksworld":
        domain_desc = BlocksWorldDomain.get_domain_description()
        examples = (get_blocksworld_json_goal_examples() if goal_tracking
                    else get_blocksworld_json_examples())
    else:
        domain_desc = LogisticsDomain.get_domain_description()
        examples = (get_logistics_json_goal_examples() if goal_tracking
                    else get_logistics_json_examples())

    solver = AoTPlus(
        domain_desc, provider,
        use_json_state=True,
        state_memoization_interval=checkpoint_interval,
        goal_tracking=goal_tracking,
    )
    for ex in examples:
        solver.add_formatted_example(ex)
    return solver


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def _print_prompt_structure(solver: "AoTPlus", domain: str, init_json: str,
                            goal_json: str, extra: str = "") -> None:
    """Print a human-readable summary of the messages being sent."""
    msgs = solver.prompt_builder.build_messages(init_json, goal_json, extra)
    print(f"\n  {'─'*56}")
    print(f"  PROMPT STRUCTURE ({domain} / JSON extension)")
    print(f"  Messages: {len(msgs)}  "
          f"(system + {len(msgs)-2}×user/asst turns + 1 final user)")
    total_chars = sum(len(m["content"]) for m in msgs)
    print(f"  Total chars: {total_chars:,}")
    for i, m in enumerate(msgs):
        role = m["role"].upper()
        chars = len(m["content"])
        preview = m["content"][:80].replace("\n", " ")
        print(f"    [{i}] {role:<9} {chars:5,} chars  │ {preview}…")
    print(f"  {'─'*56}\n")


def evaluate_domain(
    domain: str,
    problems: List[Tuple[str, str]],
    provider: str,
    model_name: str,
    condition: str = "B",
    checkpoint_interval: int = 3,
    max_tokens: int = 3000,
    delay_secs: float = 1.0,
    output_dir: Optional[str] = None,
    city_map: Optional[Dict[str, str]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    solver = make_aot_plus_solver(domain, provider, condition, checkpoint_interval)
    results = []

    print(f"\n{'='*60}")
    print(f"  Domain  : {domain}")
    print(f"  Model   : {model_name}")
    print(f"  Problems: {len(problems)}")
    print(f"{'='*60}")

    for idx, (init_nl, goal_nl) in enumerate(problems, start=1):
        print(f"  [{idx:2d}/{len(problems)}] ", end="", flush=True)
        error_msg = ""
        raw = ""
        n_actions = 0
        success = False

        try:
            if domain == "blocksworld":
                init_dict = _nl_to_4op_state(init_nl)
                goal_dict = _nl_to_4op_state(goal_nl)
                if condition == "C":
                    preds = _bw_goal_predicates(goal_dict)
                    init_dict = _augment_with_goal_tracking(init_dict, preds)
                init_json = json.dumps(init_dict, sort_keys=True)
                goal_json = json.dumps(goal_dict, sort_keys=True)
            else:
                init_json = json.dumps(log_nl_to_canonical(init_nl), sort_keys=True)
                goal_json = json.dumps(log_nl_to_canonical(goal_nl), sort_keys=True)

            extra = ""
            if domain == "logistics":
                cm = city_map or LOGISTICS_CITY_MAP
                city_map_lines = "\n".join(
                    f"  {loc} → {c}" for loc, c in sorted(cm.items())
                )
                extra = (
                    "IMPORTANT — City map (trucks may ONLY drive within one city; "
                    "cross-city travel requires loading onto an airplane at an airport):\n"
                    + city_map_lines
                    + "\nLocations ending in '-0' are airports. "
                    "Airplanes fly between airports only."
                )

            if verbose and idx == 1:
                _print_prompt_structure(solver, domain, init_json, goal_json, extra)

            raw = solver.solve(init_json, goal_json, temperature=0.0,
                               max_tokens=max_tokens, extra_context=extra)

            if domain == "blocksworld":
                success, n_actions, error_msg = verify_blocksworld(init_nl, goal_nl, raw)
            else:
                success, n_actions, error_msg = verify_logistics(
                    init_nl, goal_nl, city_map or LOGISTICS_CITY_MAP, raw
                )

        except Exception as e:
            error_msg = str(e)
            success = False

        # Error taxonomy — only for JSON-parseable conditions (B and C)
        taxonomy: Dict[str, Any] = {}
        if domain == "blocksworld" and raw:
            taxonomy = _compute_error_taxonomy_bw(raw, init_nl, goal_nl)

        status = "✓" if success else "✗"
        tax_str = ""
        if taxonomy and taxonomy.get("state_parseable"):
            ec = taxonomy.get("error_counts", {})
            prop = "P" if taxonomy.get("propagation") else ""
            tax_str = (f"  [om={ec.get('omission',0)} "
                       f"hal={ec.get('hallucination',0)} "
                       f"wv={ec.get('wrong_value',0)}{prop}]")
        elif taxonomy and taxonomy.get("n_state_blocks", 0) == 0:
            tax_str = "  [no STATE_JSON]"
        print(f"{status}  actions={n_actions:2d}  {error_msg or ''}{tax_str}")

        if verbose and not success:
            print(f"\n  ── RAW OUTPUT (problem {idx}) ──────────────────────────")
            print(raw[:3000])
            if len(raw) > 3000:
                print(f"  … [{len(raw)-3000} chars truncated]")
            print(f"  ── END RAW OUTPUT ──────────────────────────────────\n")

        result_entry: Dict[str, Any] = {
            "problem_idx": idx,
            "initial_state": init_nl,
            "goal_state":    goal_nl,
            "success":       success,
            "n_actions":     n_actions,
            "error":         error_msg,
            "condition":     condition,
        }
        if taxonomy:
            result_entry["error_taxonomy"] = {
                k: v for k, v in taxonomy.items() if k != "block_errors"
            }
        results.append(result_entry)

        if output_dir:
            os.makedirs(f"{output_dir}/{domain}", exist_ok=True)
            with open(f"{output_dir}/{domain}/problem_{idx:03d}.txt", "w") as f:
                f.write(raw)
            with open(f"{output_dir}/{domain}/problem_{idx:03d}_result.json", "w") as f:
                json.dump(results[-1], f, indent=2)

        if idx < len(problems):
            time.sleep(delay_secs)

    n_success = sum(r["success"] for r in results)
    accuracy  = n_success / len(results) * 100 if results else 0.0

    # Aggregate error taxonomy across problems
    taxonomy_agg: Dict[str, Any] = {}
    if domain == "blocksworld":
        all_tax = [r.get("error_taxonomy", {}) for r in results if "error_taxonomy" in r]
        if all_tax:
            parseable = [t for t in all_tax if t.get("state_parseable")]
        taxonomy_agg = {
            "n_problems_with_state_json": sum(1 for t in all_tax if t.get("n_state_blocks", 0) > 0),
            "n_state_parseable":          len(parseable),
            "omission":      sum(t["error_counts"]["omission"]      for t in parseable),
            "hallucination": sum(t["error_counts"]["hallucination"]  for t in parseable),
            "wrong_value":   sum(t["error_counts"]["wrong_value"]    for t in parseable),
            "propagation":   sum(1 for t in parseable if t.get("propagation")),
            "n_with_illegal_action": sum(
                1 for t in all_tax if t.get("first_illegal_step") is not None
            ),
        }

    summary = {
        "domain":             domain,
        "model":              model_name,
        "method":             "AoT+",
        "condition":          condition,
        "checkpoint_interval": checkpoint_interval,
        "n_problems":         len(results),
        "n_success":          n_success,
        "accuracy_pct":       round(accuracy, 1),
        "error_taxonomy_agg": taxonomy_agg,
        "results":            results,
    }

    if output_dir:
        with open(f"{output_dir}/{domain}/summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run AoT+ JSON extension (Conditions B/C)")
    parser.add_argument("--provider", default="openai",
                        choices=["openai", "sandbox", "azure_openai"])
    parser.add_argument("--domain",   default="both",
                        choices=["blocksworld", "logistics", "both"])
    parser.add_argument("--condition", default="B", choices=["B", "C"],
                        help="B=JSON state only; C=JSON state + goal tracking")
    parser.add_argument("--checkpoint_interval", type=int, default=3,
                        help="Output STATE_JSON every N actions (default 3, matches paper)")
    parser.add_argument("--n_bw",  type=int, default=20,
                        help="Number of Blocksworld problems to run")
    parser.add_argument("--n_log", type=int, default=15,
                        help="Number of Logistics problems to run")
    parser.add_argument("--seed",  type=int, default=42,
                        help="Random seed for PlanBench instance selection")
    parser.add_argument("--no_planbench", action="store_true",
                        help="Use hardcoded fallback problems instead of PlanBench")
    parser.add_argument("--cache_dir", default=".planbench_cache",
                        help="Local directory for caching downloaded PlanBench PDDL files")
    parser.add_argument("--max_tokens", type=int, default=3000)
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Seconds between API calls")
    parser.add_argument("--output_dir", default="outputs/aot_plus_mini",
                        help="Directory to save raw outputs and per-problem results")
    parser.add_argument("--verbose", action="store_true",
                        help="Print prompt structure for problem 1 and raw output for failures")
    args = parser.parse_args()

    available = Config.get_available_providers()
    if not available:
        print(
            "\nError: No LLM provider configured.\n"
            "Create a .env file with:\n"
            "    OPENAI_API_KEY=sk-...\n"
            "    OPENAI_MODEL_NAME=gpt-4o-mini\n"
        )
        sys.exit(1)

    if args.provider not in available:
        print(f"Provider '{args.provider}' not configured. Available: {available}")
        sys.exit(1)

    model_map = {
        "openai":       Config.OPENAI_MODEL_NAME or "gpt-4o-mini",
        "sandbox":      Config.SANDBOX_MODEL_NAME or "gpt-4o-mini",
        "azure_openai": Config.AZURE_OPENAI_DEPLOYMENT_NAME or "unknown",
        "anthropic":    Config.ANTHROPIC_MODEL_NAME or "claude-sonnet-4-6",
    }
    model_name = model_map.get(args.provider, args.provider)

    # ------------------------------------------------------------------
    # Load problem banks
    # ------------------------------------------------------------------
    use_planbench = not args.no_planbench
    city_map = LOGISTICS_CITY_MAP  # default; overwritten by PlanBench parser

    if use_planbench:
        try:
            from src.benchmarks.planbench import (
                get_blocksworld_problems,
                get_logistics_problems,
            )
            print(f"Downloading PlanBench instances (seed={args.seed}, "
                  f"cache={args.cache_dir!r}) …")
            bw_dicts  = get_blocksworld_problems(args.n_bw,  args.seed, cache_dir=args.cache_dir)
            log_dicts, city_map = get_logistics_problems(args.n_log, args.seed, cache_dir=args.cache_dir)
            bw_probs  = [(d["initial_state"], d["goal_state"]) for d in bw_dicts]
            log_probs = [(d["initial_state"], d["goal_state"]) for d in log_dicts]
            print(f"  Blocksworld : {len(bw_probs)} problems "
                  f"(ids {[d['instance_id'] for d in bw_dicts]})")
            print(f"  Logistics   : {len(log_probs)} problems "
                  f"(ids {[d['instance_id'] for d in log_dicts]})\n")
        except Exception as e:
            print(f"Warning: PlanBench download failed ({e}). Falling back to hardcoded bank.")
            use_planbench = False

    if not use_planbench:
        bw_probs  = BLOCKSWORLD_PROBLEMS[:args.n_bw]
        log_probs = LOGISTICS_PROBLEMS[:args.n_log]

    summaries = {}

    if args.domain in ("blocksworld", "both"):
        summaries["blocksworld"] = evaluate_domain(
            domain="blocksworld",
            problems=bw_probs,
            provider=args.provider,
            model_name=model_name,
            condition=args.condition,
            checkpoint_interval=args.checkpoint_interval,
            max_tokens=args.max_tokens,
            delay_secs=args.delay,
            output_dir=args.output_dir,
            verbose=args.verbose,
        )

    if args.domain in ("logistics", "both"):
        summaries["logistics"] = evaluate_domain(
            domain="logistics",
            problems=log_probs,
            provider=args.provider,
            model_name=model_name,
            condition=args.condition,
            checkpoint_interval=args.checkpoint_interval,
            max_tokens=args.max_tokens,
            delay_secs=args.delay,
            output_dir=args.output_dir,
            city_map=city_map,
            verbose=args.verbose,
        )

    # Print results table
    source = "PlanBench" if use_planbench else "fallback bank"
    cond_label = f"Condition {args.condition}, k={args.checkpoint_interval}"
    print(f"\n{'='*60}")
    print(f"  AoT+ JSON Extension — {model_name}  [{source}]")
    print(f"  {cond_label}")
    print(f"  {'Domain':<15} {'Solved':>8} {'Accuracy':>10}")
    print(f"  {'-'*35}")
    for domain, s in summaries.items():
        print(f"  {domain:<15} {s['n_success']:>4}/{s['n_problems']:<4} {s['accuracy_pct']:>8.1f}%")
        if s.get("error_taxonomy_agg"):
            ta = s["error_taxonomy_agg"]
            print(f"    taxonomy: omission={ta['omission']} "
                  f"hallucination={ta['hallucination']} "
                  f"wrong_value={ta['wrong_value']} "
                  f"propagation={ta['propagation']} "
                  f"(of {ta['n_state_parseable']} parseable STATE_JSON blocks)")
    print(f"{'='*60}\n")

    if args.output_dir:
        with open(f"{args.output_dir}/final_summary.json", "w") as f:
            json.dump({k: {kk: vv for kk, vv in v.items() if kk != "results"}
                       for k, v in summaries.items()}, f, indent=2)
        print(f"Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
