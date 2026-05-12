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
    apply_action as log_apply_action,
    clone_state as log_clone_state,
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
    r"((?:ACTION:[^\n]+\n?)+)\s*STATE_JSON:\s*```json\s*(.*?)\s*```",
    re.DOTALL,
)


def _compute_error_taxonomy_bw(
    raw_output: str,
    initial_nl: str,
    goal_nl: str,
) -> Dict[str, Any]:
    """
    Per-step error taxonomy comparing model STATE_JSON vs 4-op simulator at each checkpoint.

    For each (action, STATE_JSON) pair at step t, classifies errors per block using t-1 as
    the reference — not the initial state. This distinguishes four error modes:

      omission     — model was correct at t-1; sim changed block X but model's JSON left
                     X unchanged (model forgot to apply the action's effect)
      hallucination — model was correct at t-1; sim left X unchanged but model's JSON
                     changed X (model invented an effect that didn't happen)
      propagation  — model was correct at t-1; both sim and model changed X but to
                     different values (model applied the action with wrong arguments/order)
      accumulation — model was already wrong at t-1 and still wrong at t with same belief
                     (classic AoT failure: wrong belief carried forward every step)

    Counts accumulate across all checkpoints. first_illegal_step comes from the committed
    plan rollout (Plan summary) independently of the per-step walk.
    """
    empty = {
        "n_state_blocks": 0, "state_parseable": False,
        "n_steps_checked": 0, "first_illegal_step": None,
        "error_counts": {"omission": 0, "hallucination": 0,
                         "propagation": 0, "accumulation": 0},
        "block_errors": {},
    }

    try:
        initial = _nl_to_4op_state(initial_nl)
    except Exception:
        return empty

    body = raw_output.split("Plan summary:")[0] if "Plan summary:" in raw_output else raw_output
    segments = _SEGMENT_RE.findall(body)
    n_state_blocks = len(segments)

    if not segments:
        all_blocks = _STATE_JSON_RE.findall(body)
        return {**empty, "n_state_blocks": len(all_blocks)}

    # Simulate committed plan for first_illegal_step
    plan_raw = extract_plan_summary(raw_output)
    parsed_plan: List[Tuple[str, List[str]]] = []
    for a in plan_raw:
        try:
            parsed_plan.append(_parse_4op_action(a))
        except ValueError:
            pass
    _, committed_bad_idx, _ = _rollout_4op(initial, parsed_plan)

    sim_prev = copy.deepcopy(initial)
    model_prev_on: Dict[str, str] = dict(initial["on"])  # model starts from the known initial

    total_errors: Dict[str, int] = {"omission": 0, "hallucination": 0,
                                     "propagation": 0, "accumulation": 0}
    all_block_errors: Dict[str, str] = {}
    n_steps_checked = 0
    state_parseable_count = 0

    for actions_block, state_json_str in segments:
        action_texts = _ACTION_LINE_RE.findall(actions_block)

        # Advance simulator through all actions in this segment
        sim_curr = copy.deepcopy(sim_prev)
        for action_text in action_texts:
            try:
                op, args = _parse_4op_action(action_text)
                sim_curr = _apply_4op(sim_curr, op, args)
            except ValueError:
                break

        # Parse model's STATE_JSON for this checkpoint
        try:
            model_state = json.loads(state_json_str.strip())
            state_parseable_count += 1
        except json.JSONDecodeError:
            sim_prev = sim_curr
            n_steps_checked += 1
            continue

        model_curr_on = model_state.get("on", {})
        sim_curr_on   = sim_curr.get("on", {})
        sim_prev_on   = sim_prev.get("on", {})

        for block in set(list(sim_curr_on) + list(model_curr_on)):
            sim_curr_val   = sim_curr_on.get(block)
            sim_prev_val   = sim_prev_on.get(block)
            model_curr_val = model_curr_on.get(block)
            model_prev_val = model_prev_on.get(block)

            if model_curr_val == sim_curr_val:
                continue  # correct at this step

            error_at_prev = (model_prev_val != sim_prev_val)

            if error_at_prev:
                # Wrong belief from t-1 carried forward unchanged
                total_errors["accumulation"] += 1
                all_block_errors[block] = "accumulation"
            else:
                # New error introduced this step — classify why
                sim_changed   = (sim_curr_val   != sim_prev_val)
                model_changed = (model_curr_val  != model_prev_val)

                if sim_changed and not model_changed:
                    total_errors["omission"] += 1
                    all_block_errors[block] = "omission"
                elif not sim_changed and model_changed:
                    total_errors["hallucination"] += 1
                    all_block_errors[block] = "hallucination"
                else:
                    # Both changed but to different values
                    total_errors["propagation"] += 1
                    all_block_errors[block] = "propagation"

        sim_prev = sim_curr
        model_prev_on = dict(model_curr_on)
        n_steps_checked += 1

    return {
        "n_state_blocks":     n_state_blocks,
        "state_parseable":    state_parseable_count > 0,
        "n_steps_checked":    n_steps_checked,
        "first_illegal_step": committed_bad_idx,
        "error_counts":       total_errors,
        "block_errors":       all_block_errors,
    }



def _compute_error_taxonomy_log(
    raw_output: str,
    initial_nl: str,
    goal_nl: str,
    city_map: Dict[str, str],
) -> Dict[str, Any]:
    """
    Per-step error taxonomy for logistics, comparing model STATE_JSON vs simulator.

    Tracks both `at[entity]=location` and `in[package]=vehicle` predicates.
    Uses the same 4-type classification as blocksworld (t-1 baseline):

      omission     — sim changed entity X but model's JSON left it at t-1 value
      hallucination — sim left X unchanged but model's JSON changed X from t-1
      propagation  — both changed X but to different values (action applied wrong)
      accumulation — wrong belief from t-1 carried forward unchanged

    Counts accumulate across all checkpoints.
    """
    empty = {
        "n_state_blocks": 0, "state_parseable": False,
        "n_steps_checked": 0, "first_illegal_step": None,
        "error_counts": {"omission": 0, "hallucination": 0,
                         "propagation": 0, "accumulation": 0},
    }

    try:
        initial = log_nl_to_canonical(initial_nl)
        meta = validate_city_map(city_map)
    except Exception:
        return empty

    body = raw_output.split("Plan summary:")[0] if "Plan summary:" in raw_output else raw_output
    segments = _SEGMENT_RE.findall(body)
    n_state_blocks = len(segments)

    if not segments:
        all_blocks = _STATE_JSON_RE.findall(body)
        return {**empty, "n_state_blocks": len(all_blocks)}

    # Simulate committed plan for first_illegal_step
    plan_raw = extract_plan_summary(raw_output)
    parsed_plan = []
    for a in plan_raw:
        try:
            parsed_plan.append(log_parse_action(a))
        except ValueError:
            pass
    _, committed_bad_idx, _ = log_rollout(initial, parsed_plan, meta)

    sim_prev = log_clone_state(initial)
    # Model starts from the known initial (at+in dicts)
    model_prev_at: Dict[str, Any] = dict(initial["at"])
    model_prev_in: Dict[str, Any] = dict(initial["in"])

    total_errors: Dict[str, int] = {"omission": 0, "hallucination": 0,
                                     "propagation": 0, "accumulation": 0}
    n_steps_checked = 0
    state_parseable_count = 0

    for actions_block, state_json_str in segments:
        action_texts = _ACTION_LINE_RE.findall(actions_block)

        # Advance simulator through all actions in this segment
        sim_curr = log_clone_state(sim_prev)
        for action_text in action_texts:
            try:
                act = log_parse_action(action_text)
                nxt = log_clone_state(sim_curr)
                log_apply_action(nxt, act, meta)
                sim_curr = nxt
            except (ValueError, KeyError):
                break

        # Parse model's STATE_JSON
        try:
            model_state = json.loads(state_json_str.strip())
            state_parseable_count += 1
        except json.JSONDecodeError:
            sim_prev = sim_curr
            n_steps_checked += 1
            continue

        model_curr_at = {k.lower(): (v.lower() if v else None)
                         for k, v in model_state.get("at", {}).items()}
        model_curr_in = {k.lower(): (v.lower() if v else None)
                         for k, v in model_state.get("in", {}).items()}
        sim_curr_at   = sim_curr["at"]
        sim_curr_in   = sim_curr["in"]
        sim_prev_at   = sim_prev["at"]
        sim_prev_in   = sim_prev["in"]

        def _classify(sim_curr_val, sim_prev_val, model_curr_val, model_prev_val):
            if model_curr_val == sim_curr_val:
                return  # correct
            if model_prev_val != sim_prev_val:
                total_errors["accumulation"] += 1
            else:
                sim_changed   = (sim_curr_val   != sim_prev_val)
                model_changed = (model_curr_val  != model_prev_val)
                if sim_changed and not model_changed:
                    total_errors["omission"] += 1
                elif not sim_changed and model_changed:
                    total_errors["hallucination"] += 1
                else:
                    total_errors["propagation"] += 1

        for entity in set(list(sim_curr_at) + list(model_curr_at)):
            _classify(sim_curr_at.get(entity), sim_prev_at.get(entity),
                      model_curr_at.get(entity), model_prev_at.get(entity))

        for pkg in set(list(sim_curr_in) + list(model_curr_in)):
            _classify(sim_curr_in.get(pkg), sim_prev_in.get(pkg),
                      model_curr_in.get(pkg), model_prev_in.get(pkg))

        sim_prev = sim_curr
        model_prev_at = dict(model_curr_at)
        model_prev_in = dict(model_curr_in)
        n_steps_checked += 1

    return {
        "n_state_blocks":     n_state_blocks,
        "state_parseable":    state_parseable_count > 0,
        "n_steps_checked":    n_steps_checked,
        "first_illegal_step": committed_bad_idx,
        "error_counts":       total_errors,
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


def _log_goal_predicates(goal_nl: str) -> List[str]:
    """Return goal predicates as strings like ['P0 at L1-1', 'P1 at L0-2']."""
    goal_state = log_nl_to_canonical(goal_nl)
    preds = []
    for entity, loc in sorted(goal_state["at"].items()):
        if loc is not None:
            preds.append(f"{entity.upper()} at {loc.upper()}")
    return preds


def _log_check_predicates(
    state: Dict[str, Any], predicates: List[str]
) -> Tuple[List[str], List[str]]:
    """Split logistics goal predicates into (satisfied, remaining) for a given state."""
    satisfied, remaining = [], []
    for pred in predicates:
        m = re.match(r"(\S+)\s+at\s+(\S+)", pred)
        if m:
            entity, loc = m.group(1).lower(), m.group(2).lower()
            (satisfied if state["at"].get(entity) == loc else remaining).append(pred)
        else:
            remaining.append(pred)
    return satisfied, remaining


def _log_augment_with_goal_tracking(
    state: Dict[str, Any], predicates: List[str]
) -> Dict[str, Any]:
    """Add goal_satisfied/goal_remaining fields to a logistics state dict."""
    s = dict(state)
    satisfied, remaining = _log_check_predicates(s, predicates)
    s["goal_satisfied"] = satisfied
    s["goal_remaining"] = remaining
    return s


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
                init_dict = log_nl_to_canonical(init_nl)
                if condition == "C":
                    preds = _log_goal_predicates(goal_nl)
                    init_dict = _log_augment_with_goal_tracking(init_dict, preds)
                init_json = json.dumps(init_dict, sort_keys=True)
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
        if raw:
            if domain == "blocksworld":
                taxonomy = _compute_error_taxonomy_bw(raw, init_nl, goal_nl)
            else:
                taxonomy = _compute_error_taxonomy_log(
                    raw, init_nl, goal_nl, city_map or LOGISTICS_CITY_MAP
                )

        status = "✓" if success else "✗"
        tax_str = ""
        if taxonomy and taxonomy.get("state_parseable"):
            ec = taxonomy.get("error_counts", {})
            tax_str = (f"  [om={ec.get('omission',0)} "
                       f"hal={ec.get('hallucination',0)} "
                       f"prop={ec.get('propagation',0)} "
                       f"acc={ec.get('accumulation',0)}]")
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
    all_tax = [r.get("error_taxonomy", {}) for r in results if "error_taxonomy" in r]
    if all_tax:
        parseable = [t for t in all_tax if t.get("state_parseable")]
        total_steps = sum(t.get("n_steps_checked", 0) for t in parseable)
        om  = sum(t["error_counts"]["omission"]      for t in parseable)
        hal = sum(t["error_counts"]["hallucination"]  for t in parseable)
        pro = sum(t["error_counts"]["propagation"]    for t in parseable)
        acc = sum(t["error_counts"]["accumulation"]   for t in parseable)
        taxonomy_agg = {
            "n_problems_with_state_json": sum(1 for t in all_tax if t.get("n_state_blocks", 0) > 0),
            "n_state_parseable":          len(parseable),
            "n_steps_checked":            total_steps,
            "omission":      om,
            "hallucination": hal,
            "propagation":   pro,
            "accumulation":  acc,
            "omission_per_step":      round(om  / total_steps, 4) if total_steps else None,
            "hallucination_per_step": round(hal / total_steps, 4) if total_steps else None,
            "propagation_per_step":   round(pro / total_steps, 4) if total_steps else None,
            "accumulation_per_step":  round(acc / total_steps, 4) if total_steps else None,
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
                        choices=["openai", "sandbox", "azure_openai", "anthropic"])
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
                        help="Base directory; results go to <output_dir>/condition_<B|C>/")
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

    # Condition-scoped output directory so B and C never overwrite each other
    cond_dir = f"{args.output_dir}/condition_{args.condition}" if args.output_dir else None

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
            output_dir=cond_dir,
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
            output_dir=cond_dir,
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
            ns = ta['n_steps_checked']
            print(f"    taxonomy ({ns} steps, {ta['n_state_parseable']} problems parseable):")
            print(f"      raw:  om={ta['omission']} hal={ta['hallucination']} "
                  f"prop={ta['propagation']} acc={ta['accumulation']}")
            print(f"      /step: om={ta.get('omission_per_step','?'):.4f} "
                  f"hal={ta.get('hallucination_per_step','?'):.4f} "
                  f"prop={ta.get('propagation_per_step','?'):.4f} "
                  f"acc={ta.get('accumulation_per_step','?'):.4f}")
    print(f"{'='*60}\n")

    if cond_dir:
        os.makedirs(cond_dir, exist_ok=True)
        # final_summary.json — top-level numbers without per-problem detail
        with open(f"{cond_dir}/final_summary.json", "w") as f:
            json.dump({k: {kk: vv for kk, vv in v.items() if kk != "results"}
                       for k, v in summaries.items()}, f, indent=2)
        # taxonomy_report.json — all per-problem taxonomy entries in one place
        taxonomy_report = {}
        for domain, s in summaries.items():
            taxonomy_report[domain] = [
                {
                    "problem_idx": r["problem_idx"],
                    "success":     r["success"],
                    "n_actions":   r["n_actions"],
                    "error":       r.get("error", ""),
                    **r.get("error_taxonomy", {}),
                }
                for r in s.get("results", [])
                if "error_taxonomy" in r
            ]
        with open(f"{cond_dir}/taxonomy_report.json", "w") as f:
            json.dump(taxonomy_report, f, indent=2)
        print(f"Results saved to {cond_dir}/")


if __name__ == "__main__":
    main()
