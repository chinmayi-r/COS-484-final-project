"""
Pure-Python Blocksworld 4-ops plan validator.

Implements the blocksworld-4ops domain used by PlanBench:
  pick-up ?x      — pick block x off the table
  put-down ?x     — put held block x on the table
  unstack ?x ?y   — unstack block x from on top of block y
  stack ?x ?y     — stack held block x onto block y

Returns True iff the plan reaches the goal from the initial state.
"""
import re
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

from benchmark.pddl_parser import BlocksworldInstance


@dataclass
class BWState:
    on_table: Set[str]          # blocks on table
    on: dict                    # on[x] = y means x is directly on y
    clear: Set[str]             # blocks with nothing on top
    holding: Optional[str]      # block in hand, or None

    def arm_empty(self) -> bool:
        return self.holding is None

    def copy(self) -> "BWState":
        return BWState(
            on_table=set(self.on_table),
            on=dict(self.on),
            clear=set(self.clear),
            holding=self.holding,
        )


def _init_state(instance: BlocksworldInstance) -> BWState:
    on = {x: y for x, y in instance.on}
    return BWState(
        on_table=set(instance.on_table),
        on=on,
        clear=set(instance.clear),
        holding=None,
    )


def _apply_action(state: BWState, action: str, args: List[str]) -> Optional[BWState]:
    """Apply one action; return new state or None if preconditions unmet."""
    s = state.copy()

    if action == "pick-up":
        x, = args
        if not (s.arm_empty() and x in s.clear and x in s.on_table):
            return None
        s.holding = x
        s.on_table.discard(x)
        s.clear.discard(x)

    elif action == "put-down":
        x, = args
        if s.holding != x:
            return None
        s.holding = None
        s.on_table.add(x)
        s.clear.add(x)

    elif action == "unstack":
        x, y = args
        if not (s.arm_empty() and x in s.clear and s.on.get(x) == y):
            return None
        s.holding = x
        s.clear.discard(x)
        s.clear.add(y)
        del s.on[x]

    elif action == "stack":
        x, y = args
        if not (s.holding == x and y in s.clear):
            return None
        s.holding = None
        s.on[x] = y
        s.clear.discard(y)
        s.clear.add(x)

    else:
        return None  # unknown action

    return s


def _goal_reached(state: BWState, instance: BlocksworldInstance) -> bool:
    for x, y in instance.goal_on:
        if state.on.get(x) != y:
            return False
    for b in instance.goal_on_table:
        if b not in state.on_table:
            return False
    return True


# ── Plan text parsing ──────────────────────────────────────────────────────────

_PDDL_ACTION_RE = re.compile(
    r"\(\s*(pick-up|put-down|unstack|stack)\s+([\w\d]+)(?:\s+([\w\d]+))?\s*\)",
    re.IGNORECASE,
)

_NL_ACTION_RE = re.compile(
    r"(?:"
    r"(unstack)\s+([\w\d]+)\s+(?:from\s+(?:on\s+top\s+of\s+)?|from\s+)([\w\d]+)"
    r"|(stack)\s+([\w\d]+)\s+(?:on(?:\s+top\s+of)?\s+)([\w\d]+)"
    r"|(pick.?up)\s+([\w\d]+)"
    r"|(put.?down)\s+([\w\d]+)"
    r")",
    re.IGNORECASE,
)


def _parse_action_pddl(line: str) -> Optional[Tuple[str, List[str]]]:
    m = _PDDL_ACTION_RE.search(line)
    if not m:
        return None
    act = m.group(1).lower()
    args = [m.group(2).lower()]
    if m.group(3):
        args.append(m.group(3).lower())
    return act, args


def _parse_action_nl(line: str) -> Optional[Tuple[str, List[str]]]:
    m = _NL_ACTION_RE.search(line)
    if not m:
        return None
    if m.group(1):
        return "unstack", [m.group(2).lower(), m.group(3).lower()]
    if m.group(4):
        return "stack", [m.group(5).lower(), m.group(6).lower()]
    if m.group(7):
        return "pick-up", [m.group(8).lower()]
    if m.group(9):
        return "put-down", [m.group(10).lower()]
    return None


def extract_plan_from_text(text: str) -> List[Tuple[str, List[str]]]:
    """
    Extract ordered PDDL actions from LLM output.

    Preference order:
    1. Lines inside a ```pddl ... ``` block
    2. Bare PDDL s-expressions anywhere in the text
    3. Natural-language action lines (fallback)
    """
    # Try fenced code block first
    code_block = re.search(r"```(?:pddl|plan)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    source = code_block.group(1) if code_block else text

    actions = []
    for line in source.splitlines():
        parsed = _parse_action_pddl(line)
        if parsed:
            actions.append(parsed)

    if actions:
        return actions

    # Fallback: natural language
    for line in text.splitlines():
        parsed = _parse_action_nl(line)
        if parsed:
            actions.append(parsed)

    return actions


def validate(instance: BlocksworldInstance, plan_text: str) -> Tuple[bool, str]:
    """
    Validate an LLM-generated plan against a Blocksworld instance.

    Returns (is_valid, reason).
    """
    actions = extract_plan_from_text(plan_text)

    if not actions:
        return False, "no_actions_found"

    state = _init_state(instance)

    for i, (act, args) in enumerate(actions):
        new_state = _apply_action(state, act, args)
        if new_state is None:
            return False, f"precondition_failed_step_{i+1}:{act}({','.join(args)})"
        state = new_state

    if _goal_reached(state, instance):
        return True, "ok"
    return False, "goal_not_reached"
