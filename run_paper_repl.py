"""
Paper replication runner — AoT+ (Appendix B.3 Blocksworld, Appendix B.4 Logistics).

Uses the EXACT system prompts and verbatim few-shot examples from the paper.
Plan is extracted exclusively from the 'The PDDL version:' block.

Usage:
    python run_paper_repl.py --provider sandbox --domain blocksworld
    python run_paper_repl.py --provider sandbox --domain logistics
    python run_paper_repl.py --provider sandbox --domain both
"""
from __future__ import annotations

import copy
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(__file__))

from src.paper_replication.bw_prompt import build_bw_messages
from src.paper_replication.log_prompt import build_log_messages
from src.paper_replication.bw_4op_sim import (
    nl_to_4op_state,
    apply_4op_action,
    parse_pddl_bw_action,
    extract_pddl_plan,
    rollout_4op,
    bw_4op_goal_satisfied,
)
from src.paper_replication.state_formatter import (
    bw_nl_to_compact_initial,
    bw_nl_to_compact_goal,
    log_nl_to_compact,
    log_goal_nl_to_compact,
    derive_city_map,
)
from src.sim.logistics_sim import (
    apply_action as log_apply_action,
    clone_state as log_clone_state,
    nl_state_to_canonical as log_nl_to_canonical,
    parse_action_text as log_parse_action_text,
    rollout_actions as log_rollout,
    validate_city_map as log_validate_city_map,
)
from src.models.llm_client import get_llm_client
from src.utils.config import Config


# ---------------------------------------------------------------------------
# Problem bank (same 20 problems used in run_aot_mini.py for comparability)
# ---------------------------------------------------------------------------

BLOCKSWORLD_PROBLEMS: List[Tuple[str, str]] = [
    # 3-block (easy)
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
    # 4-block (medium)
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
    # 5-block (hard)
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


# ---------------------------------------------------------------------------
# Error taxonomy (paper replication — NL compact state format)
# ---------------------------------------------------------------------------

# BW: capture ALL consecutive numbered action lines before a '- <compact state>' line.
# One or more numbered lines (each ending with \n) are captured, so k>1 checkpoints
# correctly include all actions in the group when advancing the simulator.
_NL_SEGMENT_RE = re.compile(
    r"((?:\d+\.[^\n]+\n)+)[ \t]*-[ \t]*([^\n]+)",
    re.MULTILINE,
)

# Logistics: match a block of numbered actions followed by an Airplanes/Trucks/Packages state.
# \n\n? allows an optional blank line between each pair of state header lines.
_LOG_PHASE_RE = re.compile(
    r"((?:\d+\.[^\n]+\n?)+)\n*(Airplanes:[^\n]*\n\n?Trucks:[^\n]*\n\n?Packages:[^\n]*)",
    re.MULTILINE | re.IGNORECASE,
)


def _parse_compact_log_state(state_text: str) -> Dict[str, str]:
    """Parse 'Airplanes: A0 at L1-0\nTrucks: T0 at L0-0\nPackages: P1 at L2-1' → at dict.

    Returns lowercase entity→location mapping. Skips qualifiers like '(driven but not change)'.
    Ignores 'All packages are at their goal locations.' line.
    """
    text = re.sub(r'\([^)]*\)', '', state_text)
    at: Dict[str, str] = {}
    for m in re.finditer(r'([A-Za-z][A-Za-z0-9]*)\s+at\s+(L[0-9]+-[0-9]+)', text, re.IGNORECASE):
        at[m.group(1).lower()] = m.group(2).lower()
    return at



def _parse_compact_bw_state(text: str) -> Dict[str, str]:
    """Parse compact BW state like 'D on table, A on C on B on table, E (held)' → on dict."""
    on: Dict[str, str] = {}
    for clause in text.replace(".", "").split(","):
        clause = clause.strip()
        # Held: "E (held)" or "E held"
        m = re.match(r"^(\w+)\s+\(?held\)?$", clause, re.IGNORECASE)
        if m:
            on[m.group(1).upper()] = "held"
            continue
        # Chain: "A on C on B on table"
        parts = [p.strip() for p in re.split(r"\s+on\s+", clause, flags=re.IGNORECASE) if p.strip()]
        for i, blk in enumerate(parts[:-1]):
            dest = parts[i + 1]
            on[blk.upper()] = "table" if dest.lower() == "table" else dest.upper()
        if parts and parts[-1].lower() != "table":
            on[parts[-1].upper()] = "table"
    return on


def _parse_nl_bw_action(text: str) -> Tuple[str, List[str]]:
    """Parse paper-repl NL action: 'Unstack D from A', 'Stack D on B', 'Pick-up D', etc."""
    text = text.strip().rstrip(".")
    m = re.match(r"pick-?up\s+(\w+)", text, re.IGNORECASE)
    if m:
        return "pick-up", [m.group(1).upper()]
    m = re.match(r"put-?down\s+(\w+)", text, re.IGNORECASE)
    if m:
        return "put-down", [m.group(1).upper()]
    m = re.match(r"unstack\s+(\w+)\s+from\s+(\w+)", text, re.IGNORECASE)
    if m:
        return "unstack", [m.group(1).upper(), m.group(2).upper()]
    m = re.match(r"stack\s+(\w+)\s+on(?:\s+top\s+of)?\s+(\w+)", text, re.IGNORECASE)
    if m:
        return "stack", [m.group(1).upper(), m.group(2).upper()]
    raise ValueError(f"Cannot parse NL action: {text!r}")


def _compute_error_taxonomy_bw_nl(
    raw_output: str,
    initial_nl: str,
) -> Dict[str, Any]:
    """
    Same 4-type taxonomy as B/C but for paper-repl NL output.

    Parses numbered-action + compact-state-claim pairs from the body (before PDDL block).
    Simulates each numbered action with the 4-op simulator; compares model's compact
    state claim to simulator at each checkpoint using t-1 as baseline.

      omission     — sim changed X, model left X unchanged (forgot effect)
      hallucination — sim left X unchanged, model changed X (invented effect)
      propagation  — both changed X but to different values (applied action wrong)
      accumulation — model was wrong at t-1, same wrong belief at t
    """
    empty = {
        "n_state_blocks": 0, "state_parseable": False,
        "n_steps_checked": 0, "first_illegal_step": None,
        "error_counts": {"omission": 0, "hallucination": 0,
                         "propagation": 0, "accumulation": 0},
        "block_errors": {},
    }

    try:
        initial = nl_to_4op_state(initial_nl)
    except Exception:
        return empty

    # Body = everything before "The PDDL version:"
    pddl_marker = re.search(r"The PDDL version\s*:", raw_output, re.IGNORECASE)
    body = raw_output[:pddl_marker.start()] if pddl_marker else raw_output

    segments = _NL_SEGMENT_RE.findall(body)
    n_state_blocks = len(segments)
    if not segments:
        return empty

    # first_illegal_step from committed PDDL plan
    pddl_lines = extract_pddl_plan(raw_output)
    parsed_plan: List[Tuple[str, List[str]]] = []
    for line in pddl_lines:
        try:
            parsed_plan.append(parse_pddl_bw_action(line))
        except ValueError:
            pass
    _, committed_bad_idx, _ = rollout_4op(initial, parsed_plan)

    sim_prev = copy.deepcopy(initial)
    model_prev_on: Dict[str, str] = dict(initial["on"])

    total_errors: Dict[str, int] = {"omission": 0, "hallucination": 0,
                                     "propagation": 0, "accumulation": 0}
    all_block_errors: Dict[str, str] = {}
    n_steps_checked = 0
    state_parseable_count = 0

    for action_text, state_text in segments:
        # Extract all individual numbered action lines in this group
        action_lines = re.findall(r"^\d+\.\s*(.+)$", action_text, re.MULTILINE)

        sim_curr = copy.deepcopy(sim_prev)
        for action_line in action_lines:
            try:
                op, args = _parse_nl_bw_action(action_line.strip())
                sim_curr = apply_4op_action(sim_curr, op, args)
            except ValueError:
                pass  # keep sim_curr as-is if action unparseable

        model_on = _parse_compact_bw_state(state_text)
        if not model_on:
            sim_prev = sim_curr
            n_steps_checked += 1
            continue
        state_parseable_count += 1

        sim_curr_on = sim_curr.get("on", {})
        sim_prev_on = sim_prev.get("on", {})

        for block in set(list(sim_curr_on) + list(model_on)):
            sim_curr_val   = sim_curr_on.get(block)
            sim_prev_val   = sim_prev_on.get(block)
            model_curr_val = model_on.get(block)
            model_prev_val = model_prev_on.get(block)

            if model_curr_val == sim_curr_val:
                continue

            if model_prev_val != sim_prev_val:
                total_errors["accumulation"] += 1
                all_block_errors[block] = "accumulation"
            else:
                sim_changed   = (sim_curr_val  != sim_prev_val)
                model_changed = (model_curr_val != model_prev_val)
                if sim_changed and not model_changed:
                    total_errors["omission"] += 1
                    all_block_errors[block] = "omission"
                elif not sim_changed and model_changed:
                    total_errors["hallucination"] += 1
                    all_block_errors[block] = "hallucination"
                else:
                    total_errors["propagation"] += 1
                    all_block_errors[block] = "propagation"

        sim_prev = sim_curr
        model_prev_on = dict(model_on)
        n_steps_checked += 1

    return {
        "n_state_blocks":     n_state_blocks,
        "state_parseable":    state_parseable_count > 0,
        "n_steps_checked":    n_steps_checked,
        "first_illegal_step": committed_bad_idx,
        "error_counts":       total_errors,
        "block_errors":       all_block_errors,
    }


def _compute_error_taxonomy_log_nl(
    raw_output: str,
    initial_nl: str,
    city_map: Dict[str, str],
) -> Dict[str, Any]:
    """
    Phase-boundary error taxonomy for paper-replication logistics output.

    The paper's logistics format groups several numbered actions between each
    compact state block (Airplanes/Trucks/Packages).  We use the PDDL block
    (authoritative action sequence) to advance the simulator to each phase
    boundary and compare the model's compact state report.

    Applies the same 4-type taxonomy (omission/hallucination/propagation/
    accumulation) using t-1 as the reference state.  Only entities explicitly
    mentioned in the model's compact state are compared; packages already
    delivered (and omitted from the state block) are skipped.

    Vehicle positions (all airplanes and trucks) are always listed in the
    paper format, giving reliable taxonomy data for those entities.
    """
    empty = {
        "n_state_blocks": 0, "state_parseable": False,
        "n_steps_checked": 0, "first_illegal_step": None,
        "error_counts": {"omission": 0, "hallucination": 0,
                         "propagation": 0, "accumulation": 0},
    }

    try:
        initial = log_nl_to_canonical(initial_nl)
        meta = log_validate_city_map(city_map)
    except Exception:
        return empty

    # Parse all PDDL actions — these are the authoritative action sequence
    pddl_lines = extract_pddl_plan(raw_output)
    pddl_actions = []
    for line in pddl_lines:
        plain = _strip_pddl_parens(line)
        try:
            pddl_actions.append(log_parse_action_text(plain))
        except ValueError:
            pass
    _, committed_bad_idx, _ = log_rollout(initial, pddl_actions, meta)

    pddl_marker = re.search(r"The PDDL version\s*:", raw_output, re.IGNORECASE)
    body = raw_output[:pddl_marker.start()] if pddl_marker else raw_output

    segments = _LOG_PHASE_RE.findall(body)
    n_state_blocks = len(segments)
    if not segments or not pddl_actions:
        return {**empty, "n_state_blocks": n_state_blocks,
                "first_illegal_step": committed_bad_idx}

    # Pre-compute sim state after each PDDL action (for phase-boundary lookups)
    sim_states = [log_clone_state(initial)]
    cur = log_clone_state(initial)
    for act in pddl_actions:
        nxt = log_clone_state(cur)
        log_apply_action(nxt, act, meta)
        sim_states.append(log_clone_state(nxt))
        cur = nxt

    total_errors: Dict[str, int] = {"omission": 0, "hallucination": 0,
                                     "propagation": 0, "accumulation": 0}
    n_steps_checked = 0
    state_parseable_count = 0
    pddl_done = 0  # cumulative PDDL actions applied so far

    # model_prev_at: running model belief for at[] (starts from initial)
    model_prev_at: Dict[str, str] = {k: v for k, v in initial["at"].items() if v is not None}

    for actions_block, state_block in segments:
        # Count numbered actions in this phase to advance the PDDL cursor
        nums = [int(m) for m in re.findall(r"^(\d+)\.", actions_block, re.MULTILINE)]
        if not nums:
            continue
        phase_end = max(nums)
        # Clip to available PDDL actions
        phase_end = min(phase_end, len(pddl_actions))

        sim_prev_at = {k: v for k, v in sim_states[pddl_done]["at"].items() if v is not None}
        sim_curr_at = {k: v for k, v in sim_states[phase_end]["at"].items() if v is not None}

        model_curr_at = _parse_compact_log_state(state_block)
        if not model_curr_at:
            pddl_done = phase_end
            n_steps_checked += 1
            continue
        state_parseable_count += 1

        for entity in set(list(sim_curr_at) + list(model_curr_at)):
            # Only evaluate entities the model explicitly reported
            if entity not in model_curr_at:
                continue
            sim_curr_val   = sim_curr_at.get(entity)
            sim_prev_val   = sim_prev_at.get(entity)
            model_curr_val = model_curr_at.get(entity)
            model_prev_val = model_prev_at.get(entity)

            if model_curr_val == sim_curr_val:
                continue

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

        pddl_done = phase_end
        model_prev_at.update(model_curr_at)
        n_steps_checked += 1

    return {
        "n_state_blocks":     n_state_blocks,
        "state_parseable":    state_parseable_count > 0,
        "n_steps_checked":    n_steps_checked,
        "first_illegal_step": committed_bad_idx,
        "error_counts":       total_errors,
    }


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

def verify_blocksworld(
    init_nl: str,
    goal_nl: str,
    raw_output: str,
) -> Tuple[bool, int, str]:
    """Returns (success, n_actions_in_pddl_block, error_message)."""
    pddl_lines = extract_pddl_plan(raw_output)
    if not pddl_lines:
        return False, 0, "No 'The PDDL version:' block found"

    actions: List[Tuple[str, List[str]]] = []
    for line in pddl_lines:
        try:
            actions.append(parse_pddl_bw_action(line))
        except ValueError as e:
            return False, len(actions), f"Unparseable PDDL: {e}"

    initial = nl_to_4op_state(init_nl)
    states, bad_idx, bad_msg = rollout_4op(initial, actions)
    if bad_idx is not None:
        return False, len(actions), f"Illegal action at step {bad_idx + 1}: {bad_msg}"

    if bw_4op_goal_satisfied(states[-1], goal_nl):
        return True, len(actions), ""
    return False, len(actions), "Goal not reached after executing PDDL plan"


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def _print_paper_prompt_structure(domain: str, messages: List[Dict[str, str]]) -> None:
    print(f"\n  {'─'*56}")
    print(f"  PROMPT STRUCTURE ({domain} / paper replication)")
    print(f"  Messages: {len(messages)}  "
          f"(system + {len(messages)-2}×user/asst turns + 1 final user)")
    total_chars = sum(len(m["content"]) for m in messages)
    print(f"  Total chars: {total_chars:,}")
    for i, m in enumerate(messages):
        role = m["role"].upper()
        chars = len(m["content"])
        preview = m["content"][:80].replace("\n", " ")
        print(f"    [{i}] {role:<9} {chars:5,} chars  │ {preview}…")
    print(f"  {'─'*56}\n")


def evaluate_blocksworld(
    problems: List[Tuple[str, str]],
    provider: str,
    model_name: str,
    max_tokens: int = 2000,
    delay_secs: float = 1.0,
    output_dir: Optional[str] = None,
    verbose: bool = False,
    checkpoint_interval: int = 1,
) -> Dict[str, Any]:
    client = get_llm_client(provider)

    print(f"\n{'='*60}")
    print(f"  PAPER REPLICATION — Blocksworld (4-op, PDDL extraction)")
    print(f"  Model   : {model_name}  k={checkpoint_interval}")
    print(f"  Problems: {len(problems)}")
    print(f"{'='*60}")

    # Build the checkpoint instruction injected into the system message
    if checkpoint_interval == 1:
        _ckpt_instr = (
            " After every action, write the current state on a new line starting with '- ',"
            " using the compact format shown in the examples (e.g. '- D on table, A on C on B on table.')."
        )
    else:
        _ckpt_instr = (
            f" After every {checkpoint_interval} actions, write the current state on a new line"
            " starting with '- ', using the compact format shown in the examples."
        )

    results = []
    for idx, (init_nl, goal_nl) in enumerate(problems, start=1):
        print(f"  [{idx:2d}/{len(problems)}] ", end="", flush=True)
        error_msg = ""
        raw = ""
        n_actions = 0
        success = False

        try:
            init_compact = bw_nl_to_compact_initial(init_nl)
            goal_compact = bw_nl_to_compact_goal(goal_nl)
            messages = build_bw_messages(init_compact, goal_compact)
            # Inject checkpoint instruction into system message without touching bw_prompt.py
            messages[0] = dict(messages[0])
            messages[0]["content"] = messages[0]["content"] + _ckpt_instr
            if verbose and idx == 1:
                _print_paper_prompt_structure("blocksworld", messages)
            raw = client.generate_messages(messages, temperature=0.0, max_tokens=max_tokens)
            success, n_actions, error_msg = verify_blocksworld(init_nl, goal_nl, raw)
        except Exception as e:
            error_msg = str(e)

        taxonomy = _compute_error_taxonomy_bw_nl(raw, init_nl) if raw else {}
        tax_str = ""
        if taxonomy and taxonomy.get("state_parseable"):
            ec = taxonomy["error_counts"]
            tax_str = (f"  [om={ec['omission']} hal={ec['hallucination']} "
                       f"prop={ec['propagation']} acc={ec['accumulation']}]")
        elif taxonomy and taxonomy.get("n_state_blocks", 0) == 0:
            tax_str = "  [no state claims]"

        print(f"{'✓' if success else '✗'}  actions={n_actions:2d}  {error_msg or ''}{tax_str}")

        if verbose and not success:
            print(f"\n  ── RAW OUTPUT (problem {idx}) ──────────────────────────")
            print(raw[:3000])
            if len(raw) > 3000:
                print(f"  … [{len(raw)-3000} chars truncated]")
            print(f"  ── END RAW OUTPUT ──────────────────────────────────\n")

        result: Dict[str, Any] = {
            "problem_idx":  idx,
            "initial_state": init_nl,
            "goal_state":    goal_nl,
            "success":       success,
            "n_actions":     n_actions,
            "error":         error_msg,
        }
        if taxonomy:
            result["error_taxonomy"] = {k: v for k, v in taxonomy.items() if k != "block_errors"}
        results.append(result)

        if output_dir:
            os.makedirs(f"{output_dir}/blocksworld", exist_ok=True)
            with open(f"{output_dir}/blocksworld/problem_{idx:03d}.txt", "w") as f:
                f.write(raw)
            with open(f"{output_dir}/blocksworld/problem_{idx:03d}_result.json", "w") as f:
                json.dump(result, f, indent=2)

        if idx < len(problems):
            time.sleep(delay_secs)

    n_success = sum(r["success"] for r in results)
    accuracy = n_success / len(results) * 100 if results else 0.0

    # Aggregate taxonomy
    all_tax = [r["error_taxonomy"] for r in results if "error_taxonomy" in r]
    parseable = [t for t in all_tax if t.get("state_parseable")]
    total_steps = sum(t.get("n_steps_checked", 0) for t in parseable)
    om  = sum(t["error_counts"]["omission"]      for t in parseable)
    hal = sum(t["error_counts"]["hallucination"]  for t in parseable)
    pro = sum(t["error_counts"]["propagation"]    for t in parseable)
    acc = sum(t["error_counts"]["accumulation"]   for t in parseable)
    taxonomy_agg: Dict[str, Any] = {
        "n_problems_with_state_claims": sum(1 for t in all_tax if t.get("n_state_blocks", 0) > 0),
        "n_state_parseable":            len(parseable),
        "n_steps_checked":  total_steps,
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
        "method":              "paper_replication",
        "domain":              "blocksworld",
        "model":               model_name,
        "checkpoint_interval": checkpoint_interval,
        "n_problems":          len(results),
        "n_success":           n_success,
        "accuracy_pct":        round(accuracy, 1),
        "error_taxonomy_agg":  taxonomy_agg,
        "results":             results,
    }

    if output_dir:
        with open(f"{output_dir}/blocksworld/summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    return summary


# ---------------------------------------------------------------------------
# Logistics problem bank (same 15 problems as run_aot_mini.py for comparability)
# ---------------------------------------------------------------------------

LOGISTICS_CITY_MAP = {
    "L0-0": "C0", "L0-1": "C0", "L0-2": "C0",
    "L1-0": "C1", "L1-1": "C1", "L1-2": "C1",
    "L2-0": "C2", "L2-1": "C2", "L2-2": "C2",
}

LOGISTICS_PROBLEMS: List[Tuple[str, str]] = [
    ("Package P0 is at L0-2. Truck T0 is at L0-1. Airplane A0 is at L0-0.",
     "Package P0 is at L0-1."),
    ("Package P0 is at L0-1. Truck T0 is at L0-0. Airplane A0 is at L0-0.",
     "Package P0 is at L0-2."),
    ("Package P0 is at L1-2. Truck T1 is at L1-1. Airplane A0 is at L0-0.",
     "Package P0 is at L1-0."),
    ("Package P0 is at L0-1. Truck T0 is at L0-2. Truck T1 is at L1-2. Airplane A0 is at L0-0.",
     "Package P0 is at L1-1."),
    ("Package P0 is at L0-2. Truck T0 is at L0-1. Truck T1 is at L1-1. Airplane A0 is at L1-0.",
     "Package P0 is at L1-2."),
    ("Package P0 is at L1-1. Truck T0 is at L0-0. Truck T1 is at L1-0. Airplane A0 is at L0-0.",
     "Package P0 is at L0-2."),
    ("Package P0 is at L0-1. Package P1 is at L0-2. Truck T0 is at L0-0. Airplane A0 is at L0-0.",
     "Package P0 is at L0-2. Package P1 is at L0-1."),
    ("Package P0 is at L0-0. Package P1 is at L0-2. Truck T0 is at L0-1. Airplane A0 is at L0-0.",
     "Package P0 is at L0-2. Package P1 is at L0-0."),
    ("Package P0 is at L0-1. Package P1 is at L1-1. "
     "Truck T0 is at L0-1. Truck T1 is at L1-1. Airplane A0 is at L0-0.",
     "Package P0 is at L1-1. Package P1 is at L0-1."),
    ("Package P0 is at L0-2. Package P1 is at L1-2. "
     "Truck T0 is at L0-0. Truck T1 is at L1-0. Airplane A0 is at L0-0.",
     "Package P0 is at L1-1. Package P1 is at L0-1."),
    ("Package P0 is at L0-1. Package P1 is at L0-2. Package P2 is at L1-1. "
     "Truck T0 is at L0-0. Truck T1 is at L1-0. Airplane A0 is at L0-0.",
     "Package P0 is at L1-1. Package P1 is at L0-1. Package P2 is at L0-2."),
    ("Package P0 is at L1-2. Package P1 is at L0-1. Package P2 is at L1-1. "
     "Truck T0 is at L0-2. Truck T1 is at L1-2. Airplane A0 is at L1-0.",
     "Package P0 is at L0-1. Package P1 is at L1-2. Package P2 is at L0-2."),
    ("Package P0 is at L0-1. Truck T0 is at L0-2. Truck T1 is at L1-1. "
     "Truck T2 is at L2-1. Airplane A0 is at L0-0. Airplane A1 is at L2-0.",
     "Package P0 is at L2-2."),
    ("Package P0 is at L2-1. Package P1 is at L0-2. "
     "Truck T0 is at L0-1. Truck T1 is at L1-2. Truck T2 is at L2-0. "
     "Airplane A0 is at L1-0. Airplane A1 is at L2-0.",
     "Package P0 is at L0-1. Package P1 is at L2-2."),
    ("Package P0 is at L0-2. Package P1 is at L1-1. Package P2 is at L2-2. "
     "Truck T0 is at L0-1. Truck T1 is at L1-0. Truck T2 is at L2-1. "
     "Airplane A0 is at L0-0. Airplane A1 is at L1-0.",
     "Package P0 is at L2-1. Package P1 is at L0-2. Package P2 is at L1-2."),
]


# ---------------------------------------------------------------------------
# Logistics PDDL extractor and verifier
# ---------------------------------------------------------------------------

def _strip_pddl_parens(line: str) -> str:
    """'(drive-truck T1 L1-0 L1-1 C1)' → 'drive-truck T1 L1-0 L1-1 C1'."""
    s = line.strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
    return s


def verify_logistics(
    init_nl: str,
    goal_nl: str,
    city_map: Dict[str, str],
    raw_output: str,
) -> Tuple[bool, int, str]:
    """Returns (success, n_actions_in_pddl_block, error_message)."""
    pddl_lines = extract_pddl_plan(raw_output)
    if not pddl_lines:
        return False, 0, "No 'The PDDL version:' block found"

    meta = log_validate_city_map(city_map)
    initial = log_nl_to_canonical(init_nl)

    actions = []
    for line in pddl_lines:
        plain = _strip_pddl_parens(line)
        try:
            actions.append(log_parse_action_text(plain))
        except ValueError as e:
            return False, len(actions), f"Unparseable PDDL: {e}"

    states, bad_idx, bad_text = log_rollout(initial, actions, meta)
    if bad_idx is not None:
        return False, len(actions), f"Illegal action at step {bad_idx}: {bad_text!r}"

    # Goal check: all packages in goal_nl must be at correct location in final state
    final = states[-1]
    import re as _re
    for m in _re.finditer(r"Package\s+(\S+?)\s+is\s+at\s+(\S+?)\.?(?:\s|$)", goal_nl, _re.IGNORECASE):
        pkg = m.group(1).lower()
        loc = m.group(2).lower().rstrip(".")
        if final["at"].get(pkg) != loc:
            return False, len(actions), f"Goal not reached: {pkg} at {final['at'].get(pkg)!r}, expected {loc!r}"

    return True, len(actions), ""


# ---------------------------------------------------------------------------
# Logistics evaluation loop
# ---------------------------------------------------------------------------

def evaluate_logistics(
    problems: List[Tuple[str, str]],
    provider: str,
    model_name: str,
    city_map: Optional[Dict[str, str]] = None,
    max_tokens: int = 4000,
    delay_secs: float = 1.0,
    output_dir: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    client = get_llm_client(provider)
    _city_map = city_map or LOGISTICS_CITY_MAP

    print(f"\n{'='*60}")
    print(f"  PAPER REPLICATION — Logistics (PDDL extraction)")
    print(f"  Model   : {model_name}")
    print(f"  Problems: {len(problems)}")
    print(f"{'='*60}")

    results = []
    for idx, (init_nl, goal_nl) in enumerate(problems, start=1):
        print(f"  [{idx:2d}/{len(problems)}] ", end="", flush=True)
        error_msg = ""
        raw = ""
        n_actions = 0
        success = False

        try:
            start_block = log_nl_to_compact(init_nl)
            goal_str    = log_goal_nl_to_compact(goal_nl)
            per_map = derive_city_map(init_nl, goal_nl) or {k.lower(): v.lower()
                                                             for k, v in _city_map.items()}
            messages = build_log_messages(start_block, goal_str)
            if verbose and idx == 1:
                _print_paper_prompt_structure("logistics", messages)
            raw = client.generate_messages(messages, temperature=0.0, max_tokens=max_tokens)
            success, n_actions, error_msg = verify_logistics(init_nl, goal_nl, per_map, raw)
        except Exception as e:
            error_msg = str(e)

        taxonomy: Dict[str, Any] = {}
        if raw:
            try:
                per_map_local = derive_city_map(init_nl, goal_nl) or {
                    k.lower(): v.lower() for k, v in _city_map.items()
                }
                taxonomy = _compute_error_taxonomy_log_nl(raw, init_nl, per_map_local)
            except Exception:
                pass

        tax_str = ""
        if taxonomy and taxonomy.get("state_parseable"):
            ec = taxonomy["error_counts"]
            tax_str = (f"  [om={ec['omission']} hal={ec['hallucination']} "
                       f"prop={ec['propagation']} acc={ec['accumulation']}]")
        print(f"{'✓' if success else '✗'}  actions={n_actions:2d}  {error_msg or ''}{tax_str}")

        if verbose and not success:
            print(f"\n  ── RAW OUTPUT (problem {idx}) ──────────────────────────")
            print(raw[:3000])
            if len(raw) > 3000:
                print(f"  … [{len(raw)-3000} chars truncated]")
            print(f"  ── END RAW OUTPUT ──────────────────────────────────\n")

        result = {
            "problem_idx":   idx,
            "initial_state": init_nl,
            "goal_state":    goal_nl,
            "success":       success,
            "n_actions":     n_actions,
            "error":         error_msg,
        }
        if taxonomy:
            result["error_taxonomy"] = {k: v for k, v in taxonomy.items()
                                        if k != "block_errors"}
        results.append(result)

        if output_dir:
            os.makedirs(f"{output_dir}/logistics", exist_ok=True)
            with open(f"{output_dir}/logistics/problem_{idx:03d}.txt", "w") as f:
                f.write(raw)
            with open(f"{output_dir}/logistics/problem_{idx:03d}_result.json", "w") as f:
                json.dump(result, f, indent=2)

        if idx < len(problems):
            time.sleep(delay_secs)

    n_success = sum(r["success"] for r in results)
    accuracy = n_success / len(results) * 100 if results else 0.0

    all_tax = [r["error_taxonomy"] for r in results if "error_taxonomy" in r]
    parseable = [t for t in all_tax if t.get("state_parseable")]
    total_steps = sum(t.get("n_steps_checked", 0) for t in parseable)
    om  = sum(t["error_counts"]["omission"]      for t in parseable)
    hal = sum(t["error_counts"]["hallucination"]  for t in parseable)
    pro = sum(t["error_counts"]["propagation"]    for t in parseable)
    acc = sum(t["error_counts"]["accumulation"]   for t in parseable)
    taxonomy_agg: Dict[str, Any] = {
        "n_problems_with_state_blocks": sum(1 for t in all_tax if t.get("n_state_blocks", 0) > 0),
        "n_state_parseable":            len(parseable),
        "n_phases_checked":             total_steps,
        "omission":      om,
        "hallucination": hal,
        "propagation":   pro,
        "accumulation":  acc,
        "omission_per_phase":      round(om  / total_steps, 4) if total_steps else None,
        "hallucination_per_phase": round(hal / total_steps, 4) if total_steps else None,
        "propagation_per_phase":   round(pro / total_steps, 4) if total_steps else None,
        "accumulation_per_phase":  round(acc / total_steps, 4) if total_steps else None,
        "n_with_illegal_action": sum(
            1 for t in all_tax if t.get("first_illegal_step") is not None
        ),
    }

    summary = {
        "method":             "paper_replication",
        "domain":             "logistics",
        "model":              model_name,
        "n_problems":         len(results),
        "n_success":          n_success,
        "accuracy_pct":       round(accuracy, 1),
        "error_taxonomy_agg": taxonomy_agg,
        "results":            results,
    }

    if output_dir:
        with open(f"{output_dir}/logistics/summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Paper replication runner (AoT+ Appendix B.3/B.4)")
    parser.add_argument("--provider", default="openai",
                        choices=["openai", "sandbox", "azure_openai", "anthropic"])
    parser.add_argument("--domain", default="both",
                        choices=["blocksworld", "logistics", "both"])
    parser.add_argument("--n_bw",  type=int, default=20)
    parser.add_argument("--n_log", type=int, default=15)
    parser.add_argument("--no_planbench", action="store_true",
                        help="Use hardcoded fallback bank instead of PlanBench instances")
    parser.add_argument("--cache_dir", default=".planbench_cache")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_interval", type=int, default=1,
                        help="Output state every N actions (default 1 = after every action)")
    parser.add_argument("--max_tokens", type=int, default=3000)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--output_dir", default="outputs/paper_repl")
    parser.add_argument("--verbose", action="store_true",
                        help="Print prompt structure for problem 1 and raw output for failures")
    args = parser.parse_args()

    available = Config.get_available_providers()
    if not available:
        print("Error: No LLM provider configured. Check your .env file.")
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

    use_planbench = not args.no_planbench
    bw_probs   = BLOCKSWORLD_PROBLEMS[:args.n_bw]
    log_probs  = LOGISTICS_PROBLEMS[:args.n_log]
    city_map   = LOGISTICS_CITY_MAP

    if use_planbench:
        try:
            from src.benchmarks.planbench import get_blocksworld_problems, get_logistics_problems
            print(f"Downloading PlanBench instances (seed={args.seed}, cache={args.cache_dir!r}) …")
            bw_dicts              = get_blocksworld_problems(args.n_bw,  args.seed, cache_dir=args.cache_dir)
            log_dicts, city_map   = get_logistics_problems( args.n_log, args.seed, cache_dir=args.cache_dir)
            bw_probs  = [(d["initial_state"], d["goal_state"]) for d in bw_dicts]
            log_probs = [(d["initial_state"], d["goal_state"]) for d in log_dicts]
            print(f"  Blocksworld : {len(bw_probs)} problems")
            print(f"  Logistics   : {len(log_probs)} problems")
        except Exception as e:
            print(f"Warning: PlanBench download failed ({e}). Using hardcoded bank.")
            use_planbench = False

    summaries: Dict[str, Any] = {}

    if args.domain in ("blocksworld", "both"):
        summaries["blocksworld"] = evaluate_blocksworld(
            problems=bw_probs, provider=args.provider, model_name=model_name,
            max_tokens=args.max_tokens, delay_secs=args.delay, output_dir=args.output_dir,
            verbose=args.verbose, checkpoint_interval=args.checkpoint_interval,
        )

    if args.domain in ("logistics", "both"):
        summaries["logistics"] = evaluate_logistics(
            problems=log_probs, provider=args.provider, model_name=model_name,
            city_map=city_map,  max_tokens=args.max_tokens, delay_secs=args.delay,
            output_dir=args.output_dir, verbose=args.verbose,
        )

    source = "PlanBench" if use_planbench else "fallback bank"
    print(f"\n{'='*60}")
    print(f"  Paper Replication — {model_name}  [{source}]")
    print(f"  {'Domain':<15} {'Solved':>8} {'Accuracy':>10}")
    print(f"  {'-'*35}")
    for domain, s in summaries.items():
        print(f"  {domain:<15} {s['n_success']:>4}/{s['n_problems']:<4} {s['accuracy_pct']:>8.1f}%")
        if s.get("error_taxonomy_agg"):
            ta = s["error_taxonomy_agg"]
            ns_key = "n_phases_checked" if domain == "logistics" else "n_steps_checked"
            ns = ta.get(ns_key, ta.get("n_steps_checked", 0))
            rate_sfx = "phase" if domain == "logistics" else "step"
            print(f"    taxonomy ({ns} {rate_sfx}s, {ta['n_state_parseable']} problems parseable):")
            print(f"      raw:  om={ta['omission']} hal={ta['hallucination']} "
                  f"prop={ta['propagation']} acc={ta['accumulation']}")
            per_om  = ta.get(f"omission_per_{rate_sfx}")
            per_hal = ta.get(f"hallucination_per_{rate_sfx}")
            per_pro = ta.get(f"propagation_per_{rate_sfx}")
            per_acc = ta.get(f"accumulation_per_{rate_sfx}")
            if per_om is not None:
                print(f"      /{rate_sfx}: om={per_om:.4f} hal={per_hal:.4f} "
                      f"prop={per_pro:.4f} acc={per_acc:.4f}")
    print(f"{'='*60}\n")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(f"{args.output_dir}/final_summary.json", "w") as f:
            json.dump({k: {kk: vv for kk, vv in v.items() if kk != "results"}
                       for k, v in summaries.items()}, f, indent=2)
        print(f"Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
