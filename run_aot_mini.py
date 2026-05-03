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

from src.aot_plus.core import AoTPlus
from src.aot_plus.blocksworld_examples import get_blocksworld_json_examples
from src.aot_plus.logistics_examples import get_logistics_json_examples
from src.domains.blocksworld import BlocksWorldDomain
from src.domains.logistics import LogisticsDomain
from src.sim.blocksworld_sim import (
    nl_state_to_canonical as bw_nl_to_canonical,
    parse_action_text as bw_parse_action,
    rollout_actions as bw_rollout,
)
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
# Goal checking
# ---------------------------------------------------------------------------

def _bw_goal_satisfied(final_state: Dict[str, Any], goal_nl: str) -> bool:
    """Check if the final blocksworld state satisfies the NL goal.

    Parses only predicates that are *explicitly stated* in goal_nl so that
    partial goal specifications (e.g. "Block C is on Block B" with no mention
    of other blocks) don't generate spurious 'block X on table' constraints.
    """
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
    goal_achieved is True iff all extracted actions are legal AND goal is reached.
    """
    try:
        initial = bw_nl_to_canonical(initial_nl)
    except Exception as e:
        return False, 0, f"Failed to parse initial state: {e}"

    raw_actions = extract_plan_summary(raw_output)
    if not raw_actions:
        return False, 0, "No actions extracted"

    parsed = []
    for a in raw_actions:
        try:
            parsed.append(bw_parse_action(a))
        except ValueError:
            return False, len(parsed), f"Unparseable action: {a!r}"

    states, bad_idx, bad_text = bw_rollout(initial, parsed)
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

def make_aot_plus_solver(domain: str, provider: str) -> AoTPlus:
    """Build an AoT+ solver using the hand-crafted JSON few-shot examples."""
    if domain == "blocksworld":
        domain_desc = BlocksWorldDomain.get_domain_description()
        examples = get_blocksworld_json_examples()
    else:
        domain_desc = LogisticsDomain.get_domain_description()
        examples = get_logistics_json_examples()

    solver = AoTPlus(domain_desc, provider, use_json_state=True)
    for ex in examples:
        solver.add_formatted_example(ex)
    return solver


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate_domain(
    domain: str,
    problems: List[Tuple[str, str]],
    provider: str,
    model_name: str,
    max_tokens: int = 3000,
    delay_secs: float = 1.0,
    output_dir: Optional[str] = None,
    city_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    solver = make_aot_plus_solver(domain, provider)
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
            # Convert NL states to JSON for the JSON-mode prompt.
            # Validation still uses NL (goal satisfaction checks are NL-based).
            if domain == "blocksworld":
                init_json = json.dumps(bw_nl_to_canonical(init_nl), sort_keys=True)
                goal_json = json.dumps(bw_nl_to_canonical(goal_nl), sort_keys=True)
            else:
                init_json = json.dumps(log_nl_to_canonical(init_nl), sort_keys=True)
                goal_json = json.dumps(log_nl_to_canonical(goal_nl), sort_keys=True)

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
                raw = solver.solve(init_json, goal_json, temperature=0.0,
                                   max_tokens=max_tokens, extra_context=extra)
            else:
                raw = solver.solve(init_json, goal_json, temperature=0.0, max_tokens=max_tokens)

            if domain == "blocksworld":
                success, n_actions, error_msg = verify_blocksworld(init_nl, goal_nl, raw)
            else:
                success, n_actions, error_msg = verify_logistics(
                    init_nl, goal_nl, city_map or LOGISTICS_CITY_MAP, raw
                )

        except Exception as e:
            error_msg = str(e)
            success = False

        status = "✓" if success else "✗"
        print(f"{status}  actions={n_actions:2d}  {error_msg or ''}")

        results.append({
            "problem_idx": idx,
            "initial_state": init_nl,
            "goal_state":    goal_nl,
            "success":       success,
            "n_actions":     n_actions,
            "error":         error_msg,
        })

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

    summary = {
        "domain":      domain,
        "model":       model_name,
        "method":      "AoT+",
        "n_problems":  len(results),
        "n_success":   n_success,
        "accuracy_pct": round(accuracy, 1),
        "results":     results,
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
    parser = argparse.ArgumentParser(description="Run AoT+ with gpt-4o-mini")
    parser.add_argument("--provider", default="openai",
                        choices=["openai", "sandbox", "azure_openai"])
    parser.add_argument("--domain",   default="both",
                        choices=["blocksworld", "logistics", "both"])
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
    }
    model_name = model_map[args.provider]

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
            max_tokens=args.max_tokens,
            delay_secs=args.delay,
            output_dir=args.output_dir,
        )

    if args.domain in ("logistics", "both"):
        summaries["logistics"] = evaluate_domain(
            domain="logistics",
            problems=log_probs,
            provider=args.provider,
            model_name=model_name,
            max_tokens=args.max_tokens,
            delay_secs=args.delay,
            output_dir=args.output_dir,
            city_map=city_map,
        )

    # Print results table
    source = "PlanBench" if use_planbench else "fallback bank"
    print(f"\n{'='*60}")
    print(f"  AoT+ Results — {model_name}  [{source}]")
    print(f"  {'Domain':<15} {'Solved':>8} {'Accuracy':>10}")
    print(f"  {'-'*35}")
    for domain, s in summaries.items():
        print(f"  {domain:<15} {s['n_success']:>4}/{s['n_problems']:<4} {s['accuracy_pct']:>8.1f}%")
    print(f"{'='*60}\n")

    if args.output_dir:
        with open(f"{args.output_dir}/final_summary.json", "w") as f:
            json.dump({k: {kk: vv for kk, vv in v.items() if kk != "results"}
                       for k, v in summaries.items()}, f, indent=2)
        print(f"Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
