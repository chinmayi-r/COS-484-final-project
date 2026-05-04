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
) -> Dict[str, Any]:
    client = get_llm_client(provider)

    print(f"\n{'='*60}")
    print(f"  PAPER REPLICATION — Blocksworld (4-op, PDDL extraction)")
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
            init_compact = bw_nl_to_compact_initial(init_nl)
            goal_compact = bw_nl_to_compact_goal(goal_nl)
            messages = build_bw_messages(init_compact, goal_compact)
            if verbose and idx == 1:
                _print_paper_prompt_structure("blocksworld", messages)
            raw = client.generate_messages(messages, temperature=0.0, max_tokens=max_tokens)
            success, n_actions, error_msg = verify_blocksworld(init_nl, goal_nl, raw)
        except Exception as e:
            error_msg = str(e)

        print(f"{'✓' if success else '✗'}  actions={n_actions:2d}  {error_msg or ''}")

        if verbose and not success:
            print(f"\n  ── RAW OUTPUT (problem {idx}) ──────────────────────────")
            print(raw[:3000])
            if len(raw) > 3000:
                print(f"  … [{len(raw)-3000} chars truncated]")
            print(f"  ── END RAW OUTPUT ──────────────────────────────────\n")

        result = {
            "problem_idx":  idx,
            "initial_state": init_nl,
            "goal_state":    goal_nl,
            "success":       success,
            "n_actions":     n_actions,
            "error":         error_msg,
        }
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

    summary = {
        "method":       "paper_replication",
        "domain":       "blocksworld",
        "model":        model_name,
        "n_problems":   len(results),
        "n_success":    n_success,
        "accuracy_pct": round(accuracy, 1),
        "results":      results,
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

        print(f"{'✓' if success else '✗'}  actions={n_actions:2d}  {error_msg or ''}")

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

    summary = {
        "method":       "paper_replication",
        "domain":       "logistics",
        "model":        model_name,
        "n_problems":   len(results),
        "n_success":    n_success,
        "accuracy_pct": round(accuracy, 1),
        "results":      results,
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
                        choices=["openai", "sandbox", "azure_openai"])
    parser.add_argument("--domain", default="both",
                        choices=["blocksworld", "logistics", "both"])
    parser.add_argument("--n_bw",  type=int, default=20)
    parser.add_argument("--n_log", type=int, default=15)
    parser.add_argument("--no_planbench", action="store_true",
                        help="Use hardcoded fallback bank instead of PlanBench instances")
    parser.add_argument("--cache_dir", default=".planbench_cache")
    parser.add_argument("--seed", type=int, default=42)
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
            verbose=args.verbose,
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
    print(f"{'='*60}\n")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(f"{args.output_dir}/final_summary.json", "w") as f:
            json.dump({k: {kk: vv for kk, vv in v.items() if kk != "results"}
                       for k, v in summaries.items()}, f, indent=2)
        print(f"Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
