"""
Paper replication runner — AoT+ (Appendix B.3, Blocksworld).

Uses the EXACT system prompt and verbatim few-shot examples from the paper.
Plan is extracted exclusively from the 'The PDDL version:' block.
Validated against the 4-operator Blocksworld simulator.

Usage:
    python run_paper_repl.py --provider sandbox --domain blocksworld

Logistics replication requires B.4 examples and is NOT yet implemented here.
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

def evaluate_blocksworld(
    problems: List[Tuple[str, str]],
    provider: str,
    model_name: str,
    max_tokens: int = 2000,
    delay_secs: float = 1.0,
    output_dir: Optional[str] = None,
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
            raw = client.generate_messages(messages, temperature=0.0, max_tokens=max_tokens)
            success, n_actions, error_msg = verify_blocksworld(init_nl, goal_nl, raw)
        except Exception as e:
            error_msg = str(e)

        print(f"{'✓' if success else '✗'}  actions={n_actions:2d}  {error_msg or ''}")

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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Paper replication runner (AoT+ Appendix B.3)")
    parser.add_argument("--provider", default="openai",
                        choices=["openai", "sandbox", "azure_openai"])
    parser.add_argument("--domain", default="blocksworld",
                        choices=["blocksworld"],
                        help="Only blocksworld is implemented; logistics requires B.4 examples.")
    parser.add_argument("--n_problems", type=int, default=20)
    parser.add_argument("--no_planbench", action="store_true",
                        help="Use hardcoded fallback bank instead of PlanBench instances")
    parser.add_argument("--cache_dir", default=".planbench_cache")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_tokens", type=int, default=2000)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--output_dir", default="outputs/paper_repl")
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
    }
    model_name = model_map[args.provider]

    # Load problems
    use_planbench = not args.no_planbench
    problems = BLOCKSWORLD_PROBLEMS[:args.n_problems]

    if use_planbench:
        try:
            from src.benchmarks.planbench import get_blocksworld_problems
            print(f"Downloading PlanBench instances (seed={args.seed}, cache={args.cache_dir!r}) …")
            bw_dicts = get_blocksworld_problems(args.n_problems, args.seed, cache_dir=args.cache_dir)
            problems = [(d["initial_state"], d["goal_state"]) for d in bw_dicts]
            print(f"  Blocksworld: {len(problems)} problems "
                  f"(ids {[d['instance_id'] for d in bw_dicts]})")
        except Exception as e:
            print(f"Warning: PlanBench download failed ({e}). Falling back to hardcoded bank.")
            use_planbench = False

    summary = evaluate_blocksworld(
        problems=problems,
        provider=args.provider,
        model_name=model_name,
        max_tokens=args.max_tokens,
        delay_secs=args.delay,
        output_dir=args.output_dir,
    )

    source = "PlanBench" if use_planbench else "fallback bank"
    print(f"\n{'='*60}")
    print(f"  Paper Replication — {model_name}  [{source}]")
    print(f"  Blocksworld: {summary['n_success']}/{summary['n_problems']} "
          f"({summary['accuracy_pct']:.1f}%)")
    print(f"{'='*60}\n")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(f"{args.output_dir}/final_summary.json", "w") as f:
            json.dump({k: v for k, v in summary.items() if k != "results"}, f, indent=2)
        print(f"Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
