"""
Experiment runner for all AoT+ conditions.

Saves raw LLM outputs + metadata sidecars, then runs Abby's evaluators
and writes a JSON results file per condition.

Usage:
    python -m src.experiments.run_experiments \\
        --domain blocksworld \\
        --condition aot_text \\
        --provider sandbox \\
        --num_problems 5 \\
        --output_dir outputs

Conditions:
    aot_text          — original AoT+ with natural-language STATE N: restatements
    json_structured   — AoT+ with STATE_JSON fenced blocks (no goal residual)
    json_goal         — AoT+ with STATE_JSON + goal_satisfied / goal_remaining fields

Domains: blocksworld | logistics
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.aot_plus.blocksworld_examples import (
    get_blocksworld_json_examples,
    get_blocksworld_json_goal_examples,
)
from src.aot_plus.logistics_examples import (
    get_logistics_json_examples,
    get_logistics_json_goal_examples,
)
from src.aot_plus.core import AoTPlus
from src.domains.blocksworld import BlocksWorldDomain
from src.domains.logistics import LogisticsDomain
from src.eval.blocksworld_aot_eval import evaluate_blocksworld_aot_output
from src.eval.blocksworld_json_goal_eval import evaluate_blocksworld_json_goal_output
from src.eval.blocksworld_json_structured_eval import evaluate_blocksworld_json_structured_output
from src.eval.logistics_aot_eval import evaluate_logistics_aot_output
from src.eval.logistics_json_goal_eval import evaluate_logistics_json_goal_output
from src.eval.logistics_json_structured_eval import evaluate_logistics_json_structured_output
from src.sim.blocksworld_sim import nl_state_to_canonical as bw_nl_to_canonical
from src.sim.blocksworld_sim import validate_json_state as bw_validate_json
from src.sim.logistics_sim import nl_state_to_canonical as log_nl_to_canonical
from src.utils.config import Config


# ---------------------------------------------------------------------------
# Problem banks
# ---------------------------------------------------------------------------

BLOCKSWORLD_PROBLEMS: List[Tuple[str, str]] = [
    (
        "Block A is on the table, Block B is on Block A, Block C is on the table",
        "Block A is on Block B, Block B is on Block C, Block C is on the table",
    ),
    (
        "Block A is on Block B, Block B is on Block C, Block C is on the table",
        "Block C is on Block B, Block B is on Block A, Block A is on the table",
    ),
    (
        "Block A is on the table, Block B is on the table, Block C is on the table",
        "Block C is on Block B, Block B is on Block A, Block A is on the table",
    ),
    (
        "Block A is on the table, Block B is on Block A, Block C is on Block B, Block D is on the table",
        "Block D is on Block C, Block C is on Block B, Block B is on Block A, Block A is on the table",
    ),
    (
        "Block A is on Block B, Block B is on Block C, Block C is on Block D, Block D is on the table",
        "Block D is on Block A, Block A is on the table, Block B is on the table, Block C is on the table",
    ),
]

LOGISTICS_PROBLEMS: List[Tuple[str, str]] = [
    # Problem 1: intra-city delivery (P0 from L0-2 to L0-1)
    (
        "Package P0 is at L0-2. Truck T0 is at L0-1. Airplane A0 is at L0-0.",
        "Package P0 is at L0-1.",
    ),
    # Problem 2: cross-city delivery (P0 from L0-1 to L1-1)
    (
        "Package P0 is at L0-1. Truck T0 is at L0-2. Truck T1 is at L1-2. Airplane A0 is at L0-0.",
        "Package P0 is at L1-1.",
    ),
    # Problem 3: two packages, intra-city swap
    (
        "Package P0 is at L0-1. Package P1 is at L0-2. Truck T0 is at L0-0. Airplane A0 is at L0-0.",
        "Package P0 is at L0-2. Package P1 is at L0-1.",
    ),
    # Problem 4: two packages, cross-city delivery
    (
        "Package P0 is at L0-1. Package P1 is at L1-1. "
        "Truck T0 is at L0-1. Truck T1 is at L1-1. Airplane A0 is at L0-0.",
        "Package P0 is at L1-1. Package P1 is at L0-1.",
    ),
]

# PDDL location names: locations ending in "-0" are airports.
DEFAULT_CITY_MAP = {
    "L0-0": "C0", "L0-1": "C0", "L0-2": "C0",
    "L1-0": "C1", "L1-1": "C1", "L1-2": "C1",
}


# ---------------------------------------------------------------------------
# State serialization helpers (NL → JSON string for prompt)
# ---------------------------------------------------------------------------

def _bw_nl_to_json_str(nl: str) -> str:
    state = bw_nl_to_canonical(nl)
    return json.dumps(state)


def _log_nl_to_json_str(nl: str) -> str:
    state = log_nl_to_canonical(nl)
    return json.dumps(state)


def _compute_log_goal_diff(current: dict, goal: dict):
    satisfied, remaining = [], []
    for pkg, loc in goal["at"].items():
        if loc is None:
            continue
        label = f"{pkg} at {loc}"
        if current["at"].get(pkg) == loc:
            satisfied.append(label)
        else:
            remaining.append(label)
    return sorted(satisfied), sorted(remaining)


def _log_nl_to_json_goal_str(nl_initial: str, nl_goal: str) -> str:
    init = log_nl_to_canonical(nl_initial)
    goal = log_nl_to_canonical(nl_goal)
    satisfied, remaining = _compute_log_goal_diff(init, goal)
    init["goal_satisfied"] = satisfied
    init["goal_remaining"] = remaining
    return json.dumps(init)


def _bw_nl_to_json_goal_str(nl_initial: str, nl_goal: str) -> str:
    """Return initial state JSON with goal_satisfied / goal_remaining appended."""
    import re
    init = bw_nl_to_canonical(nl_initial)
    goal = bw_nl_to_canonical(nl_goal)

    satisfied, remaining = _compute_bw_goal_diff(init, goal)
    init["goal_satisfied"] = satisfied
    init["goal_remaining"] = remaining
    return json.dumps(init)


def _compute_bw_goal_diff(current: dict, goal: dict):
    satisfied, remaining = [], []
    for block, dest in goal["on"].items():
        label = f"{block} on {dest}"
        if current["on"].get(block) == dest:
            satisfied.append(label)
        else:
            remaining.append(label)
    for block in goal["clear"]:
        label = f"{block} clear"
        if block in current["clear"]:
            satisfied.append(label)
        else:
            remaining.append(label)
    return sorted(satisfied), sorted(remaining)


# ---------------------------------------------------------------------------
# AoTPlus factory per condition
# ---------------------------------------------------------------------------

def _make_solver(domain: str, condition: str, provider: str) -> AoTPlus:
    use_json = condition in ("json_structured", "json_goal")

    if domain == "blocksworld":
        domain_desc = BlocksWorldDomain.get_domain_description()
        init_ex, goal_ex, sol_ex, rand_ex = BlocksWorldDomain.generate_example_problem()
    else:
        domain_desc = LogisticsDomain.get_domain_description()
        init_ex, goal_ex, sol_ex, rand_ex = LogisticsDomain.generate_example_problem()

    solver = AoTPlus(domain_desc, provider, use_json_state=use_json)

    if use_json:
        if domain == "blocksworld":
            examples = (get_blocksworld_json_goal_examples() if condition == "json_goal"
                        else get_blocksworld_json_examples())
        else:
            examples = (get_logistics_json_goal_examples() if condition == "json_goal"
                        else get_logistics_json_examples())
        for ex in examples:
            solver.add_formatted_example(ex)
    else:
        solver.add_example(init_ex, goal_ex, sol_ex, rand_ex)

    return solver


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def _bw_initial_json(nl: str) -> dict:
    return bw_nl_to_canonical(nl)


def _log_initial_json(nl: str) -> dict:
    return log_nl_to_canonical(nl)


# ---------------------------------------------------------------------------
# Output file management
# ---------------------------------------------------------------------------

def _raw_path(output_dir: str, domain: str, condition: str, idx: int) -> str:
    return os.path.join(output_dir, domain, condition, f"problem_{idx:03d}.txt")


def _meta_path(output_dir: str, domain: str, condition: str, idx: int) -> str:
    return os.path.join(output_dir, domain, condition, f"problem_{idx:03d}_meta.json")


def _save(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


# ---------------------------------------------------------------------------
# Single-problem run
# ---------------------------------------------------------------------------

def _run_one(
    solver: AoTPlus,
    domain: str,
    condition: str,
    nl_initial: str,
    nl_goal: str,
    output_dir: str,
    idx: int,
    temperature: float,
    max_tokens: int,
    model_name: str,
) -> Dict[str, Any]:
    # Build prompt inputs
    if condition == "aot_text":
        prompt_initial = nl_initial
        prompt_goal    = nl_goal
    elif condition == "json_goal":
        if domain == "blocksworld":
            prompt_initial = _bw_nl_to_json_goal_str(nl_initial, nl_goal)
            prompt_goal    = _bw_nl_to_json_str(nl_goal)
        else:
            prompt_initial = _log_nl_to_json_goal_str(nl_initial, nl_goal)
            prompt_goal    = _log_nl_to_json_str(nl_goal)
    else:  # json_structured
        if domain == "blocksworld":
            prompt_initial = _bw_nl_to_json_str(nl_initial)
            prompt_goal    = _bw_nl_to_json_str(nl_goal)
        else:
            prompt_initial = _log_nl_to_json_str(nl_initial)
            prompt_goal    = _log_nl_to_json_str(nl_goal)

    raw_output = solver.solve(prompt_initial, prompt_goal, temperature=temperature,
                              max_tokens=max_tokens)
    usage = getattr(solver.llm_client, "last_usage", None) or {}

    # Save raw output
    raw_path = _raw_path(output_dir, domain, condition, idx)
    _save(raw_path, raw_output)

    # Build metadata
    if domain == "blocksworld":
        init_json = _bw_initial_json(nl_initial)
        goal_json = _bw_initial_json(nl_goal)
    else:
        init_json = _log_initial_json(nl_initial)
        goal_json = _log_initial_json(nl_goal)

    meta = {
        "problem_id":         f"{idx:03d}",
        "domain":             domain,
        "condition":          condition,
        "initial_state_nl":   nl_initial,
        "goal_state_nl":      nl_goal,
        "initial_state":      init_json,
        "goal_state":         goal_json,
        "memoization_interval": solver.prompt_builder.state_memoization_interval,
        "model":              model_name,
        "prompt_tokens":      usage.get("prompt_tokens"),
        "completion_tokens":  usage.get("completion_tokens"),
        "total_tokens":       usage.get("total_tokens"),
    }
    _save(_meta_path(output_dir, domain, condition, idx), json.dumps(meta, indent=2))

    # Run evaluation
    try:
        if domain == "blocksworld":
            if condition == "aot_text":
                result = evaluate_blocksworld_aot_output(nl_initial, raw_output)
            elif condition == "json_goal":
                result = evaluate_blocksworld_json_goal_output(init_json, raw_output)
            else:
                result = evaluate_blocksworld_json_structured_output(init_json, raw_output)
        else:
            if condition == "aot_text":
                result = evaluate_logistics_aot_output(nl_initial, DEFAULT_CITY_MAP, raw_output)
            elif condition == "json_goal":
                result = evaluate_logistics_json_goal_output(init_json, DEFAULT_CITY_MAP, raw_output)
            else:
                result = evaluate_logistics_json_structured_output(init_json, DEFAULT_CITY_MAP, raw_output)

        plan_success = result.first_error_step is None and len(result.actions) > 0
        step_errors  = result.step_errors
        depth_errors = result.depth_errors
        first_error  = result.first_error_step
        n_actions    = len(result.actions)
        error_types_by_step = [
            sorted(d.get("error_types", set())) for d in result.details
        ]
    except Exception as e:
        plan_success = False
        step_errors  = []
        depth_errors = []
        first_error  = None
        n_actions    = 0
        error_types_by_step = []
        print(f"    Eval error: {e}")

    return {
        "problem_id":         idx,
        "initial_state":      nl_initial,
        "goal_state":         nl_goal,
        "plan_success":       plan_success,
        "n_actions":          n_actions,
        "first_error_step":   first_error,
        "step_errors":        step_errors,
        "depth_errors":       depth_errors,
        "error_types_by_step": error_types_by_step,
        "prompt_tokens":      usage.get("prompt_tokens"),
        "completion_tokens":  usage.get("completion_tokens"),
        "total_tokens":       usage.get("total_tokens"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run AoT+ experiments")
    parser.add_argument("--domain",     choices=["blocksworld", "logistics"], default="blocksworld")
    parser.add_argument("--condition",  choices=["aot_text", "json_structured", "json_goal"],
                        default="aot_text")
    parser.add_argument("--provider",   choices=["sandbox", "openai", "azure_openai"],
                        default="sandbox")
    parser.add_argument("--num_problems", type=int, default=5)
    parser.add_argument("--temperature",  type=float, default=0.0)
    parser.add_argument("--max_tokens",   type=int, default=3000)
    parser.add_argument("--output_dir",   default="outputs")
    args = parser.parse_args()

    if args.provider not in Config.get_available_providers():
        print(f"Provider '{args.provider}' not configured. Available: {Config.get_available_providers()}")
        return

    model_name = {
        "sandbox":     Config.SANDBOX_MODEL_NAME,
        "openai":      Config.OPENAI_MODEL_NAME,
        "azure_openai": Config.AZURE_OPENAI_DEPLOYMENT_NAME,
    }.get(args.provider, "unknown")

    problems = BLOCKSWORLD_PROBLEMS if args.domain == "blocksworld" else LOGISTICS_PROBLEMS
    problems = problems[:args.num_problems]

    solver = _make_solver(args.domain, args.condition, args.provider)

    print(f"\n=== AoT+ Experiment ===")
    print(f"  Domain    : {args.domain}")
    print(f"  Condition : {args.condition}")
    print(f"  Model     : {model_name}")
    print(f"  Problems  : {len(problems)}\n")

    all_results = []
    for idx, (nl_init, nl_goal) in enumerate(problems, start=1):
        print(f"  [{idx}/{len(problems)}] Running …", flush=True)
        result = _run_one(
            solver=solver,
            domain=args.domain,
            condition=args.condition,
            nl_initial=nl_init,
            nl_goal=nl_goal,
            output_dir=args.output_dir,
            idx=idx,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            model_name=model_name,
        )
        all_results.append(result)
        status = "✓" if result["plan_success"] else "✗"
        print(f"    {status}  actions={result['n_actions']}  "
              f"first_error={result['first_error_step']}  "
              f"tokens={result['total_tokens']}")

        if idx < len(problems):
            time.sleep(1)

    # Aggregate summary
    n_success = sum(r["plan_success"] for r in all_results)
    total_tokens = sum(r["total_tokens"] or 0 for r in all_results)

    summary = {
        "domain":        args.domain,
        "condition":     args.condition,
        "model":         model_name,
        "num_problems":  len(all_results),
        "num_success":   n_success,
        "success_rate":  n_success / len(all_results) if all_results else 0.0,
        "total_tokens":  total_tokens,
        "results":       all_results,
    }

    out_path = os.path.join(
        args.output_dir, args.domain, args.condition, "summary.json"
    )
    _save(out_path, json.dumps(summary, indent=2))

    print(f"\n=== Summary ===")
    print(f"  Success rate : {n_success}/{len(all_results)} = {summary['success_rate']:.1%}")
    print(f"  Total tokens : {total_tokens}")
    print(f"  Saved to     : {out_path}")


if __name__ == "__main__":
    main()
