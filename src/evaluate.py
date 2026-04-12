"""
Evaluation script for AoT+ on Blocksworld and Logistics domains.

Measures plan success rate: the fraction of problems for which the model
produces a valid action sequence that reaches the goal state from the
initial state according to domain rules.

Usage:
    python -m src.evaluate --domain blocksworld --provider openai --num_problems 20
    python -m src.evaluate --domain logistics   --provider openai --num_problems 20
"""
import argparse
import json
import time
from dataclasses import dataclass, field, asdict
from typing import List, Tuple

from src.aot_plus.core import AoTPlus
from src.domains.blocksworld import BlocksWorldDomain, BlocksWorldState
from src.domains.logistics import LogisticsDomain, LogisticsState
from src.utils.config import Config


# ---------------------------------------------------------------------------
# Problem banks
# ---------------------------------------------------------------------------

def blocksworld_problems() -> List[Tuple[str, str, List[str]]]:
    """
    Hand-crafted Blocksworld problems of increasing difficulty.
    Each entry is (initial_state_str, goal_state_str, optimal_solution).
    The optimal_solution is used for reference only; the evaluator runs
    the extracted plan through the simulator to check correctness.
    """
    return [
        # 3-block problems
        (
            "Block A is on the table, Block B is on Block A, Block C is on the table",
            "Block A is on Block B, Block B is on Block C, Block C is on the table",
            ["Move Block B from Block A to the table",
             "Move Block A from the table to Block C",
             "Move Block B from the table to Block A"],
        ),
        (
            "Block A is on Block B, Block B is on Block C, Block C is on the table",
            "Block C is on Block B, Block B is on Block A, Block A is on the table",
            ["Move Block A from Block B to the table",
             "Move Block B from Block C to Block A",
             "Move Block C from the table to Block B"],
        ),
        (
            "Block A is on the table, Block B is on the table, Block C is on the table",
            "Block C is on Block B, Block B is on Block A, Block A is on the table",
            ["Move Block B from the table to Block A",
             "Move Block C from the table to Block B"],
        ),
        # 4-block problems
        (
            "Block A is on the table, Block B is on Block A, Block C is on Block B, Block D is on the table",
            "Block D is on Block C, Block C is on Block B, Block B is on Block A, Block A is on the table",
            ["Move Block C from Block B to the table",
             "Move Block B from Block A to the table",
             "Move Block B from the table to Block A",
             "Move Block C from the table to Block B",
             "Move Block D from the table to Block C"],
        ),
        (
            "Block A is on Block B, Block B is on Block C, Block C is on Block D, Block D is on the table",
            "Block D is on Block A, Block A is on the table, Block B is on the table, Block C is on the table",
            ["Move Block A from Block B to the table",
             "Move Block B from Block C to the table",
             "Move Block C from Block D to the table",
             "Move Block D from the table to Block A"],
        ),
    ]


def logistics_problems() -> List[Tuple[str, str, List[str]]]:
    """
    Hand-crafted Logistics problems of increasing difficulty.
    Solution lists are reference answers; correctness is verified via simulator.
    """
    return [
        # Single package, single city
        (
            "Package P1 is at Location1, Truck T1 is at Location2, Airplane A1 is at Airport1",
            "Package P1 is at Location2",
            ["Drive Truck T1 from Location2 to Location1",
             "Load Package P1 onto Truck T1",
             "Drive Truck T1 from Location1 to Location2",
             "Unload Package P1 from Truck T1 at Location2"],
        ),
        # Two packages, same city
        (
            "Package P1 is at Location1, Package P2 is at Location2, "
            "Truck T1 is at Location2, Airplane A1 is at Airport1",
            "Package P1 is at Location2, Package P2 is at Location1",
            ["Drive Truck T1 from Location2 to Location1",
             "Load Package P1 onto Truck T1",
             "Drive Truck T1 from Location1 to Location2",
             "Unload Package P1 from Truck T1 at Location2",
             "Load Package P2 onto Truck T1",
             "Drive Truck T1 from Location2 to Location1",
             "Unload Package P2 from Truck T1 at Location1"],
        ),
        # Cross-city delivery (requires airplane)
        (
            "Package P1 is at Location1, Truck T1 is at Location2, "
            "Truck T2 is at Location4, Airplane A1 is at Airport1",
            "Package P1 is at Location4",
            ["Drive Truck T1 from Location2 to Location1",
             "Load Package P1 onto Truck T1",
             "Drive Truck T1 from Location1 to Airport1",
             "Unload Package P1 from Truck T1 at Airport1",
             "Load Package P1 onto Airplane A1",
             "Fly Airplane A1 from Airport1 to Airport2",
             "Unload Package P1 from Airplane A1 at Airport2",
             "Drive Truck T2 from Location4 to Airport2",
             "Load Package P1 onto Truck T2",
             "Drive Truck T2 from Airport2 to Location4",
             "Unload Package P1 from Truck T2 at Location4"],
        ),
        # The full example from the paper (two packages, cross-city)
        (
            "Package P1 is at Location1, Package P2 is at Location3, "
            "Truck T1 is at Location2, Truck T2 is at Location4, Airplane A1 is at Airport1",
            "Package P1 is at Location4, Package P2 is at Location2",
            [],   # reference solution omitted for brevity — evaluator uses simulator
        ),
    ]


# ---------------------------------------------------------------------------
# Simulation-based verifiers
# ---------------------------------------------------------------------------

def verify_blocksworld_plan(initial_str: str, goal_str: str, steps: List[str]) -> bool:
    """
    Run extracted plan steps through the BlocksWorld simulator.
    Returns True if goal is reached without any invalid moves.
    """
    import re
    state = BlocksWorldDomain.parse_state_description(initial_str)
    goal  = BlocksWorldDomain.parse_state_description(goal_str)

    for step in steps:
        step = step.strip()
        # Pattern: "Move Block X from ... to (the table | Block Y)"
        m = re.match(
            r"[Mm]ove [Bb]lock (\w+) from .+ to (?:the table|[Bb]lock (\w+))",
            step,
        )
        if not m:
            return False  # unparseable action → invalid plan
        block = m.group(1)
        dest  = m.group(2) if m.group(2) else "table"
        if not state.move(block, dest):
            return False  # illegal move

    return state.is_goal_state(goal)


def verify_logistics_plan(initial_str: str, goal_str: str, steps: List[str]) -> bool:
    """
    Run extracted plan steps through the Logistics simulator.
    Returns True if all goal packages are delivered correctly.
    """
    import re
    state = LogisticsDomain.parse_state_description(initial_str)
    goal  = LogisticsDomain.parse_state_description(goal_str)

    for step in steps:
        step = step.strip()
        ok = False

        m = re.match(r"[Ll]oad [Pp]ackage (\w+) onto (?:[Tt]ruck|[Aa]irplane) (\w+)", step)
        if m:
            ok = state.load_package(m.group(1), m.group(2))

        if not ok:
            m = re.match(r"[Uu]nload [Pp]ackage (\w+) from (?:[Tt]ruck|[Aa]irplane) \w+ at (\w+)", step)
            if m:
                ok = state.unload_package(m.group(1), m.group(2))

        if not ok:
            m = re.match(r"[Dd]rive [Tt]ruck (\w+) from \w+ to (\w+)", step)
            if m:
                ok = state.drive_vehicle(m.group(1), m.group(2))

        if not ok:
            m = re.match(r"[Ff]ly [Aa]irplane (\w+) from \w+ to (\w+)", step)
            if m:
                ok = state.fly_airplane(m.group(1), m.group(2))

        if not ok:
            return False  # illegal or unparseable action

    return state.is_goal_state(goal)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ProblemResult:
    problem_idx: int
    initial_state: str
    goal_state: str
    raw_output: str
    extracted_steps: List[str]
    success: bool
    error: str = ""


@dataclass
class EvalSummary:
    domain: str
    model: str
    provider: str
    num_problems: int
    num_success: int
    success_rate: float
    results: List[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(domain: str, provider: str, num_problems: int,
                   temperature: float, max_tokens: int) -> EvalSummary:
    if domain == "blocksworld":
        domain_description = BlocksWorldDomain.get_domain_description()
        few_shot = BlocksWorldDomain.generate_example_problem()
        problems  = blocksworld_problems()[:num_problems]
        verifier  = verify_blocksworld_plan
    else:
        domain_description = LogisticsDomain.get_domain_description()
        few_shot = LogisticsDomain.generate_example_problem()
        problems  = logistics_problems()[:num_problems]
        verifier  = verify_logistics_plan

    init_ex, goal_ex, sol_ex, rand_ex = few_shot

    aot = AoTPlus(domain_description, provider)
    aot.add_example(init_ex, goal_ex, sol_ex, rand_ex)

    model_name = Config.OPENAI_MODEL_NAME or "gpt-4o-mini"
    results: List[ProblemResult] = []

    for idx, (init_str, goal_str, _ref) in enumerate(problems):
        print(f"  [{idx+1}/{len(problems)}] Solving problem …", flush=True)
        error = ""
        try:
            raw = aot.solve(init_str, goal_str, temperature=temperature, max_tokens=max_tokens)
            steps = aot.extract_plan(raw)
            success = verifier(init_str, goal_str, steps)
        except Exception as e:
            raw = ""
            steps = []
            success = False
            error = str(e)

        results.append(ProblemResult(
            problem_idx=idx,
            initial_state=init_str,
            goal_state=goal_str,
            raw_output=raw,
            extracted_steps=steps,
            success=success,
            error=error,
        ))
        print(f"    success={success}  steps={len(steps)}")

        # Small delay to avoid rate-limit spikes
        if idx < len(problems) - 1:
            time.sleep(1)

    num_success = sum(r.success for r in results)
    summary = EvalSummary(
        domain=domain,
        model=model_name,
        provider=provider,
        num_problems=len(results),
        num_success=num_success,
        success_rate=num_success / len(results) if results else 0.0,
        results=[asdict(r) for r in results],
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate AoT+ on planning benchmarks")
    parser.add_argument("--domain",   choices=["blocksworld", "logistics"], default="blocksworld")
    parser.add_argument("--provider", choices=["openai", "azure_openai", "anthropic", "google"],
                        default="openai")
    parser.add_argument("--num_problems", type=int, default=5,
                        help="How many problems to evaluate (max depends on problem bank size)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=3000)
    parser.add_argument("--output",   type=str, default=None,
                        help="Optional JSON file to save results")
    args = parser.parse_args()

    available = Config.get_available_providers()
    if args.provider not in available:
        print(f"Provider '{args.provider}' not configured. Available: {available}")
        return

    print(f"\n=== AoT+ Evaluation ===")
    print(f"  Domain   : {args.domain}")
    print(f"  Provider : {args.provider}")
    print(f"  Model    : {Config.OPENAI_MODEL_NAME or 'gpt-4o-mini'}")
    print(f"  Problems : {args.num_problems}\n")

    summary = run_evaluation(
        domain=args.domain,
        provider=args.provider,
        num_problems=args.num_problems,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    print(f"\n=== Results ===")
    print(f"  Success rate: {summary.num_success}/{summary.num_problems} "
          f"= {summary.success_rate:.1%}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(asdict(summary), f, indent=2)
        print(f"  Saved to: {args.output}")


if __name__ == "__main__":
    main()
