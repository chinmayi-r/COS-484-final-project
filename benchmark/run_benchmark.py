"""
Benchmark runner for AoT+ on PlanBench Blocksworld instances.

Usage:
    python benchmark/run_benchmark.py \\
        --instances /path/to/LLMs-Planning/llm_planning_analysis/instances/blocksworld/task_1_plan_generation \\
        --model gpt-4o-mini \\
        [--max-instances 30] \\
        [--output results.json]

The instances directory is from the karthikv792/LLMs-Planning repo:
    git clone https://github.com/karthikv792/LLMs-Planning.git

Difficulty tiers (matching AoT+ paper hyphenated scores):
    easy   — 2-4 blocks
    medium — 5 blocks
    hard   — 6+ blocks

(Actual tier boundaries depend on the dataset; adjust --easy-max / --hard-min
 flags if the PlanBench distribution differs from the defaults above.)
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

# Make sure project root is on path when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.pddl_parser import BlocksworldInstance, load_instances
from benchmark.prompts import build_aot_plus_prompt
from benchmark.validator import validate, extract_plan_from_text


# ── LLM call ──────────────────────────────────────────────────────────────────

def call_openai(prompt: str, model: str, api_key: str,
                temperature: float = 0.0, max_tokens: int = 2048) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a planning assistant. Follow the AoT+ format shown in the examples exactly."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


# ── Difficulty classification ─────────────────────────────────────────────────

def classify_difficulty(instance: BlocksworldInstance,
                         easy_max: int, hard_min: int) -> str:
    n = instance.num_blocks
    if n <= easy_max:
        return "easy"
    elif n < hard_min:
        return "medium"
    else:
        return "hard"


# ── Main ──────────────────────────────────────────────────────────────────────

def run_benchmark(
    instances_dir: Path,
    model: str,
    api_key: str,
    max_instances: Optional[int],
    easy_max: int,
    hard_min: int,
    output_path: Optional[Path],
    verbose: bool,
    delay: float,
) -> None:
    instances = load_instances(instances_dir)
    if not instances:
        print(f"No PDDL files found in {instances_dir}")
        sys.exit(1)

    if max_instances:
        instances = instances[:max_instances]

    print(f"Loaded {len(instances)} instances from {instances_dir}")
    print(f"Model: {model}   easy≤{easy_max} blocks   hard≥{hard_min} blocks\n")

    tiers = ["easy", "medium", "hard"]
    correct: Dict[str, int] = defaultdict(int)
    total: Dict[str, int] = defaultdict(int)
    records = []

    for i, inst in enumerate(instances):
        tier = classify_difficulty(inst, easy_max, hard_min)
        total[tier] += 1

        prompt = build_aot_plus_prompt(inst)

        if verbose:
            print(f"[{i+1}/{len(instances)}] {inst.name} ({inst.num_blocks} blocks, {tier})")

        try:
            response = call_openai(prompt, model, api_key)
        except Exception as e:
            print(f"  API error: {e}")
            response = ""

        is_valid, reason = validate(inst, response)
        if is_valid:
            correct[tier] += 1

        record = {
            "instance": inst.name,
            "num_blocks": inst.num_blocks,
            "tier": tier,
            "valid": is_valid,
            "reason": reason,
            "response": response if verbose else None,
        }
        records.append(record)

        if verbose:
            status = "✓" if is_valid else "✗"
            print(f"  {status}  {reason}")

        if delay > 0 and i < len(instances) - 1:
            time.sleep(delay)

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"AoT+ Blocksworld results — {model}")
    print("=" * 55)
    tier_pct = []
    for tier in tiers:
        n = total[tier]
        c = correct[tier]
        pct = round(100 * c / n) if n > 0 else "-"
        tier_pct.append(str(pct) if n > 0 else "-")
        print(f"  {tier.capitalize():8s} ({n:3d} instances): {c}/{n} = {pct}%")
    overall_total = sum(total.values())
    overall_correct = sum(correct.values())
    overall_pct = round(100 * overall_correct / overall_total) if overall_total else 0
    print(f"  {'Overall':8s} ({overall_total:3d} instances): {overall_correct}/{overall_total} = {overall_pct}%")
    print(f"\n  Hyphenated score: {'-'.join(tier_pct)}")
    print("=" * 55)

    if output_path:
        result = {
            "model": model,
            "instances_dir": str(instances_dir),
            "easy_max_blocks": easy_max,
            "hard_min_blocks": hard_min,
            "summary": {t: {"correct": correct[t], "total": total[t]} for t in tiers},
            "hyphenated_score": "-".join(tier_pct),
            "records": records,
        }
        output_path.write_text(json.dumps(result, indent=2))
        print(f"\nFull results written to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AoT+ Blocksworld benchmark")
    parser.add_argument(
        "--instances", required=True, type=Path,
        help="Path to PlanBench task_1_plan_generation directory with .pddl files",
    )
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="OpenAI model name (default: gpt-4o-mini)")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"),
                        help="OpenAI API key (default: $OPENAI_API_KEY)")
    parser.add_argument("--max-instances", type=int, default=None,
                        help="Cap number of instances (useful for quick testing)")
    parser.add_argument("--easy-max", type=int, default=4,
                        help="Max block count for 'easy' tier (default: 4)")
    parser.add_argument("--hard-min", type=int, default=6,
                        help="Min block count for 'hard' tier (default: 6)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Write JSON results to this file")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-instance results")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Seconds between API calls (default: 0.5)")
    args = parser.parse_args()

    if not args.api_key:
        print("Error: provide --api-key or set OPENAI_API_KEY")
        sys.exit(1)

    if not args.instances.exists():
        print(f"Error: instances directory not found: {args.instances}")
        print("\nClone the PlanBench repo first:")
        print("  git clone https://github.com/karthikv792/LLMs-Planning.git")
        print("  Then point --instances at:")
        print("  LLMs-Planning/llm_planning_analysis/instances/blocksworld/task_1_plan_generation")
        sys.exit(1)

    run_benchmark(
        instances_dir=args.instances,
        model=args.model,
        api_key=args.api_key,
        max_instances=args.max_instances,
        easy_max=args.easy_max,
        hard_min=args.hard_min,
        output_path=args.output,
        verbose=args.verbose,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
