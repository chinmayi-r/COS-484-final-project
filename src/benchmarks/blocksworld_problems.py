"""Blocksworld benchmark problem generator.

Difficulty tiers (matching paper's 3-column accuracy format):
  easy   — optimal plan length 2–4  steps
  medium — optimal plan length 6–8  steps
  hard   — optimal plan length 10–12 steps
"""

import random
from collections import deque
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Minimal state representation (hashable, for BFS)
# ---------------------------------------------------------------------------

def _state_to_key(on: Dict[str, str]) -> tuple:
    return tuple(sorted(on.items()))


def _get_clear(on: Dict[str, str], blocks: List[str]) -> List[str]:
    """Blocks with nothing on top."""
    tops = set(on.values())
    return [b for b in blocks if b not in tops]


def _apply_move(on: Dict[str, str], block: str, dest: str, blocks: List[str]) -> Optional[Dict[str, str]]:
    clear = set(_get_clear(on, blocks))
    if block not in clear:
        return None
    if dest != "table" and dest not in clear:
        return None
    if dest == block:
        return None
    new_on = dict(on)
    new_on[block] = dest
    return new_on


def _bfs_plan(
    initial: Dict[str, str],
    goal: Dict[str, str],
    blocks: List[str],
    max_depth: int = 20,
) -> Optional[List[Tuple[str, str]]]:
    """BFS that returns the shortest plan as list of (block, destination) moves."""
    start_key = _state_to_key(initial)
    goal_key = _state_to_key(goal)
    if start_key == goal_key:
        return []
    queue: deque = deque([(initial, [])])
    visited = {start_key}
    while queue:
        state, path = queue.popleft()
        if len(path) >= max_depth:
            continue
        for block in blocks:
            dests = ["table"] + blocks
            for dest in dests:
                new_state = _apply_move(state, block, dest, blocks)
                if new_state is None:
                    continue
                new_key = _state_to_key(new_state)
                if new_key == goal_key:
                    return path + [(block, dest)]
                if new_key not in visited:
                    visited.add(new_key)
                    queue.append((new_state, path + [(block, dest)]))
    return None


# ---------------------------------------------------------------------------
# Problem generation
# ---------------------------------------------------------------------------

def _random_state(blocks: List[str], rng: random.Random) -> Dict[str, str]:
    """Generate a random legal Blocksworld state."""
    on: Dict[str, str] = {}
    placed: List[str] = []
    for block in rng.sample(blocks, len(blocks)):
        if placed and rng.random() < 0.5:
            on[block] = rng.choice(placed)
        else:
            on[block] = "table"
        placed.append(block)
    return on


def _state_to_nl(on: Dict[str, str], blocks: List[str]) -> str:
    parts = []
    for b in sorted(blocks):
        parts.append(f"Block {b} is on {on[b]}")
    return ", ".join(parts)


def _plan_to_nl(plan: List[Tuple[str, str]]) -> List[str]:
    steps = []
    for block, dest in plan:
        if dest == "table":
            steps.append(f"Move Block {block} to the table")
        else:
            steps.append(f"Move Block {block} on top of Block {dest}")
    return steps


def generate_blocksworld_problems(
    tier: str,
    n_problems: int = 20,
    seed: int = 42,
) -> List[dict]:
    """Generate Blocksworld problems for a difficulty tier.

    Args:
        tier: "easy" (2–4 steps), "medium" (6–8 steps), or "hard" (10–12 steps)
        n_problems: how many problems to generate
        seed: random seed for reproducibility

    Returns:
        List of dicts with keys:
          initial_state (str), goal_state (str),
          optimal_plan (list[str]), optimal_length (int),
          n_blocks (int), tier (str)
    """
    tier_ranges = {
        "easy":   (2, 4,  4, 5),   # (min_steps, max_steps, min_blocks, max_blocks)
        "medium": (6, 8,  5, 7),
        "hard":   (10, 12, 7, 9),
    }
    if tier not in tier_ranges:
        raise ValueError(f"tier must be 'easy', 'medium', or 'hard', got {tier!r}")

    min_steps, max_steps, min_blocks, max_blocks = tier_ranges[tier]
    rng = random.Random(seed)
    problems = []
    attempts = 0

    while len(problems) < n_problems and attempts < n_problems * 200:
        attempts += 1
        n_blocks = rng.randint(min_blocks, max_blocks)
        blocks = [chr(ord('A') + i) for i in range(n_blocks)]

        initial = _random_state(blocks, rng)
        goal = _random_state(blocks, rng)

        if _state_to_key(initial) == _state_to_key(goal):
            continue

        plan = _bfs_plan(initial, goal, blocks, max_depth=max_steps + 2)
        if plan is None:
            continue
        if not (min_steps <= len(plan) <= max_steps):
            continue

        problems.append({
            "initial_state": _state_to_nl(initial, blocks),
            "goal_state":    _state_to_nl(goal, blocks),
            "optimal_plan":  _plan_to_nl(plan),
            "optimal_length": len(plan),
            "n_blocks":      n_blocks,
            "tier":          tier,
            "blocks":        blocks,
            "initial_on":    initial,
            "goal_on":       goal,
        })

    if len(problems) < n_problems:
        raise RuntimeError(
            f"Could only generate {len(problems)}/{n_problems} problems for tier={tier!r}"
        )
    return problems


def get_all_blocksworld_problems(n_per_tier: int = 20, seed: int = 42) -> Dict[str, List[dict]]:
    """Return problems for all three tiers keyed by tier name."""
    return {
        "easy":   generate_blocksworld_problems("easy",   n_per_tier, seed),
        "medium": generate_blocksworld_problems("medium", n_per_tier, seed + 1),
        "hard":   generate_blocksworld_problems("hard",   n_per_tier, seed + 2),
    }
