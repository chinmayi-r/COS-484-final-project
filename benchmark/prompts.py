"""
AoT+ prompts for Blocksworld — following the structure described in
Section 5.1 and Appendix B of the AoT+ paper (Sel et al., 2025).

Key features vs plain CoT:
  1. Periodic State Memoization: full state is restated every few actions
  2. Random Trajectory Augmentation: few-shot examples include dead-end
     explorations that get explicitly abandoned before the correct path
  3. Output in PDDL action format inside a ```pddl block for reliable parsing
"""
from benchmark.pddl_parser import BlocksworldInstance, state_to_nl, goal_to_nl

# ── Domain system prompt ───────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
I am playing with a set of blocks where I need to arrange them into stacks on a table.
Here are the actions I can do:
  - Pick up a block from the table
  - Unstack a block from on top of another block
  - Put down a block on the table
  - Stack a block on top of another block

I have the following restrictions:
  - I can only pick up or unstack one block at a time.
  - I can only pick up or unstack a block if my hand is empty.
  - I can only pick up a block if the block is on the table and is clear (nothing on top).
  - I can only unstack a block from another block if it is directly on top of it and is clear.
  - I can only put down or stack a block I am currently holding.
  - I can only stack a block onto another block if that block is clear.
  - After putting down or stacking a block my hand becomes empty.
  - After picking up or unstacking a block the source block (or table spot) becomes clear.

For each problem I will think through the solution step by step, tracking the world state
explicitly after each action. If I reach a dead end I will backtrack. At the end I will
provide the final plan as PDDL actions inside a ```pddl block.\
"""

# ── Few-shot examples ──────────────────────────────────────────────────────────
# These follow the AoT+ paper structure: explicit state tracking + one failed
# exploration that gets abandoned (random trajectory augmentation).

# Example 1 — 3 blocks
# Initial: b2 on b1, b1 on table, b3 on table; hand empty; clear: b2, b3
# Goal:    b1 on b2, b2 on table, b3 on b1  (i.e. stack top-to-bottom: b3,b1,b2)
_EXAMPLE_1 = """\
[PROBLEM]
Initial state: b2 is on b1; b1 is on the table; b3 is on the table; hand is empty
Goal: b1 is on b2; b3 is on b1

[SOLVING]
I need: b3 on b1, b1 on b2, b2 on table.

STATE 0: b2 on b1 | b1 on table | b3 on table | hand empty | clear: b2, b3

Let me first try picking up b3 and putting it somewhere useful.
  pick-up b3  →  holding b3 | b1 on table | b2 on b1 | clear: b2
  — but to stack b3 on b1, b1 must be clear — b2 is on b1, so b1 is NOT clear.
  Dead end. I must unstack b2 from b1 first. Backtrack to STATE 0.

STATE 0 (revisit): b2 on b1 | b1 on table | b3 on table | hand empty | clear: b2, b3

Correct approach — first free b1 by removing b2:
  Action 1: unstack b2 b1
STATE 1: b2 in hand | b1 on table | b3 on table | clear: b1, b3

  Action 2: put-down b2
STATE 2: b1 on table | b2 on table | b3 on table | hand empty | clear: b1, b2, b3

Now build the tower from the bottom up (b2 → b1 → b3):
  Action 3: pick-up b1
STATE 3: b1 in hand | b2 on table | b3 on table | clear: b2, b3

  Action 4: stack b1 b2
STATE 4: b1 on b2 | b2 on table | b3 on table | hand empty | clear: b1, b3

  Action 5: pick-up b3
STATE 5: b3 in hand | b1 on b2 | b2 on table | clear: b1

  Action 6: stack b3 b1
STATE 6: b3 on b1 | b1 on b2 | b2 on table | hand empty | clear: b3  ✓ GOAL REACHED

```pddl
(unstack b2 b1)
(put-down b2)
(pick-up b1)
(stack b1 b2)
(pick-up b3)
(stack b3 b1)
```\
"""

# Example 2 — 4 blocks
# Initial: b1 on table (clear), b2 on table (clear), b3 on b4 (clear), b4 on table; hand empty
# Goal:    b2 on b4, b1 on b2, b3 on b1
_EXAMPLE_2 = """\
[PROBLEM]
Initial state: b1 is on the table; b2 is on the table; b3 is on b4; b4 is on the table; hand is empty
Goal: b2 is on b4; b1 is on b2; b3 is on b1

[SOLVING]
Target tower (bottom to top): b4 → b2 → b1 → b3.

STATE 0: b1 on table | b2 on table | b3 on b4 | b4 on table | hand empty | clear: b1, b2, b3

First, I need b4 clear so I can stack b2 on it. b3 is on b4, so unstack b3:
  Action 1: unstack b3 b4
STATE 1: b3 in hand | b1 on table | b2 on table | b4 on table | clear: b1, b2, b4

  Action 2: put-down b3
STATE 2: b1 on table | b2 on table | b3 on table | b4 on table | hand empty | clear: b1, b2, b3, b4

Now build the tower bottom-up:
  Action 3: pick-up b2
STATE 3: b2 in hand | b1 on table | b3 on table | b4 on table | clear: b1, b3, b4

  Action 4: stack b2 b4
STATE 4: b2 on b4 | b1 on table | b3 on table | b4 on table | hand empty | clear: b1, b2, b3

  Action 5: pick-up b1
STATE 5: b1 in hand | b2 on b4 | b3 on table | b4 on table | clear: b2, b3

  Action 6: stack b1 b2
STATE 6: b1 on b2 | b2 on b4 | b3 on table | b4 on table | hand empty | clear: b1, b3

  Action 7: pick-up b3
STATE 7: b3 in hand | b1 on b2 | b2 on b4 | b4 on table | clear: b1

  Action 8: stack b3 b1
STATE 8: b3 on b1 | b1 on b2 | b2 on b4 | b4 on table | hand empty | clear: b3  ✓ GOAL REACHED

```pddl
(unstack b3 b4)
(put-down b3)
(pick-up b2)
(stack b2 b4)
(pick-up b1)
(stack b1 b2)
(pick-up b3)
(stack b3 b1)
```\
"""

FEW_SHOT_EXAMPLES = [_EXAMPLE_1, _EXAMPLE_2]


# ── Prompt builder ─────────────────────────────────────────────────────────────

def build_aot_plus_prompt(instance: BlocksworldInstance) -> str:
    """Build the complete AoT+ prompt for a Blocksworld instance."""
    init_nl = state_to_nl(instance)
    goal_nl = goal_to_nl(instance)

    examples_text = "\n\n---\n\n".join(FEW_SHOT_EXAMPLES)

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Here are example problems with solutions:\n\n"
        f"{examples_text}\n\n"
        f"---\n\n"
        f"[PROBLEM]\n"
        f"Initial state: {init_nl}\n"
        f"Goal: {goal_nl}\n\n"
        f"[SOLVING]\n"
        f"STATE 0: {init_nl}\n"
    )
