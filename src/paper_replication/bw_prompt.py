"""Exact system prompt and verbatim few-shot examples for Blocksworld (Appendix B.3).

Source: "Algorithm of Thoughts" paper, Appendix B.3.
Do NOT modify the string literals below — they must match the paper verbatim.
"""
from __future__ import annotations
from typing import Dict, List

# ---------------------------------------------------------------------------
# Verbatim system prompt (Appendix B.3)
# ---------------------------------------------------------------------------

BW_SYSTEM_PROMPT = (
    "I am playing with a set of blocks where I need to arrange the blocks into "
    "stacks. Here are the actions I can do "
    "Pick up a block "
    "Unstack a block from on top of another block "
    "Put down a block "
    "Stack a block on top of another block "
    "I have the following restrictions on my actions: "
    "I can only pick up or unstack one block at a time. "
    "I can only pick up or unstack a block if my hand is empty. "
    "I can only pick up a block if the block is on the table and the block is clear. "
    "A block is clear if the block has no other blocks on top of it and if the block is not picked up. "
    "I can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block. "
    "I can only unstack a block from on top of another block if the block I am unstacking is clear. "
    "Once I pick up or unstack a block , I am holding the block. "
    "I can only put down a block that I am holding. "
    "I can only stack a block on top of another block if I am holding the block being stacked. "
    "I can only stack a block on top of another block if the block onto which I am stacking the block is clear. "
    "Once I put down or stack a block , my hand becomes empty. "
    "I cannot stack a block under another block. "
    "I can only stack a block on top of a block. "
    "Also, in the last part, do not use a numbered list or something to return the actions, "
    "just return them like in the examples. "
    "Use 'The PDDL version: bla bla' to return the PDDL code, "
    "and do not add anything after the actions."
)

# ---------------------------------------------------------------------------
# Verbatim few-shot examples (Appendix B.3)
# Each entry: {"user": <problem text>, "assistant": <full solution text>}
# ---------------------------------------------------------------------------

_EX1_USER = "Start: D on A on C on B on table.\nGoal: A on D on B."

_EX1_ASST = """\
Firstly, we should satisfy D being on B without B being on A. Since B is \
on the table, we should remove every block on top of B to later stack \
first D then A on top of B.

1. Unstack D from A.
2. Put-down D
- D on table, A on C on B on table.
3. Unstack A from C
4. Put-down A
- A on table, D on table, C on B on table.
5. Unstack C from B
6. Put-down C
- A on table, B on table, C on table, D on table.

Now everything is on the table and we need to simply one by one stack \
the blocks in sequence. So, first D on B, then A on D.

7. Pick-up D
8. Stack D on B
- A on table, C on table, D on B on table.
9. Pick-up A
10. Stack A on D

The PDDL version:
(unstack D A)
(put-down D)
(unstack A C)
(put-down A)
(unstack C B)
(put-down C)
(pick-up D)
(stack D B)
(pick-up A)
(stack A D)"""

_EX2_USER = "Start: A on table, B on C on table, D on table.\nGoal: A on B on D on C."

_EX2_ASST = """\
For the goal state, all blocks need to be stacked on top of each other \
with C being on the table (because there are only four blocks and all \
four blocks are on top of each other). Then in sequence, first D, then \
B, then A needs to be stacked.

1. Pick-up D
2. Stack D on C. C is not clear. Backtracking...

1. Unstack B from C
2. Put-down B
- A on table, B on table, C on table, D on table.

Now that all blocks are on the table, we just have to follow the initial \
plan of ours.

3. Pick-up D
4. Stack D on C
- A on table, B on table, D on C on table.
5. Pick-up B
6. Stack B on D
- A on table, B on D on C on table.
7. Pick-up A
8. Stack A on B.
- A on B on D on C on table.

The PDDL version:
(unstack B C)
(put-down B)
(pick-up D)
(stack D C)
(pick-up B)
(stack B D)
(pick-up A)
(stack A B)"""

_EX3_USER = "Start: B on C on A on table, D on table.\nGoal: B on D, C on A."

_EX3_ASST = """\
B is clear (clear means no block in on top of it) and it only needs to \
be put on D, and D is clear also. Then, C should be on A, and it's \
satisfied currently. Let's start with placing B on D and let's examine \
again then.

1. Unstack B from C
2. Stack B on D
- B on D on table, C on A on table.

Goal is satisfied.

The PDDL version:
(unstack B C)
(stack B D)"""

_EX4_USER = "Start: B on table, C on A on D on table.\nGoal: B on A on D on C."

_EX4_ASST = """\
For the goal state, all blocks need to be stacked on top of each other \
with C being on the table (because there are only four blocks and all \
four blocks are on top of each other). Then in sequence, first D, then \
A, then B needs to be stacked.

1. Unstack A from D. However, A is not clear, there C on top of it. \
Backtracking...

1. Unstack C from A
2. Put-down C
- B on table, C on table, A on D on table.

Now that C is on the table, we can unstack A from D.

3. Unstack A from D
4. Put-down A
- A on table, B on table, C on table, D on table.

Now that all blocks are on the table, we just have to follow the initial \
plan of ours.

5. Pick-up D
6. Stack D on C
- A on table, B on table, D on C on table.
7. Pick-up A
8. Stack A on D
- A on D on C on table, B on table.
9. Pick-up B
10. Stack B on A.
- B on A on D on C on table.

The PDDL version:
(unstack C A)
(put-down C)
(unstack A D)
(put-down A)
(pick-up D)
(stack D C)
(pick-up A)
(stack A D)
(pick-up B)
(stack B A)"""

_EX5_USER = "Start: D on A on B on C on table.\nGoal: D on B, A on C."

_EX5_ASST = """\
D is already on B, but B is not clear. A is on C, but C is not clear. \
Let's start with making B clear.

1. Unstack D from A
2. Put-down D
- D on table, A on B on C on table.
3. Unstack A from B
4. Put-down A
- A on table, B on C on table, D on table.

Now, we can place D on B to make it D on B on C, which would satisfy \
one of the goals (D on B). A on C is not satisfied since B is on C. \
Then B needs to also be unstacked.

5. Unstack B from C
6. Put-down B
- A on table, B on table, C on table, D on table.

Now that all blocks are on the table, we just have to follow the initial \
plan of ours.

7. Pick-up D
8. Stack D on B
- A on table, D on B on table, C on table.
9. Pick-up A
10. Stack A on C.
- A on C on table, D on B on table.

The PDDL version:
(unstack D A)
(put-down D)
(unstack A B)
(put-down A)
(unstack B C)
(put-down B)
(pick-up D)
(stack D B)
(pick-up A)
(stack A C)"""

_EX6_USER = "Start: D on B on C on A on table.\nGoal: D on C on A on B."

_EX6_ASST = """\
For the goal state, all blocks need to be stacked on top of each other \
with B being on the table (because there are only four blocks and all \
four blocks are on top of each other). Then in sequence, first A, then \
C, then D needs to be stacked.

1. Unstack D from B
2. Put-down D
- D on table, B on C on A on table.
3. Unstack B from C
4. Put-down B
- B on table, C on A on table, D on table.
5. Unstack C from A
6. Put-down C
- A on table, B on table, C on table, D on table.

Now that all blocks are on the table, we just have to follow the initial \
plan of ours.

7. Pick-up A
8. Stack A on B
- A on B on table, C on table, D on table.
9. Pick-up C
10. Stack C on A
- C on A on B on table, D on table.
11. Pick-up D
12. Stack D on C.
- D on C on A on B on table.

The PDDL version:
(unstack D B)
(put-down D)
(unstack B C)
(put-down B)
(unstack C A)
(put-down C)
(pick-up A)
(stack A B)
(pick-up C)
(stack C A)
(pick-up D)
(stack D C)"""

BW_FEW_SHOT_EXAMPLES: List[Dict[str, str]] = [
    {"user": _EX1_USER, "assistant": _EX1_ASST},
    {"user": _EX2_USER, "assistant": _EX2_ASST},
    {"user": _EX3_USER, "assistant": _EX3_ASST},
    {"user": _EX4_USER, "assistant": _EX4_ASST},
    {"user": _EX5_USER, "assistant": _EX5_ASST},
    {"user": _EX6_USER, "assistant": _EX6_ASST},
]


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_bw_messages(init_compact: str, goal_compact: str) -> List[Dict[str, str]]:
    """Build the full messages list for a blocksworld problem.

    Structure: system → [user/assistant few-shot pairs] → user (new problem).
    """
    messages: List[Dict[str, str]] = [{"role": "system", "content": BW_SYSTEM_PROMPT}]
    for ex in BW_FEW_SHOT_EXAMPLES:
        messages.append({"role": "user",      "content": ex["user"]})
        messages.append({"role": "assistant", "content": ex["assistant"]})
    messages.append({"role": "user", "content": f"Start: {init_compact}\nGoal: {goal_compact}"})
    return messages
