"""
Few-shot Blocksworld examples using 4-op PDDL actions for AoT+ JSON extension.

Operators: pick-up, put-down, unstack, stack — identical to the AoT+ paper.
State format: {"on": {block: "table"|"held"|block}, "clear": [...], "holding": null|block, "handempty": bool}

Checkpoint interval: STATE_JSON every 3 actions + always after the final action.

Condition B: plain JSON state.
Condition C: JSON state + "goal_satisfied" / "goal_remaining" fields.
"""

# ---------------------------------------------------------------------------
# Condition B — 4-op actions, JSON state, no goal tracking
#
# Example 1: 3 blocks
#   Initial: on={A:table, B:A, C:table}  clear=[B,C]
#   Goal:    on={A:B, B:C, C:table}      clear=[A]
#
# 10-action trace with one wrong detour (pick-up C → stack C B → undo):
#   1. (pick-up C)          wrong direction
#   2. (stack C B)
#   3. (unstack C B)        undo  [CHECKPOINT]
#   4. (put-down C)
#   5. (unstack B A)
#   6. (put-down B)         [CHECKPOINT]
#   7. (pick-up B)
#   8. (stack B C)
#   9. (pick-up A)          [CHECKPOINT]
#  10. (stack A B)          [CHECKPOINT — final, GOAL]
# ---------------------------------------------------------------------------

BLOCKSWORLD_4OP_JSON_EXAMPLE_1 = """\
Initial state (JSON):
{"on": {"A": "table", "B": "A", "C": "table"}, "clear": ["B", "C"], "holding": null, "handempty": true}

Goal state (JSON):
{"on": {"A": "B", "B": "C", "C": "table"}, "clear": ["A"], "holding": null, "handempty": true}

STATE_JSON: ```json
{"on": {"A": "table", "B": "A", "C": "table"}, "clear": ["B", "C"], "holding": null, "handempty": true}
```

I'll solve this step by step.

Let me first try placing C on B.

ACTION: (pick-up C)
ACTION: (stack C B)

That leaves A blocked and doesn't progress toward the goal. Let me undo.

ACTION: (unstack C B)
STATE_JSON: ```json
{"on": {"A": "table", "B": "A", "C": "held"}, "clear": ["B"], "holding": "C", "handempty": false}
```

Good. Put C back down, then unstack B from A first.

ACTION: (put-down C)
ACTION: (unstack B A)
ACTION: (put-down B)
STATE_JSON: ```json
{"on": {"A": "table", "B": "table", "C": "table"}, "clear": ["A", "B", "C"], "holding": null, "handempty": true}
```

All three blocks are on the table. Build the goal stack bottom-up: B on C, then A on B.

ACTION: (pick-up B)
ACTION: (stack B C)
ACTION: (pick-up A)
STATE_JSON: ```json
{"on": {"A": "held", "B": "C", "C": "table"}, "clear": ["B"], "holding": "A", "handempty": false}
```

ACTION: (stack A B)
STATE_JSON: ```json
{"on": {"A": "B", "B": "C", "C": "table"}, "clear": ["A"], "holding": null, "handempty": true}
```

Goal reached: A on B, B on C, C on table.

Plan summary:
1. (unstack B A)
2. (put-down B)
3. (pick-up B)
4. (stack B C)
5. (pick-up A)
6. (stack A B)\
"""

# ---------------------------------------------------------------------------
# Condition B — Example 2: 4 blocks, clean reverse-stack (no backtracking)
#   Initial: on={A:B, B:C, C:D, D:table}  clear=[A]
#   Goal:    on={A:table, B:A, C:B, D:C}  clear=[D]
#
# 12-action trace:
#   1. (unstack A B)
#   2. (put-down A)
#   3. (unstack B C)   [CHECKPOINT]
#   4. (put-down B)
#   5. (unstack C D)
#   6. (put-down C)    [CHECKPOINT]
#   7. (pick-up B)
#   8. (stack B A)
#   9. (pick-up C)     [CHECKPOINT]
#  10. (stack C B)
#  11. (pick-up D)
#  12. (stack D C)     [CHECKPOINT — final, GOAL]
# ---------------------------------------------------------------------------

BLOCKSWORLD_4OP_JSON_EXAMPLE_2 = """\
Initial state (JSON):
{"on": {"A": "B", "B": "C", "C": "D", "D": "table"}, "clear": ["A"], "holding": null, "handempty": true}

Goal state (JSON):
{"on": {"A": "table", "B": "A", "C": "B", "D": "C"}, "clear": ["D"], "holding": null, "handempty": true}

STATE_JSON: ```json
{"on": {"A": "B", "B": "C", "C": "D", "D": "table"}, "clear": ["A"], "holding": null, "handempty": true}
```

I need to reverse the stack. Unstack everything to the table, then rebuild bottom-up.

ACTION: (unstack A B)
ACTION: (put-down A)
ACTION: (unstack B C)
STATE_JSON: ```json
{"on": {"A": "table", "B": "held", "C": "D", "D": "table"}, "clear": ["A", "C"], "holding": "B", "handempty": false}
```

ACTION: (put-down B)
ACTION: (unstack C D)
ACTION: (put-down C)
STATE_JSON: ```json
{"on": {"A": "table", "B": "table", "C": "table", "D": "table"}, "clear": ["A", "B", "C", "D"], "holding": null, "handempty": true}
```

All blocks on the table. Build the new stack: B on A, then C on B, then D on C.

ACTION: (pick-up B)
ACTION: (stack B A)
ACTION: (pick-up C)
STATE_JSON: ```json
{"on": {"A": "table", "B": "A", "C": "held", "D": "table"}, "clear": ["B", "D"], "holding": "C", "handempty": false}
```

ACTION: (stack C B)
ACTION: (pick-up D)
ACTION: (stack D C)
STATE_JSON: ```json
{"on": {"A": "table", "B": "A", "C": "B", "D": "C"}, "clear": ["D"], "holding": null, "handempty": true}
```

Goal reached: B on A, C on B, D on C, A on table.

Plan summary:
1. (unstack A B)
2. (put-down A)
3. (unstack B C)
4. (put-down B)
5. (unstack C D)
6. (put-down C)
7. (pick-up B)
8. (stack B A)
9. (pick-up C)
10. (stack C B)
11. (pick-up D)
12. (stack D C)\
"""


# ---------------------------------------------------------------------------
# Condition C — same examples, with goal_satisfied / goal_remaining
#
# Goal predicates for Example 1: A on B, B on C, C on table, A clear
#
# Checkpoint goal tracking:
#   S0:  goal_satisfied=["C on table"],             goal_remaining=["A on B","B on C","A clear"]
#   S3:  goal_satisfied=[],                          goal_remaining=["A on B","B on C","C on table","A clear"]
#   S6:  goal_satisfied=["C on table","A clear"],   goal_remaining=["A on B","B on C"]
#   S9:  goal_satisfied=["B on C","C on table"],    goal_remaining=["A on B","A clear"]
#   S10: goal_satisfied=[all],                      goal_remaining=[]
# ---------------------------------------------------------------------------

BLOCKSWORLD_4OP_JSON_GOAL_EXAMPLE_1 = """\
Initial state (JSON):
{"on": {"A": "table", "B": "A", "C": "table"}, "clear": ["B", "C"], "holding": null, "handempty": true, "goal_satisfied": ["C on table"], "goal_remaining": ["A on B", "B on C", "A clear"]}

Goal state (JSON):
{"on": {"A": "B", "B": "C", "C": "table"}, "clear": ["A"], "holding": null, "handempty": true}

STATE_JSON: ```json
{"on": {"A": "table", "B": "A", "C": "table"}, "clear": ["B", "C"], "holding": null, "handempty": true, "goal_satisfied": ["C on table"], "goal_remaining": ["A on B", "B on C", "A clear"]}
```

I'll solve this step by step.

Let me first try placing C on B.

ACTION: (pick-up C)
ACTION: (stack C B)

That loses the "C on table" condition — I moved backwards. Let me undo.

ACTION: (unstack C B)
STATE_JSON: ```json
{"on": {"A": "table", "B": "A", "C": "held"}, "clear": ["B"], "holding": "C", "handempty": false, "goal_satisfied": [], "goal_remaining": ["A on B", "B on C", "C on table", "A clear"]}
```

No goal conditions satisfied — I made things worse. Put C down, then unstack B from A.

ACTION: (put-down C)
ACTION: (unstack B A)
ACTION: (put-down B)
STATE_JSON: ```json
{"on": {"A": "table", "B": "table", "C": "table"}, "clear": ["A", "B", "C"], "holding": null, "handempty": true, "goal_satisfied": ["C on table", "A clear"], "goal_remaining": ["A on B", "B on C"]}
```

Two goal conditions satisfied now. Build: B on C, then A on B.

ACTION: (pick-up B)
ACTION: (stack B C)
ACTION: (pick-up A)
STATE_JSON: ```json
{"on": {"A": "held", "B": "C", "C": "table"}, "clear": ["B"], "holding": "A", "handempty": false, "goal_satisfied": ["B on C", "C on table"], "goal_remaining": ["A on B", "A clear"]}
```

One action left.

ACTION: (stack A B)
STATE_JSON: ```json
{"on": {"A": "B", "B": "C", "C": "table"}, "clear": ["A"], "holding": null, "handempty": true, "goal_satisfied": ["A on B", "B on C", "C on table", "A clear"], "goal_remaining": []}
```

All goal conditions satisfied.

Plan summary:
1. (unstack B A)
2. (put-down B)
3. (pick-up B)
4. (stack B C)
5. (pick-up A)
6. (stack A B)\
"""

# ---------------------------------------------------------------------------
# Condition C — Example 2: 4 blocks with goal tracking
# Goal predicates: A on table, B on A, C on B, D on C, D clear
#
#   S0:  goal_satisfied=[],                              goal_remaining=[all 5]
#   S3:  goal_satisfied=["A on table"],                  goal_remaining=[4 remaining]
#   S6:  goal_satisfied=["A on table","D clear"],        goal_remaining=[3 remaining]
#   S9:  goal_satisfied=["A on table","B on A","D clear"], goal_remaining=["C on B","D on C"]
#   S12: goal_satisfied=[all],                           goal_remaining=[]
# ---------------------------------------------------------------------------

BLOCKSWORLD_4OP_JSON_GOAL_EXAMPLE_2 = """\
Initial state (JSON):
{"on": {"A": "B", "B": "C", "C": "D", "D": "table"}, "clear": ["A"], "holding": null, "handempty": true, "goal_satisfied": [], "goal_remaining": ["A on table", "B on A", "C on B", "D on C", "D clear"]}

Goal state (JSON):
{"on": {"A": "table", "B": "A", "C": "B", "D": "C"}, "clear": ["D"], "holding": null, "handempty": true}

STATE_JSON: ```json
{"on": {"A": "B", "B": "C", "C": "D", "D": "table"}, "clear": ["A"], "holding": null, "handempty": true, "goal_satisfied": [], "goal_remaining": ["A on table", "B on A", "C on B", "D on C", "D clear"]}
```

I need to reverse the stack. Unstack everything to the table, then rebuild bottom-up.

ACTION: (unstack A B)
ACTION: (put-down A)
ACTION: (unstack B C)
STATE_JSON: ```json
{"on": {"A": "table", "B": "held", "C": "D", "D": "table"}, "clear": ["A", "C"], "holding": "B", "handempty": false, "goal_satisfied": ["A on table"], "goal_remaining": ["B on A", "C on B", "D on C", "D clear"]}
```

Good — A on table is satisfied. Continue unstacking.

ACTION: (put-down B)
ACTION: (unstack C D)
ACTION: (put-down C)
STATE_JSON: ```json
{"on": {"A": "table", "B": "table", "C": "table", "D": "table"}, "clear": ["A", "B", "C", "D"], "holding": null, "handempty": true, "goal_satisfied": ["A on table", "D clear"], "goal_remaining": ["B on A", "C on B", "D on C"]}
```

Two goal conditions satisfied. Now build: B on A, then C on B, then D on C.

ACTION: (pick-up B)
ACTION: (stack B A)
ACTION: (pick-up C)
STATE_JSON: ```json
{"on": {"A": "table", "B": "A", "C": "held", "D": "table"}, "clear": ["B", "D"], "holding": "C", "handempty": false, "goal_satisfied": ["A on table", "B on A", "D clear"], "goal_remaining": ["C on B", "D on C"]}
```

Three satisfied. Two actions left.

ACTION: (stack C B)
ACTION: (pick-up D)
ACTION: (stack D C)
STATE_JSON: ```json
{"on": {"A": "table", "B": "A", "C": "B", "D": "C"}, "clear": ["D"], "holding": null, "handempty": true, "goal_satisfied": ["A on table", "B on A", "C on B", "D on C", "D clear"], "goal_remaining": []}
```

All goal conditions satisfied.

Plan summary:
1. (unstack A B)
2. (put-down A)
3. (unstack B C)
4. (put-down B)
5. (unstack C D)
6. (put-down C)
7. (pick-up B)
8. (stack B A)
9. (pick-up C)
10. (stack C B)
11. (pick-up D)
12. (stack D C)\
"""


def get_blocksworld_json_examples() -> list:
    """Condition B: 4-op actions, JSON state, no goal tracking."""
    return [BLOCKSWORLD_4OP_JSON_EXAMPLE_1, BLOCKSWORLD_4OP_JSON_EXAMPLE_2]


def get_blocksworld_json_goal_examples() -> list:
    """Condition C: 4-op actions, JSON state + goal_satisfied/goal_remaining."""
    return [BLOCKSWORLD_4OP_JSON_GOAL_EXAMPLE_1, BLOCKSWORLD_4OP_JSON_GOAL_EXAMPLE_2]
