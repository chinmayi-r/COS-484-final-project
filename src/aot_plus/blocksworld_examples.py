"""
Blocksworld few-shot examples for AoT+ JSON extension.

Action format: parenthesized 4-op with unambiguous keywords
  (pick-up X)  (put-down X)  (unstack X from Y)  (stack X onto Y)
State format: JSON (STATE_JSON blocks every 3 actions + after final action).

Condition B: plain JSON state.
Condition C: JSON state + goal_satisfied / goal_remaining fields.
"""

# ---------------------------------------------------------------------------
# Condition B — NL 4-op actions, JSON state, no goal tracking
#
# Example 1: 3 blocks
#   Initial: on={A:table, B:A, C:table}  clear=[B,C]
#   Goal:    on={A:B, B:C, C:table}      clear=[A]
#
# 10-action trace (wrong detour then correct plan):
#   1. (pick-up C)          wrong direction
#   2. (stack C onto B)
#   3. (unstack C from B)   undo  [CHECKPOINT]
#   4. (put-down C)
#   5. (unstack B from A)
#   6. (put-down B)         [CHECKPOINT]
#   7. (pick-up B)
#   8. (stack B onto C)
#   9. (pick-up A)          [CHECKPOINT]
#  10. (stack A onto B)        [CHECKPOINT — final, GOAL]
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
ACTION: (stack C onto B)

That leaves A blocked and doesn't progress toward the goal. Let me undo.

ACTION: (unstack C from B)
STATE_JSON: ```json
{"on": {"A": "table", "B": "A", "C": "held"}, "clear": ["B"], "holding": "C", "handempty": false}
```

Good. Put C back down, then unstack B from A first.

ACTION: (put-down C)
ACTION: (unstack B from A)
ACTION: (put-down B)
STATE_JSON: ```json
{"on": {"A": "table", "B": "table", "C": "table"}, "clear": ["A", "B", "C"], "holding": null, "handempty": true}
```

All three blocks are on the table. Build the goal stack bottom-up: B on C, then A on B.

ACTION: (pick-up B)
ACTION: (stack B onto C)
ACTION: (pick-up A)
STATE_JSON: ```json
{"on": {"A": "held", "B": "C", "C": "table"}, "clear": ["B"], "holding": "A", "handempty": false}
```

ACTION: (stack A onto B)
STATE_JSON: ```json
{"on": {"A": "B", "B": "C", "C": "table"}, "clear": ["A"], "holding": null, "handempty": true}
```

Goal reached: A on B, B on C, C on table.

Plan summary:
1. (unstack B from A)
2. (put-down B)
3. (pick-up B)
4. (stack B onto C)
5. (pick-up A)
6. (stack A onto B)\
"""

# ---------------------------------------------------------------------------
# Condition B — Example 2: 4 blocks, reverse stack (no backtracking)
#   Initial: on={A:B, B:C, C:D, D:table}  clear=[A]
#   Goal:    on={A:table, B:A, C:B, D:C}  clear=[D]
#
# 12-action trace, checkpoints at 3, 6, 9, 12:
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

ACTION: (unstack A from B)
ACTION: (put-down A)
ACTION: (unstack B from C)
STATE_JSON: ```json
{"on": {"A": "table", "B": "held", "C": "D", "D": "table"}, "clear": ["A", "C"], "holding": "B", "handempty": false}
```

ACTION: (put-down B)
ACTION: (unstack C from D)
ACTION: (put-down C)
STATE_JSON: ```json
{"on": {"A": "table", "B": "table", "C": "table", "D": "table"}, "clear": ["A", "B", "C", "D"], "holding": null, "handempty": true}
```

All blocks on the table. Build: B on A, then C on B, then D on C.

ACTION: (pick-up B)
ACTION: (stack B onto A)
ACTION: (pick-up C)
STATE_JSON: ```json
{"on": {"A": "table", "B": "A", "C": "held", "D": "table"}, "clear": ["B", "D"], "holding": "C", "handempty": false}
```

ACTION: (stack C onto B)
ACTION: (pick-up D)
ACTION: (stack D onto C)
STATE_JSON: ```json
{"on": {"A": "table", "B": "A", "C": "B", "D": "C"}, "clear": ["D"], "holding": null, "handempty": true}
```

Goal reached: B on A, C on B, D on C, A on table.

Plan summary:
1. (unstack A from B)
2. (put-down A)
3. (unstack B from C)
4. (put-down B)
5. (unstack C from D)
6. (put-down C)
7. (pick-up B)
8. (stack B onto A)
9. (pick-up C)
10. (stack C onto B)
11. (pick-up D)
12. (stack D onto C)\
"""


# ---------------------------------------------------------------------------
# Condition C — same examples with goal_satisfied / goal_remaining
#
# Goal predicates for Example 1: A on B, B on C, C on table, A clear
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
ACTION: (stack C onto B)

That loses the "C on table" condition — I moved backwards. Let me undo.

ACTION: (unstack C from B)
STATE_JSON: ```json
{"on": {"A": "table", "B": "A", "C": "held"}, "clear": ["B"], "holding": "C", "handempty": false, "goal_satisfied": [], "goal_remaining": ["A on B", "B on C", "C on table", "A clear"]}
```

No goal conditions satisfied. Put C down, then unstack B from A.

ACTION: (put-down C)
ACTION: (unstack B from A)
ACTION: (put-down B)
STATE_JSON: ```json
{"on": {"A": "table", "B": "table", "C": "table"}, "clear": ["A", "B", "C"], "holding": null, "handempty": true, "goal_satisfied": ["C on table", "A clear"], "goal_remaining": ["A on B", "B on C"]}
```

Two goal conditions satisfied. Build: B on C, then A on B.

ACTION: (pick-up B)
ACTION: (stack B onto C)
ACTION: (pick-up A)
STATE_JSON: ```json
{"on": {"A": "held", "B": "C", "C": "table"}, "clear": ["B"], "holding": "A", "handempty": false, "goal_satisfied": ["B on C", "C on table"], "goal_remaining": ["A on B", "A clear"]}
```

One action left.

ACTION: (stack A onto B)
STATE_JSON: ```json
{"on": {"A": "B", "B": "C", "C": "table"}, "clear": ["A"], "holding": null, "handempty": true, "goal_satisfied": ["A on B", "B on C", "C on table", "A clear"], "goal_remaining": []}
```

All goal conditions satisfied.

Plan summary:
1. (unstack B from A)
2. (put-down B)
3. (pick-up B)
4. (stack B onto C)
5. (pick-up A)
6. (stack A onto B)\
"""

BLOCKSWORLD_4OP_JSON_GOAL_EXAMPLE_2 = """\
Initial state (JSON):
{"on": {"A": "B", "B": "C", "C": "D", "D": "table"}, "clear": ["A"], "holding": null, "handempty": true, "goal_satisfied": [], "goal_remaining": ["A on table", "B on A", "C on B", "D on C", "D clear"]}

Goal state (JSON):
{"on": {"A": "table", "B": "A", "C": "B", "D": "C"}, "clear": ["D"], "holding": null, "handempty": true}

STATE_JSON: ```json
{"on": {"A": "B", "B": "C", "C": "D", "D": "table"}, "clear": ["A"], "holding": null, "handempty": true, "goal_satisfied": [], "goal_remaining": ["A on table", "B on A", "C on B", "D on C", "D clear"]}
```

I need to reverse the stack. Unstack everything to the table, then rebuild bottom-up.

ACTION: (unstack A from B)
ACTION: (put-down A)
ACTION: (unstack B from C)
STATE_JSON: ```json
{"on": {"A": "table", "B": "held", "C": "D", "D": "table"}, "clear": ["A", "C"], "holding": "B", "handempty": false, "goal_satisfied": ["A on table"], "goal_remaining": ["B on A", "C on B", "D on C", "D clear"]}
```

Good — A on table is satisfied. Continue unstacking.

ACTION: (put-down B)
ACTION: (unstack C from D)
ACTION: (put-down C)
STATE_JSON: ```json
{"on": {"A": "table", "B": "table", "C": "table", "D": "table"}, "clear": ["A", "B", "C", "D"], "holding": null, "handempty": true, "goal_satisfied": ["A on table", "D clear"], "goal_remaining": ["B on A", "C on B", "D on C"]}
```

Two goal conditions satisfied. Build: B on A, then C on B, then D on C.

ACTION: (pick-up B)
ACTION: (stack B onto A)
ACTION: (pick-up C)
STATE_JSON: ```json
{"on": {"A": "table", "B": "A", "C": "held", "D": "table"}, "clear": ["B", "D"], "holding": "C", "handempty": false, "goal_satisfied": ["A on table", "B on A", "D clear"], "goal_remaining": ["C on B", "D on C"]}
```

Three satisfied. Two actions left.

ACTION: (stack C onto B)
ACTION: (pick-up D)
ACTION: (stack D onto C)
STATE_JSON: ```json
{"on": {"A": "table", "B": "A", "C": "B", "D": "C"}, "clear": ["D"], "holding": null, "handempty": true, "goal_satisfied": ["A on table", "B on A", "C on B", "D on C", "D clear"], "goal_remaining": []}
```

All goal conditions satisfied.

Plan summary:
1. (unstack A from B)
2. (put-down A)
3. (unstack B from C)
4. (put-down B)
5. (unstack C from D)
6. (put-down C)
7. (pick-up B)
8. (stack B onto A)
9. (pick-up C)
10. (stack C onto B)
11. (pick-up D)
12. (stack D onto C)\
"""


def get_blocksworld_json_examples() -> list:
    """Condition B: NL 4-op actions, JSON state, no goal tracking."""
    return [BLOCKSWORLD_4OP_JSON_EXAMPLE_1, BLOCKSWORLD_4OP_JSON_EXAMPLE_2]


def get_blocksworld_json_goal_examples() -> list:
    """Condition C: NL 4-op actions, JSON state + goal_satisfied/goal_remaining."""
    return [BLOCKSWORLD_4OP_JSON_GOAL_EXAMPLE_1, BLOCKSWORLD_4OP_JSON_GOAL_EXAMPLE_2]
