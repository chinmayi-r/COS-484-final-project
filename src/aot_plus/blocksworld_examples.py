"""
Blocksworld few-shot examples for AoT+ JSON extension.

Action format: (pick-up X)  (put-down X)  (unstack X from Y)  (stack X onto Y)
State format:  STATE_JSON after EVERY action (k=1) — required for per-step error taxonomy.

Condition B: plain JSON state.
Condition C: JSON state + goal_satisfied / goal_remaining fields.
"""

# ---------------------------------------------------------------------------
# Condition B — k=1, STATE_JSON after every action
#
# Example 1: 3 blocks, includes wrong detour + backtrack
#   Initial: on={A:table, B:A, C:table}  clear=[B,C]
#   Goal:    on={A:B, B:C, C:table}      clear=[A]
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

Let me try placing C on B first — but that's the wrong direction. I'll try and undo.

ACTION: (pick-up C)
STATE_JSON: ```json
{"on": {"A": "table", "B": "A", "C": "held"}, "clear": ["B"], "holding": "C", "handempty": false}
```

ACTION: (stack C onto B)
STATE_JSON: ```json
{"on": {"A": "table", "B": "A", "C": "B"}, "clear": ["C"], "holding": null, "handempty": true}
```

That blocks A. Undo.

ACTION: (unstack C from B)
STATE_JSON: ```json
{"on": {"A": "table", "B": "A", "C": "held"}, "clear": ["B"], "holding": "C", "handempty": false}
```

ACTION: (put-down C)
STATE_JSON: ```json
{"on": {"A": "table", "B": "A", "C": "table"}, "clear": ["B", "C"], "holding": null, "handempty": true}
```

Back to start. Correct plan: unstack B from A, then build B on C, then A on B.

ACTION: (unstack B from A)
STATE_JSON: ```json
{"on": {"A": "table", "B": "held", "C": "table"}, "clear": ["A", "C"], "holding": "B", "handempty": false}
```

ACTION: (put-down B)
STATE_JSON: ```json
{"on": {"A": "table", "B": "table", "C": "table"}, "clear": ["A", "B", "C"], "holding": null, "handempty": true}
```

ACTION: (pick-up B)
STATE_JSON: ```json
{"on": {"A": "table", "B": "held", "C": "table"}, "clear": ["A", "C"], "holding": "B", "handempty": false}
```

ACTION: (stack B onto C)
STATE_JSON: ```json
{"on": {"A": "table", "B": "C", "C": "table"}, "clear": ["A", "B"], "holding": null, "handempty": true}
```

ACTION: (pick-up A)
STATE_JSON: ```json
{"on": {"A": "held", "B": "C", "C": "table"}, "clear": ["B"], "holding": "A", "handempty": false}
```

ACTION: (stack A onto B)
STATE_JSON: ```json
{"on": {"A": "B", "B": "C", "C": "table"}, "clear": ["A"], "holding": null, "handempty": true}
```

Goal reached.

Plan summary:
1. (unstack B from A)
2. (put-down B)
3. (pick-up B)
4. (stack B onto C)
5. (pick-up A)
6. (stack A onto B)\
"""

# ---------------------------------------------------------------------------
# Condition B — Example 2: 4 blocks, reverse stack, no backtracking
#   Initial: on={A:B, B:C, C:D, D:table}  clear=[A]
#   Goal:    on={A:table, B:A, C:B, D:C}  clear=[D]
# ---------------------------------------------------------------------------

BLOCKSWORLD_4OP_JSON_EXAMPLE_2 = """\
Initial state (JSON):
{"on": {"A": "B", "B": "C", "C": "D", "D": "table"}, "clear": ["A"], "holding": null, "handempty": true}

Goal state (JSON):
{"on": {"A": "table", "B": "A", "C": "B", "D": "C"}, "clear": ["D"], "holding": null, "handempty": true}

STATE_JSON: ```json
{"on": {"A": "B", "B": "C", "C": "D", "D": "table"}, "clear": ["A"], "holding": null, "handempty": true}
```

Reverse the stack: unstack everything to the table, then rebuild bottom-up.

ACTION: (unstack A from B)
STATE_JSON: ```json
{"on": {"A": "held", "B": "C", "C": "D", "D": "table"}, "clear": ["B"], "holding": "A", "handempty": false}
```

ACTION: (put-down A)
STATE_JSON: ```json
{"on": {"A": "table", "B": "C", "C": "D", "D": "table"}, "clear": ["A", "B"], "holding": null, "handempty": true}
```

ACTION: (unstack B from C)
STATE_JSON: ```json
{"on": {"A": "table", "B": "held", "C": "D", "D": "table"}, "clear": ["A", "C"], "holding": "B", "handempty": false}
```

ACTION: (put-down B)
STATE_JSON: ```json
{"on": {"A": "table", "B": "table", "C": "D", "D": "table"}, "clear": ["A", "B", "C"], "holding": null, "handempty": true}
```

ACTION: (unstack C from D)
STATE_JSON: ```json
{"on": {"A": "table", "B": "table", "C": "held", "D": "table"}, "clear": ["A", "B", "D"], "holding": "C", "handempty": false}
```

ACTION: (put-down C)
STATE_JSON: ```json
{"on": {"A": "table", "B": "table", "C": "table", "D": "table"}, "clear": ["A", "B", "C", "D"], "holding": null, "handempty": true}
```

All on table. Build B on A, then C on B, then D on C.

ACTION: (pick-up B)
STATE_JSON: ```json
{"on": {"A": "table", "B": "held", "C": "table", "D": "table"}, "clear": ["A", "C", "D"], "holding": "B", "handempty": false}
```

ACTION: (stack B onto A)
STATE_JSON: ```json
{"on": {"A": "table", "B": "A", "C": "table", "D": "table"}, "clear": ["B", "C", "D"], "holding": null, "handempty": true}
```

ACTION: (pick-up C)
STATE_JSON: ```json
{"on": {"A": "table", "B": "A", "C": "held", "D": "table"}, "clear": ["B", "D"], "holding": "C", "handempty": false}
```

ACTION: (stack C onto B)
STATE_JSON: ```json
{"on": {"A": "table", "B": "A", "C": "B", "D": "table"}, "clear": ["C", "D"], "holding": null, "handempty": true}
```

ACTION: (pick-up D)
STATE_JSON: ```json
{"on": {"A": "table", "B": "A", "C": "B", "D": "held"}, "clear": ["C"], "holding": "D", "handempty": false}
```

ACTION: (stack D onto C)
STATE_JSON: ```json
{"on": {"A": "table", "B": "A", "C": "B", "D": "C"}, "clear": ["D"], "holding": null, "handempty": true}
```

Goal reached.

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

Let me try placing C on B — but that loses C on table. I'll try and undo.

ACTION: (pick-up C)
STATE_JSON: ```json
{"on": {"A": "table", "B": "A", "C": "held"}, "clear": ["B"], "holding": "C", "handempty": false, "goal_satisfied": [], "goal_remaining": ["A on B", "B on C", "C on table", "A clear"]}
```

ACTION: (stack C onto B)
STATE_JSON: ```json
{"on": {"A": "table", "B": "A", "C": "B"}, "clear": ["C"], "holding": null, "handempty": true, "goal_satisfied": [], "goal_remaining": ["A on B", "B on C", "C on table", "A clear"]}
```

Moved backwards — lost C on table. Undo.

ACTION: (unstack C from B)
STATE_JSON: ```json
{"on": {"A": "table", "B": "A", "C": "held"}, "clear": ["B"], "holding": "C", "handempty": false, "goal_satisfied": [], "goal_remaining": ["A on B", "B on C", "C on table", "A clear"]}
```

ACTION: (put-down C)
STATE_JSON: ```json
{"on": {"A": "table", "B": "A", "C": "table"}, "clear": ["B", "C"], "holding": null, "handempty": true, "goal_satisfied": ["C on table"], "goal_remaining": ["A on B", "B on C", "A clear"]}
```

Back to start. Correct plan.

ACTION: (unstack B from A)
STATE_JSON: ```json
{"on": {"A": "table", "B": "held", "C": "table"}, "clear": ["A", "C"], "holding": "B", "handempty": false, "goal_satisfied": ["C on table", "A clear"], "goal_remaining": ["A on B", "B on C"]}
```

ACTION: (put-down B)
STATE_JSON: ```json
{"on": {"A": "table", "B": "table", "C": "table"}, "clear": ["A", "B", "C"], "holding": null, "handempty": true, "goal_satisfied": ["C on table", "A clear"], "goal_remaining": ["A on B", "B on C"]}
```

ACTION: (pick-up B)
STATE_JSON: ```json
{"on": {"A": "table", "B": "held", "C": "table"}, "clear": ["A", "C"], "holding": "B", "handempty": false, "goal_satisfied": ["C on table", "A clear"], "goal_remaining": ["A on B", "B on C"]}
```

ACTION: (stack B onto C)
STATE_JSON: ```json
{"on": {"A": "table", "B": "C", "C": "table"}, "clear": ["A", "B"], "holding": null, "handempty": true, "goal_satisfied": ["B on C", "C on table", "A clear"], "goal_remaining": ["A on B"]}
```

ACTION: (pick-up A)
STATE_JSON: ```json
{"on": {"A": "held", "B": "C", "C": "table"}, "clear": ["B"], "holding": "A", "handempty": false, "goal_satisfied": ["B on C", "C on table"], "goal_remaining": ["A on B", "A clear"]}
```

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

Reverse the stack: unstack everything, then rebuild.

ACTION: (unstack A from B)
STATE_JSON: ```json
{"on": {"A": "held", "B": "C", "C": "D", "D": "table"}, "clear": ["B"], "holding": "A", "handempty": false, "goal_satisfied": [], "goal_remaining": ["A on table", "B on A", "C on B", "D on C", "D clear"]}
```

ACTION: (put-down A)
STATE_JSON: ```json
{"on": {"A": "table", "B": "C", "C": "D", "D": "table"}, "clear": ["A", "B"], "holding": null, "handempty": true, "goal_satisfied": ["A on table"], "goal_remaining": ["B on A", "C on B", "D on C", "D clear"]}
```

ACTION: (unstack B from C)
STATE_JSON: ```json
{"on": {"A": "table", "B": "held", "C": "D", "D": "table"}, "clear": ["A", "C"], "holding": "B", "handempty": false, "goal_satisfied": ["A on table"], "goal_remaining": ["B on A", "C on B", "D on C", "D clear"]}
```

ACTION: (put-down B)
STATE_JSON: ```json
{"on": {"A": "table", "B": "table", "C": "D", "D": "table"}, "clear": ["A", "B", "C"], "holding": null, "handempty": true, "goal_satisfied": ["A on table"], "goal_remaining": ["B on A", "C on B", "D on C", "D clear"]}
```

ACTION: (unstack C from D)
STATE_JSON: ```json
{"on": {"A": "table", "B": "table", "C": "held", "D": "table"}, "clear": ["A", "B", "D"], "holding": "C", "handempty": false, "goal_satisfied": ["A on table", "D clear"], "goal_remaining": ["B on A", "C on B", "D on C"]}
```

ACTION: (put-down C)
STATE_JSON: ```json
{"on": {"A": "table", "B": "table", "C": "table", "D": "table"}, "clear": ["A", "B", "C", "D"], "holding": null, "handempty": true, "goal_satisfied": ["A on table", "D clear"], "goal_remaining": ["B on A", "C on B", "D on C"]}
```

ACTION: (pick-up B)
STATE_JSON: ```json
{"on": {"A": "table", "B": "held", "C": "table", "D": "table"}, "clear": ["A", "C", "D"], "holding": "B", "handempty": false, "goal_satisfied": ["A on table", "D clear"], "goal_remaining": ["B on A", "C on B", "D on C"]}
```

ACTION: (stack B onto A)
STATE_JSON: ```json
{"on": {"A": "table", "B": "A", "C": "table", "D": "table"}, "clear": ["B", "C", "D"], "holding": null, "handempty": true, "goal_satisfied": ["A on table", "B on A", "D clear"], "goal_remaining": ["C on B", "D on C"]}
```

ACTION: (pick-up C)
STATE_JSON: ```json
{"on": {"A": "table", "B": "A", "C": "held", "D": "table"}, "clear": ["B", "D"], "holding": "C", "handempty": false, "goal_satisfied": ["A on table", "B on A", "D clear"], "goal_remaining": ["C on B", "D on C"]}
```

ACTION: (stack C onto B)
STATE_JSON: ```json
{"on": {"A": "table", "B": "A", "C": "B", "D": "table"}, "clear": ["C", "D"], "holding": null, "handempty": true, "goal_satisfied": ["A on table", "B on A", "C on B", "D clear"], "goal_remaining": ["D on C"]}
```

ACTION: (pick-up D)
STATE_JSON: ```json
{"on": {"A": "table", "B": "A", "C": "B", "D": "held"}, "clear": ["C"], "holding": "D", "handempty": false, "goal_satisfied": ["A on table", "B on A", "C on B"], "goal_remaining": ["D on C", "D clear"]}
```

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
    """Condition B: k=1, JSON state after every action, no goal tracking."""
    return [BLOCKSWORLD_4OP_JSON_EXAMPLE_1, BLOCKSWORLD_4OP_JSON_EXAMPLE_2]


def get_blocksworld_json_goal_examples() -> list:
    """Condition C: k=1, JSON state + goal_satisfied/goal_remaining after every action."""
    return [BLOCKSWORLD_4OP_JSON_GOAL_EXAMPLE_1, BLOCKSWORLD_4OP_JSON_GOAL_EXAMPLE_2]
