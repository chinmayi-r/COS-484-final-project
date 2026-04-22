"""
Hardcoded few-shot examples for Blocksworld in JSON and JSON+goal-residual formats.

Every ACTION is immediately followed by exactly one STATE_JSON block so that
Abby's positional alignment (block index == action index) is maintained.
States are computed by hand-tracing the simulator rules:
  - move(X, dest): X must be clear; dest must be clear (or table);
                   removes X from clear, adds old_location_of_X to clear,
                   removes dest from clear (if not table).
  - holding/handempty: always null/true in simplified Blocksworld (no pick-up).
"""

# ---------------------------------------------------------------------------
# Example 1 — 3 blocks, includes one wrong move + undo (backtracking)
#
# Initial : A on table, B on A (B clear), C on table (C clear)
# Goal    : A on B, B on C, C on table  (stack C→B→A from bottom)
#
# State trace:
#   S0  on={A:table,B:A,C:table}  clear=[B,C]
#   wrong: move C onto B
#   S1w on={A:table,B:A,C:B}     clear=[C]          ← wrong direction
#   undo: move C from B back to table
#   S1u on={A:table,B:A,C:table} clear=[B,C]         ← back to S0
#   a1: move B from A to table
#   S2  on={A:table,B:table,C:table}  clear=[A,B,C]
#   a2: move B from table onto C
#   S3  on={A:table,B:C,C:table}  clear=[A,B]
#   a3: move A from table onto B
#   S4  on={A:B,B:C,C:table}      clear=[A]          ← GOAL
# ---------------------------------------------------------------------------

BLOCKSWORLD_JSON_EXAMPLE_1 = """\
Initial state (JSON):
{"on": {"A": "table", "B": "A", "C": "table"}, "clear": ["B", "C"], "holding": null, "handempty": true}

Goal state (JSON):
{"on": {"A": "B", "B": "C", "C": "table"}, "clear": ["A"], "holding": null, "handempty": true}

STATE_JSON: ```json
{"on": {"A": "table", "B": "A", "C": "table"}, "clear": ["B", "C"], "holding": null, "handempty": true}
```

I'll solve this step by step.

Let me try placing C on B first.

ACTION: Move Block C from the table to Block B
STATE_JSON: ```json
{"on": {"A": "table", "B": "A", "C": "B"}, "clear": ["C"], "holding": null, "handempty": true}
```

That leaves A blocked and doesn't progress. Let me undo this move.

ACTION: Move Block C from Block B to the table
STATE_JSON: ```json
{"on": {"A": "table", "B": "A", "C": "table"}, "clear": ["B", "C"], "holding": null, "handempty": true}
```

Good, back to the start. The correct approach is to clear B off A first.

ACTION: Move Block B from Block A to the table
STATE_JSON: ```json
{"on": {"A": "table", "B": "table", "C": "table"}, "clear": ["A", "B", "C"], "holding": null, "handempty": true}
```

Now place B on C to build the middle of the stack.

ACTION: Move Block B from the table to Block C
STATE_JSON: ```json
{"on": {"A": "table", "B": "C", "C": "table"}, "clear": ["A", "B"], "holding": null, "handempty": true}
```

Finally place A on top of B.

ACTION: Move Block A from the table to Block B
STATE_JSON: ```json
{"on": {"A": "B", "B": "C", "C": "table"}, "clear": ["A"], "holding": null, "handempty": true}
```

Goal reached: A on B, B on C, C on table.

Plan summary:
1. Move Block B from Block A to the table
2. Move Block B from the table to Block C
3. Move Block A from the table to Block B\
"""

# ---------------------------------------------------------------------------
# Example 2 — 4 blocks, clean reverse-stack (no backtracking)
#
# Initial : stack D→C→B→A from bottom  (only A is clear)
# Goal    : stack A→B→C→D from bottom  (only D is clear)
#
# State trace:
#   S0  on={A:B,B:C,C:D,D:table}       clear=[A]
#   a1: move A from B to table
#   S1  on={A:table,B:C,C:D,D:table}   clear=[A,B]
#   a2: move B from C to table
#   S2  on={A:table,B:table,C:D,D:table} clear=[A,B,C]
#   a3: move C from D to table
#   S3  on={A:table,B:table,C:table,D:table} clear=[A,B,C,D]
#   a4: move B onto A
#   S4  on={A:table,B:A,C:table,D:table}     clear=[B,C,D]
#   a5: move C onto B
#   S5  on={A:table,B:A,C:B,D:table}         clear=[C,D]
#   a6: move D onto C
#   S6  on={A:table,B:A,C:B,D:C}             clear=[D]  ← GOAL
# ---------------------------------------------------------------------------

BLOCKSWORLD_JSON_EXAMPLE_2 = """\
Initial state (JSON):
{"on": {"A": "B", "B": "C", "C": "D", "D": "table"}, "clear": ["A"], "holding": null, "handempty": true}

Goal state (JSON):
{"on": {"A": "table", "B": "A", "C": "B", "D": "C"}, "clear": ["D"], "holding": null, "handempty": true}

STATE_JSON: ```json
{"on": {"A": "B", "B": "C", "C": "D", "D": "table"}, "clear": ["A"], "holding": null, "handempty": true}
```

I need to reverse the entire stack. I'll unstack everything to the table first.

ACTION: Move Block A from Block B to the table
STATE_JSON: ```json
{"on": {"A": "table", "B": "C", "C": "D", "D": "table"}, "clear": ["A", "B"], "holding": null, "handempty": true}
```

ACTION: Move Block B from Block C to the table
STATE_JSON: ```json
{"on": {"A": "table", "B": "table", "C": "D", "D": "table"}, "clear": ["A", "B", "C"], "holding": null, "handempty": true}
```

ACTION: Move Block C from Block D to the table
STATE_JSON: ```json
{"on": {"A": "table", "B": "table", "C": "table", "D": "table"}, "clear": ["A", "B", "C", "D"], "holding": null, "handempty": true}
```

Now build the new stack bottom-up: B on A, then C on B, then D on C.

ACTION: Move Block B from the table to Block A
STATE_JSON: ```json
{"on": {"A": "table", "B": "A", "C": "table", "D": "table"}, "clear": ["B", "C", "D"], "holding": null, "handempty": true}
```

ACTION: Move Block C from the table to Block B
STATE_JSON: ```json
{"on": {"A": "table", "B": "A", "C": "B", "D": "table"}, "clear": ["C", "D"], "holding": null, "handempty": true}
```

ACTION: Move Block D from the table to Block C
STATE_JSON: ```json
{"on": {"A": "table", "B": "A", "C": "B", "D": "C"}, "clear": ["D"], "holding": null, "handempty": true}
```

Goal reached: B on A, C on B, D on C, A on table.

Plan summary:
1. Move Block A from Block B to the table
2. Move Block B from Block C to the table
3. Move Block C from Block D to the table
4. Move Block B from the table to Block A
5. Move Block C from the table to Block B
6. Move Block D from the table to Block C\
"""

# ---------------------------------------------------------------------------
# JSON + Goal-Residual variants
# Same examples with goal_satisfied / goal_remaining appended to each state.
#
# Goal predicates for Example 1:
#   on(A,B), on(B,C), on(C,table), clear(A)
# Represented as strings: "A on B", "B on C", "C on table", "A clear"
# ---------------------------------------------------------------------------

BLOCKSWORLD_JSON_GOAL_EXAMPLE_1 = """\
Initial state (JSON):
{"on": {"A": "table", "B": "A", "C": "table"}, "clear": ["B", "C"], "holding": null, "handempty": true, "goal_satisfied": ["C on table"], "goal_remaining": ["A on B", "B on C", "A clear"]}

Goal state (JSON):
{"on": {"A": "B", "B": "C", "C": "table"}, "clear": ["A"], "holding": null, "handempty": true}

STATE_JSON: ```json
{"on": {"A": "table", "B": "A", "C": "table"}, "clear": ["B", "C"], "holding": null, "handempty": true, "goal_satisfied": ["C on table"], "goal_remaining": ["A on B", "B on C", "A clear"]}
```

Let me try placing C on B first.

ACTION: Move Block C from the table to Block B
STATE_JSON: ```json
{"on": {"A": "table", "B": "A", "C": "B"}, "clear": ["C"], "holding": null, "handempty": true, "goal_satisfied": [], "goal_remaining": ["A on B", "B on C", "C on table", "A clear"]}
```

That lost the C on table condition. Let me undo.

ACTION: Move Block C from Block B to the table
STATE_JSON: ```json
{"on": {"A": "table", "B": "A", "C": "table"}, "clear": ["B", "C"], "holding": null, "handempty": true, "goal_satisfied": ["C on table"], "goal_remaining": ["A on B", "B on C", "A clear"]}
```

Back to start. Now the correct approach.

ACTION: Move Block B from Block A to the table
STATE_JSON: ```json
{"on": {"A": "table", "B": "table", "C": "table"}, "clear": ["A", "B", "C"], "holding": null, "handempty": true, "goal_satisfied": ["C on table", "A clear"], "goal_remaining": ["A on B", "B on C"]}
```

Two goal conditions satisfied. Now place B on C.

ACTION: Move Block B from the table to Block C
STATE_JSON: ```json
{"on": {"A": "table", "B": "C", "C": "table"}, "clear": ["A", "B"], "holding": null, "handempty": true, "goal_satisfied": ["B on C", "C on table", "A clear"], "goal_remaining": ["A on B"]}
```

One condition left. Place A on B.

ACTION: Move Block A from the table to Block B
STATE_JSON: ```json
{"on": {"A": "B", "B": "C", "C": "table"}, "clear": ["A"], "holding": null, "handempty": true, "goal_satisfied": ["A on B", "B on C", "C on table", "A clear"], "goal_remaining": []}
```

All goal conditions satisfied.

Plan summary:
1. Move Block B from Block A to the table
2. Move Block B from the table to Block C
3. Move Block A from the table to Block B\
"""

BLOCKSWORLD_JSON_GOAL_EXAMPLE_2 = """\
Initial state (JSON):
{"on": {"A": "B", "B": "C", "C": "D", "D": "table"}, "clear": ["A"], "holding": null, "handempty": true, "goal_satisfied": ["D on table"], "goal_remaining": ["B on A", "C on B", "D on C", "D clear"]}

Goal state (JSON):
{"on": {"A": "table", "B": "A", "C": "B", "D": "C"}, "clear": ["D"], "holding": null, "handempty": true}

STATE_JSON: ```json
{"on": {"A": "B", "B": "C", "C": "D", "D": "table"}, "clear": ["A"], "holding": null, "handempty": true, "goal_satisfied": ["D on table"], "goal_remaining": ["B on A", "C on B", "D on C", "D clear"]}
```

I need to reverse the stack. Unstack everything first.

ACTION: Move Block A from Block B to the table
STATE_JSON: ```json
{"on": {"A": "table", "B": "C", "C": "D", "D": "table"}, "clear": ["A", "B"], "holding": null, "handempty": true, "goal_satisfied": ["A on table", "D on table"], "goal_remaining": ["B on A", "C on B", "D on C", "D clear"]}
```

ACTION: Move Block B from Block C to the table
STATE_JSON: ```json
{"on": {"A": "table", "B": "table", "C": "D", "D": "table"}, "clear": ["A", "B", "C"], "holding": null, "handempty": true, "goal_satisfied": ["A on table", "D on table"], "goal_remaining": ["B on A", "C on B", "D on C", "D clear"]}
```

ACTION: Move Block C from Block D to the table
STATE_JSON: ```json
{"on": {"A": "table", "B": "table", "C": "table", "D": "table"}, "clear": ["A", "B", "C", "D"], "holding": null, "handempty": true, "goal_satisfied": ["A on table", "D on table", "D clear"], "goal_remaining": ["B on A", "C on B", "D on C"]}
```

Now build the new stack.

ACTION: Move Block B from the table to Block A
STATE_JSON: ```json
{"on": {"A": "table", "B": "A", "C": "table", "D": "table"}, "clear": ["B", "C", "D"], "holding": null, "handempty": true, "goal_satisfied": ["A on table", "B on A", "D on table", "D clear"], "goal_remaining": ["C on B", "D on C"]}
```

ACTION: Move Block C from the table to Block B
STATE_JSON: ```json
{"on": {"A": "table", "B": "A", "C": "B", "D": "table"}, "clear": ["C", "D"], "holding": null, "handempty": true, "goal_satisfied": ["A on table", "B on A", "C on B", "D on table", "D clear"], "goal_remaining": ["D on C"]}
```

ACTION: Move Block D from the table to Block C
STATE_JSON: ```json
{"on": {"A": "table", "B": "A", "C": "B", "D": "C"}, "clear": ["D"], "holding": null, "handempty": true, "goal_satisfied": ["A on table", "B on A", "C on B", "D on C", "D clear"], "goal_remaining": []}
```

All goal conditions satisfied.

Plan summary:
1. Move Block A from Block B to the table
2. Move Block B from Block C to the table
3. Move Block C from Block D to the table
4. Move Block B from the table to Block A
5. Move Block C from the table to Block B
6. Move Block D from the table to Block C\
"""


def get_blocksworld_json_examples() -> list:
    return [BLOCKSWORLD_JSON_EXAMPLE_1, BLOCKSWORLD_JSON_EXAMPLE_2]


def get_blocksworld_json_goal_examples() -> list:
    return [BLOCKSWORLD_JSON_GOAL_EXAMPLE_1, BLOCKSWORLD_JSON_GOAL_EXAMPLE_2]
