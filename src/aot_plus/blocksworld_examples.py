"""
Blocksworld few-shot examples for AoT+ JSON extension.

Format matches the paper (Algorithm of Thoughts, Appendix B.3) EXCEPT:
  - State is JSON (instead of compact NL)
  - Checkpoints every 3 actions (instead of after every action)
  - Each STATE_JSON includes goal_satisfied / goal_remaining fields

Actions use composite "Move Block X from Y to Z" (maps to pick-up+stack or
unstack+put-down in the paper's 4-op set).
"""

# ---------------------------------------------------------------------------
# Example 1 — 3 blocks, includes one wrong move + undo (backtracking)
#
# Initial : A on table, B on A (B clear), C on table (C clear)
# Goal    : A on B, B on C, C on table  (stack C→B→A from bottom)
#
# Goal predicates: A on B, B on C, C on table, A clear
# Initial satisfied: C on table
#
# Steps:
#   1. Move C from table to B  (wrong — loses "C on table")
#   2. Move C from B to table  (undo)
#   3. Move B from A to table  ← CHECKPOINT (step 3)
#      goal_satisfied: ["C on table", "A clear"]
#      goal_remaining: ["A on B", "B on C"]
#   4. Move B from table to C
#   5. Move A from table to B  ← CHECKPOINT (step 5, final)
#      goal_satisfied: all 4
# ---------------------------------------------------------------------------

BLOCKSWORLD_JSON_EXAMPLE_1 = """\
Initial state (JSON):
{"on": {"A": "table", "B": "A", "C": "table"}, "clear": ["B", "C"], "holding": null, "handempty": true}

Goal state (JSON):
{"on": {"A": "B", "B": "C", "C": "table"}, "clear": ["A"], "holding": null, "handempty": true}

STATE_JSON: ```json
{"on": {"A": "table", "B": "A", "C": "table"}, "clear": ["B", "C"], "holding": null, "handempty": true}
```

Goal analysis: A on B (no), B on C (no), C on table (yes), A clear (no — B is on A).

Let me try placing C on B first.

ACTION: Move Block C from the table to Block B
That puts C on B but loses "C on table". Let me undo this.
ACTION: Move Block C from Block B to the table
Back to the start. The right approach: unstack B off A so A becomes clear.
ACTION: Move Block B from Block A to the table
STATE_JSON: ```json
{"on": {"A": "table", "B": "table", "C": "table"}, "clear": ["A", "B", "C"], "holding": null, "handempty": true, "goal_satisfied": ["C on table", "A clear"], "goal_remaining": ["A on B", "B on C"]}
```

Two conditions satisfied. Now place B on C to build the base of the goal stack.

ACTION: Move Block B from the table to Block C
Finally place A on top of B.
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

# ---------------------------------------------------------------------------
# Example 2 — 4 blocks, clean reverse-stack (no backtracking)
#
# Initial : stack D→C→B→A from bottom  (only A is clear)
# Goal    : stack A→B→C→D from bottom  (only D is clear)
#
# Goal predicates: A on table, B on A, C on B, D on C, D clear
# Initial satisfied: none (D on table but goal has D on C)
#
# Steps:
#   1. Move A from B to table
#   2. Move B from C to table
#   3. Move C from D to table  ← CHECKPOINT (step 3)
#      goal_satisfied: ["A on table", "D clear"]
#      goal_remaining: ["B on A", "C on B", "D on C"]
#   4. Move B from table to A
#   5. Move C from table to B
#   6. Move D from table to C  ← CHECKPOINT (step 6, final)
#      goal_satisfied: all 5
# ---------------------------------------------------------------------------

BLOCKSWORLD_JSON_EXAMPLE_2 = """\
Initial state (JSON):
{"on": {"A": "B", "B": "C", "C": "D", "D": "table"}, "clear": ["A"], "holding": null, "handempty": true}

Goal state (JSON):
{"on": {"A": "table", "B": "A", "C": "B", "D": "C"}, "clear": ["D"], "holding": null, "handempty": true}

STATE_JSON: ```json
{"on": {"A": "B", "B": "C", "C": "D", "D": "table"}, "clear": ["A"], "holding": null, "handempty": true}
```

Goal analysis: A on table (no — A on B), B on A (no — B on C), C on B (no — C on D), D on C (no — D on table), D clear (no — C on D).

I need to reverse the entire stack: unstack everything to the table first, then rebuild bottom-up.

ACTION: Move Block A from Block B to the table
ACTION: Move Block B from Block C to the table
ACTION: Move Block C from Block D to the table
STATE_JSON: ```json
{"on": {"A": "table", "B": "table", "C": "table", "D": "table"}, "clear": ["A", "B", "C", "D"], "holding": null, "handempty": true, "goal_satisfied": ["A on table", "D clear"], "goal_remaining": ["B on A", "C on B", "D on C"]}
```

Everything on the table. Now build the stack bottom-up: B on A, C on B, D on C.

ACTION: Move Block B from the table to Block A
ACTION: Move Block C from the table to Block B
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
