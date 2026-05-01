"""
Hardcoded few-shot examples for Logistics in JSON and JSON+goal-residual formats.

Every ACTION is immediately followed by exactly one STATE_JSON block so that
Abby's positional alignment (block index == action index) is maintained.
States are hand-traced against the simulator rules:
  - drive-truck T src dst city  : T.at src → dst  (same city, any locations)
  - load-truck P T loc          : P.at loc → null, P.in → T  (T.at == loc)
  - unload-truck P T loc        : P.in T → null, P.at → loc  (T.at == loc)
  - fly-airplane A src dst      : A.at src → dst  (both airports, different cities)
  - load-airplane P A loc       : P.at loc → null, P.in → A  (A.at == loc, loc is airport)
  - unload-airplane P A loc     : P.in A → null, P.at → loc  (A.at == loc, loc is airport)

Airport convention: a location whose name ends with "-0" is an airport.
City map used in both examples:
  L0-0 → C0 (airport), L0-1 → C0, L0-2 → C0
  L1-0 → C1 (airport), L1-1 → C1, L1-2 → C1
"""

# ---------------------------------------------------------------------------
# Example 1 — Intra-city delivery with one wrong move (backtracking)
#
# Initial : P0 at L0-2, T0 at L0-1, A0 at L0-0
# Goal    : P0 at L0-1
#
# State trace:
#   S0  at={A0:L0-0, P0:L0-2, T0:L0-1}  in={P0:null}
#   wrong: drive T0 to airport instead of to P0
#   S1w at={A0:L0-0, P0:L0-2, T0:L0-0}  in={P0:null}
#   undo: drive T0 back to L0-1
#   S1u at={A0:L0-0, P0:L0-2, T0:L0-1}  in={P0:null}  ← back to S0
#   a1: drive T0 from L0-1 to L0-2
#   S2  at={A0:L0-0, P0:L0-2, T0:L0-2}  in={P0:null}
#   a2: load P0 onto T0 at L0-2
#   S3  at={A0:L0-0, P0:null, T0:L0-2}  in={P0:T0}
#   a3: drive T0 from L0-2 to L0-1
#   S4  at={A0:L0-0, P0:null, T0:L0-1}  in={P0:T0}
#   a4: unload P0 from T0 at L0-1
#   S5  at={A0:L0-0, P0:L0-1, T0:L0-1} in={P0:null}  ← GOAL
# ---------------------------------------------------------------------------

LOGISTICS_JSON_EXAMPLE_1 = """\
Initial state (JSON):
{"at": {"A0": "L0-0", "P0": "L0-2", "T0": "L0-1"}, "in": {"P0": null}}

Goal state (JSON):
{"at": {"P0": "L0-1"}, "in": {"P0": null}}

STATE_JSON: ```json
{"at": {"A0": "L0-0", "P0": "L0-2", "T0": "L0-1"}, "in": {"P0": null}}
```

I need to deliver P0 from L0-2 to L0-1 using T0.

Let me first try driving the truck toward the airport — wait, that is the wrong direction.

ACTION: drive-truck T0 L0-1 L0-0 C0
STATE_JSON: ```json
{"at": {"A0": "L0-0", "P0": "L0-2", "T0": "L0-0"}, "in": {"P0": null}}
```

T0 is now at the airport but P0 is still at L0-2. I need T0 at L0-2, so let me backtrack.

ACTION: drive-truck T0 L0-0 L0-1 C0
STATE_JSON: ```json
{"at": {"A0": "L0-0", "P0": "L0-2", "T0": "L0-1"}, "in": {"P0": null}}
```

Back to the start. The correct move is to drive T0 directly to L0-2 where P0 is.

ACTION: drive-truck T0 L0-1 L0-2 C0
STATE_JSON: ```json
{"at": {"A0": "L0-0", "P0": "L0-2", "T0": "L0-2"}, "in": {"P0": null}}
```

T0 and P0 are both at L0-2. Load P0.

ACTION: load-truck P0 T0 L0-2
STATE_JSON: ```json
{"at": {"A0": "L0-0", "P0": null, "T0": "L0-2"}, "in": {"P0": "T0"}}
```

P0 is on T0. Drive to the goal location L0-1.

ACTION: drive-truck T0 L0-2 L0-1 C0
STATE_JSON: ```json
{"at": {"A0": "L0-0", "P0": null, "T0": "L0-1"}, "in": {"P0": "T0"}}
```

Unload P0 at L0-1 — the goal.

ACTION: unload-truck P0 T0 L0-1
STATE_JSON: ```json
{"at": {"A0": "L0-0", "P0": "L0-1", "T0": "L0-1"}, "in": {"P0": null}}
```

Goal reached: P0 is at L0-1.

Plan summary:
1. drive-truck T0 L0-1 L0-2 C0
2. load-truck P0 T0 L0-2
3. drive-truck T0 L0-2 L0-1 C0
4. unload-truck P0 T0 L0-1\
"""

# ---------------------------------------------------------------------------
# Example 2 — Cross-city delivery (clean, no backtracking)
#
# Initial : P0 at L0-1, T0 at L0-2, T1 at L1-2, A0 at L0-0
# Goal    : P0 at L1-1
#
# State trace:
#   S0   at={A0:L0-0, P0:L0-1, T0:L0-2, T1:L1-2}  in={P0:null}
#   a1: drive T0 L0-2 → L0-1
#   S1   at={A0:L0-0, P0:L0-1, T0:L0-1, T1:L1-2}  in={P0:null}
#   a2: load P0 onto T0 at L0-1
#   S2   at={A0:L0-0, P0:null, T0:L0-1, T1:L1-2}  in={P0:T0}
#   a3: drive T0 L0-1 → L0-0 (airport)
#   S3   at={A0:L0-0, P0:null, T0:L0-0, T1:L1-2}  in={P0:T0}
#   a4: unload P0 at L0-0
#   S4   at={A0:L0-0, P0:L0-0, T0:L0-0, T1:L1-2}  in={P0:null}
#   a5: load P0 onto A0 at L0-0
#   S5   at={A0:L0-0, P0:null, T0:L0-0, T1:L1-2}  in={P0:A0}
#   a6: fly A0 L0-0 → L1-0
#   S6   at={A0:L1-0, P0:null, T0:L0-0, T1:L1-2}  in={P0:A0}
#   a7: unload P0 at L1-0
#   S7   at={A0:L1-0, P0:L1-0, T0:L0-0, T1:L1-2}  in={P0:null}
#   a8: drive T1 L1-2 → L1-0
#   S8   at={A0:L1-0, P0:L1-0, T0:L0-0, T1:L1-0}  in={P0:null}
#   a9: load P0 onto T1 at L1-0
#   S9   at={A0:L1-0, P0:null, T0:L0-0, T1:L1-0}  in={P0:T1}
#   a10: drive T1 L1-0 → L1-1
#   S10  at={A0:L1-0, P0:null, T0:L0-0, T1:L1-1}  in={P0:T1}
#   a11: unload P0 at L1-1
#   S11  at={A0:L1-0, P0:L1-1, T0:L0-0, T1:L1-1}  in={P0:null}  ← GOAL
# ---------------------------------------------------------------------------

LOGISTICS_JSON_EXAMPLE_2 = """\
Initial state (JSON):
{"at": {"A0": "L0-0", "P0": "L0-1", "T0": "L0-2", "T1": "L1-2"}, "in": {"P0": null}}

Goal state (JSON):
{"at": {"P0": "L1-1"}, "in": {"P0": null}}

STATE_JSON: ```json
{"at": {"A0": "L0-0", "P0": "L0-1", "T0": "L0-2", "T1": "L1-2"}, "in": {"P0": null}}
```

P0 must travel from city C0 (L0-1) to city C1 (L1-1). Plan: truck T0 carries P0 to C0 airport, airplane A0 flies it to C1 airport, truck T1 delivers it to L1-1.

Drive T0 to L0-1 where P0 is waiting.

ACTION: drive-truck T0 L0-2 L0-1 C0
STATE_JSON: ```json
{"at": {"A0": "L0-0", "P0": "L0-1", "T0": "L0-1", "T1": "L1-2"}, "in": {"P0": null}}
```

Load P0 onto T0.

ACTION: load-truck P0 T0 L0-1
STATE_JSON: ```json
{"at": {"A0": "L0-0", "P0": null, "T0": "L0-1", "T1": "L1-2"}, "in": {"P0": "T0"}}
```

Drive T0 with P0 to the C0 airport (L0-0).

ACTION: drive-truck T0 L0-1 L0-0 C0
STATE_JSON: ```json
{"at": {"A0": "L0-0", "P0": null, "T0": "L0-0", "T1": "L1-2"}, "in": {"P0": "T0"}}
```

Unload P0 at the airport so it can be loaded onto the airplane.

ACTION: unload-truck P0 T0 L0-0
STATE_JSON: ```json
{"at": {"A0": "L0-0", "P0": "L0-0", "T0": "L0-0", "T1": "L1-2"}, "in": {"P0": null}}
```

Load P0 onto airplane A0 at L0-0.

ACTION: load-airplane P0 A0 L0-0
STATE_JSON: ```json
{"at": {"A0": "L0-0", "P0": null, "T0": "L0-0", "T1": "L1-2"}, "in": {"P0": "A0"}}
```

Fly A0 from C0 airport to C1 airport.

ACTION: fly-airplane A0 L0-0 L1-0
STATE_JSON: ```json
{"at": {"A0": "L1-0", "P0": null, "T0": "L0-0", "T1": "L1-2"}, "in": {"P0": "A0"}}
```

Unload P0 at C1 airport (L1-0).

ACTION: unload-airplane P0 A0 L1-0
STATE_JSON: ```json
{"at": {"A0": "L1-0", "P0": "L1-0", "T0": "L0-0", "T1": "L1-2"}, "in": {"P0": null}}
```

Drive T1 from L1-2 to the airport to pick up P0.

ACTION: drive-truck T1 L1-2 L1-0 C1
STATE_JSON: ```json
{"at": {"A0": "L1-0", "P0": "L1-0", "T0": "L0-0", "T1": "L1-0"}, "in": {"P0": null}}
```

Load P0 onto T1 at L1-0.

ACTION: load-truck P0 T1 L1-0
STATE_JSON: ```json
{"at": {"A0": "L1-0", "P0": null, "T0": "L0-0", "T1": "L1-0"}, "in": {"P0": "T1"}}
```

Drive T1 with P0 to the destination L1-1.

ACTION: drive-truck T1 L1-0 L1-1 C1
STATE_JSON: ```json
{"at": {"A0": "L1-0", "P0": null, "T0": "L0-0", "T1": "L1-1"}, "in": {"P0": "T1"}}
```

Unload P0 at L1-1 — the goal.

ACTION: unload-truck P0 T1 L1-1
STATE_JSON: ```json
{"at": {"A0": "L1-0", "P0": "L1-1", "T0": "L0-0", "T1": "L1-1"}, "in": {"P0": null}}
```

Goal reached: P0 is at L1-1.

Plan summary:
1. drive-truck T0 L0-2 L0-1 C0
2. load-truck P0 T0 L0-1
3. drive-truck T0 L0-1 L0-0 C0
4. unload-truck P0 T0 L0-0
5. load-airplane P0 A0 L0-0
6. fly-airplane A0 L0-0 L1-0
7. unload-airplane P0 A0 L1-0
8. drive-truck T1 L1-2 L1-0 C1
9. load-truck P0 T1 L1-0
10. drive-truck T1 L1-0 L1-1 C1
11. unload-truck P0 T1 L1-1\
"""

# ---------------------------------------------------------------------------
# JSON + Goal-Residual variants
# Same examples with goal_satisfied / goal_remaining appended to each state.
#
# Goal predicates are expressed as strings: "P0 at L0-1"
# ---------------------------------------------------------------------------

LOGISTICS_JSON_GOAL_EXAMPLE_1 = """\
Initial state (JSON):
{"at": {"A0": "L0-0", "P0": "L0-2", "T0": "L0-1"}, "in": {"P0": null}, "goal_satisfied": [], "goal_remaining": ["P0 at L0-1"]}

Goal state (JSON):
{"at": {"P0": "L0-1"}, "in": {"P0": null}}

STATE_JSON: ```json
{"at": {"A0": "L0-0", "P0": "L0-2", "T0": "L0-1"}, "in": {"P0": null}, "goal_satisfied": [], "goal_remaining": ["P0 at L0-1"]}
```

I need to deliver P0 from L0-2 to L0-1. Let me try driving toward the airport first.

ACTION: drive-truck T0 L0-1 L0-0 C0
STATE_JSON: ```json
{"at": {"A0": "L0-0", "P0": "L0-2", "T0": "L0-0"}, "in": {"P0": null}, "goal_satisfied": [], "goal_remaining": ["P0 at L0-1"]}
```

T0 went to the airport but P0 is still at L0-2. That didn't help. Backtrack.

ACTION: drive-truck T0 L0-0 L0-1 C0
STATE_JSON: ```json
{"at": {"A0": "L0-0", "P0": "L0-2", "T0": "L0-1"}, "in": {"P0": null}, "goal_satisfied": [], "goal_remaining": ["P0 at L0-1"]}
```

Back to start. Drive T0 directly to L0-2 where P0 is.

ACTION: drive-truck T0 L0-1 L0-2 C0
STATE_JSON: ```json
{"at": {"A0": "L0-0", "P0": "L0-2", "T0": "L0-2"}, "in": {"P0": null}, "goal_satisfied": [], "goal_remaining": ["P0 at L0-1"]}
```

Load P0 onto T0.

ACTION: load-truck P0 T0 L0-2
STATE_JSON: ```json
{"at": {"A0": "L0-0", "P0": null, "T0": "L0-2"}, "in": {"P0": "T0"}, "goal_satisfied": [], "goal_remaining": ["P0 at L0-1"]}
```

Drive T0 with P0 to L0-1.

ACTION: drive-truck T0 L0-2 L0-1 C0
STATE_JSON: ```json
{"at": {"A0": "L0-0", "P0": null, "T0": "L0-1"}, "in": {"P0": "T0"}, "goal_satisfied": [], "goal_remaining": ["P0 at L0-1"]}
```

Unload P0 at L0-1.

ACTION: unload-truck P0 T0 L0-1
STATE_JSON: ```json
{"at": {"A0": "L0-0", "P0": "L0-1", "T0": "L0-1"}, "in": {"P0": null}, "goal_satisfied": ["P0 at L0-1"], "goal_remaining": []}
```

All goal conditions satisfied.

Plan summary:
1. drive-truck T0 L0-1 L0-2 C0
2. load-truck P0 T0 L0-2
3. drive-truck T0 L0-2 L0-1 C0
4. unload-truck P0 T0 L0-1\
"""

LOGISTICS_JSON_GOAL_EXAMPLE_2 = """\
Initial state (JSON):
{"at": {"A0": "L0-0", "P0": "L0-1", "T0": "L0-2", "T1": "L1-2"}, "in": {"P0": null}, "goal_satisfied": [], "goal_remaining": ["P0 at L1-1"]}

Goal state (JSON):
{"at": {"P0": "L1-1"}, "in": {"P0": null}}

STATE_JSON: ```json
{"at": {"A0": "L0-0", "P0": "L0-1", "T0": "L0-2", "T1": "L1-2"}, "in": {"P0": null}, "goal_satisfied": [], "goal_remaining": ["P0 at L1-1"]}
```

P0 must cross from C0 to C1. Use T0 to bring P0 to L0-0, fly A0 to L1-0, use T1 to deliver to L1-1.

ACTION: drive-truck T0 L0-2 L0-1 C0
STATE_JSON: ```json
{"at": {"A0": "L0-0", "P0": "L0-1", "T0": "L0-1", "T1": "L1-2"}, "in": {"P0": null}, "goal_satisfied": [], "goal_remaining": ["P0 at L1-1"]}
```

ACTION: load-truck P0 T0 L0-1
STATE_JSON: ```json
{"at": {"A0": "L0-0", "P0": null, "T0": "L0-1", "T1": "L1-2"}, "in": {"P0": "T0"}, "goal_satisfied": [], "goal_remaining": ["P0 at L1-1"]}
```

ACTION: drive-truck T0 L0-1 L0-0 C0
STATE_JSON: ```json
{"at": {"A0": "L0-0", "P0": null, "T0": "L0-0", "T1": "L1-2"}, "in": {"P0": "T0"}, "goal_satisfied": [], "goal_remaining": ["P0 at L1-1"]}
```

ACTION: unload-truck P0 T0 L0-0
STATE_JSON: ```json
{"at": {"A0": "L0-0", "P0": "L0-0", "T0": "L0-0", "T1": "L1-2"}, "in": {"P0": null}, "goal_satisfied": [], "goal_remaining": ["P0 at L1-1"]}
```

ACTION: load-airplane P0 A0 L0-0
STATE_JSON: ```json
{"at": {"A0": "L0-0", "P0": null, "T0": "L0-0", "T1": "L1-2"}, "in": {"P0": "A0"}, "goal_satisfied": [], "goal_remaining": ["P0 at L1-1"]}
```

ACTION: fly-airplane A0 L0-0 L1-0
STATE_JSON: ```json
{"at": {"A0": "L1-0", "P0": null, "T0": "L0-0", "T1": "L1-2"}, "in": {"P0": "A0"}, "goal_satisfied": [], "goal_remaining": ["P0 at L1-1"]}
```

ACTION: unload-airplane P0 A0 L1-0
STATE_JSON: ```json
{"at": {"A0": "L1-0", "P0": "L1-0", "T0": "L0-0", "T1": "L1-2"}, "in": {"P0": null}, "goal_satisfied": [], "goal_remaining": ["P0 at L1-1"]}
```

ACTION: drive-truck T1 L1-2 L1-0 C1
STATE_JSON: ```json
{"at": {"A0": "L1-0", "P0": "L1-0", "T0": "L0-0", "T1": "L1-0"}, "in": {"P0": null}, "goal_satisfied": [], "goal_remaining": ["P0 at L1-1"]}
```

ACTION: load-truck P0 T1 L1-0
STATE_JSON: ```json
{"at": {"A0": "L1-0", "P0": null, "T0": "L0-0", "T1": "L1-0"}, "in": {"P0": "T1"}, "goal_satisfied": [], "goal_remaining": ["P0 at L1-1"]}
```

ACTION: drive-truck T1 L1-0 L1-1 C1
STATE_JSON: ```json
{"at": {"A0": "L1-0", "P0": null, "T0": "L0-0", "T1": "L1-1"}, "in": {"P0": "T1"}, "goal_satisfied": [], "goal_remaining": ["P0 at L1-1"]}
```

ACTION: unload-truck P0 T1 L1-1
STATE_JSON: ```json
{"at": {"A0": "L1-0", "P0": "L1-1", "T0": "L0-0", "T1": "L1-1"}, "in": {"P0": null}, "goal_satisfied": ["P0 at L1-1"], "goal_remaining": []}
```

All goal conditions satisfied.

Plan summary:
1. drive-truck T0 L0-2 L0-1 C0
2. load-truck P0 T0 L0-1
3. drive-truck T0 L0-1 L0-0 C0
4. unload-truck P0 T0 L0-0
5. load-airplane P0 A0 L0-0
6. fly-airplane A0 L0-0 L1-0
7. unload-airplane P0 A0 L1-0
8. drive-truck T1 L1-2 L1-0 C1
9. load-truck P0 T1 L1-0
10. drive-truck T1 L1-0 L1-1 C1
11. unload-truck P0 T1 L1-1\
"""


def get_logistics_json_examples() -> list:
    return [LOGISTICS_JSON_EXAMPLE_1, LOGISTICS_JSON_EXAMPLE_2]


def get_logistics_json_goal_examples() -> list:
    return [LOGISTICS_JSON_GOAL_EXAMPLE_1, LOGISTICS_JSON_GOAL_EXAMPLE_2]
