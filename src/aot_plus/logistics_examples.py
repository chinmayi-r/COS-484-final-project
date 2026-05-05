"""
Logistics few-shot examples for AoT+ JSON extension.

Format matches the paper (Algorithm of Thoughts, Appendix B.4) EXCEPT:
  - State is JSON (instead of compact NL checkpoint summaries)
  - Checkpoints at natural phase boundaries (end of each package sub-task)
  - Each STATE_JSON includes goal_satisfied / goal_remaining fields

Reasoning text is kept verbatim from B.4.  The three problems are the SAME
as the paper examples, enabling a controlled comparison where only the state
representation format differs.

City map:
  Examples 1/2: three cities  C0 (L0-x), C1 (L1-x), C2 (L2-x)
  Example 3:    two cities    C0 (L0-x), C1 (L1-x)
Airports are always Lx-0.
"""

# ---------------------------------------------------------------------------
# Example 1  (Paper B.4 Ex1)
# 3 cities, 2 airplanes, 3 trucks, 6 packages, 28 actions
#
# Initial:
#   A0 at L1-0, A1 at L1-0
#   T0 at L0-1, T1 at L1-0, T2 at L2-1
#   P0 at L2-0, P1 at L2-1, P2 at L1-1, P3 at L0-0, P4 at L2-0, P5 at L2-0
# Goal: P0@L2-0(noop), P1@L0-1, P2@L1-0, P3@L2-0, P4@L1-0, P5@L1-1
#
# Checkpoints (phase boundaries):
#   After step  4: P2 delivered intra-city
#   After step 15: P1 delivered cross-city to L0-1
#   After step 18: P3 delivered to L2-0
#   After step 21: P4 delivered to L1-0
#   After step 28: P5 delivered to L1-1  (final)
# ---------------------------------------------------------------------------

LOGISTICS_JSON_EXAMPLE_1 = """\
Initial state (JSON):
{"at": {"A0": "L1-0", "A1": "L1-0", "P0": "L2-0", "P1": "L2-1", "P2": "L1-1", "P3": "L0-0", "P4": "L2-0", "P5": "L2-0", "T0": "L0-1", "T1": "L1-0", "T2": "L2-1"}, "in": {"P0": null, "P1": null, "P2": null, "P3": null, "P4": null, "P5": null}}

Goal state (JSON):
{"at": {"P0": "L2-0", "P1": "L0-1", "P2": "L1-0", "P3": "L2-0", "P4": "L1-0", "P5": "L1-1"}, "in": {"P0": null, "P1": null, "P2": null, "P3": null, "P4": null, "P5": null}}

STATE_JSON: ```json
{"at": {"A0": "L1-0", "A1": "L1-0", "P0": "L2-0", "P1": "L2-1", "P2": "L1-1", "P3": "L0-0", "P4": "L2-0", "P5": "L2-0", "T0": "L0-1", "T1": "L1-0", "T2": "L2-1"}, "in": {"P0": null, "P1": null, "P2": null, "P3": null, "P4": null, "P5": null}}
```

There are three cities since only locations talked about are L0, L1 and L2. We have two airplanes A0 and A1, and are both in L1-0. Let's first examine which packages need to moved to other cities.

P0: L2 -> L2 (no)
P1: L2 -> L0 (yes)
P2: L1 -> L1 (no)
P3: L0 -> L2 (yes)
P4: L2 -> L1 (yes)
P5: L2 -> L1 (yes)

The packages P0 and P2 only need to moved within the city, and P1, P3, P4 and P5 will need to be transferred via planes to other cities. Let's sort out the within city ones first. P0 needs to be moved from L2-0 to L2-0, so no moves are necessary there. P2 should be moved from L1-1 to L1-0. The truck at C1 is T1 at L1-0, so it needs to go to L1-1 first and go to L1-0 after loading the package.

ACTION: drive-truck T1 L1-0 L1-1 C1
ACTION: load-truck P2 T1 L1-1
ACTION: drive-truck T1 L1-1 L1-0 C1
ACTION: unload-truck P2 T1 L1-0
STATE_JSON: ```json
{"at": {"A0": "L1-0", "A1": "L1-0", "P0": "L2-0", "P1": "L2-1", "P2": "L1-0", "P3": "L0-0", "P4": "L2-0", "P5": "L2-0", "T0": "L0-1", "T1": "L1-0", "T2": "L2-1"}, "in": {"P0": null, "P1": null, "P2": null, "P3": null, "P4": null, "P5": null}, "goal_satisfied": ["P0 at L2-0", "P2 at L1-0"], "goal_remaining": ["P1 at L0-1", "P3 at L2-0", "P4 at L1-0", "P5 at L1-1"]}
```

Now we are finished with intra-city transports and are left with the packages P1, P3, P4 and P5. Let's transfer P1 first. P1 needs to go from L2-1 to L0-1. Both locations are not airports so we need to first use a truck to carry it from L2-1 to L2-0 and then airplane to take it to L0-0, then finally another truck to take it from L0-0 to L0-1. The truck in L2 is T2 and it is in L2-1. So we can directly load it to take it to L2-0. The airplanes are in L1-0, so one of them (let's say A0) needs to fly to L2-0. The truck in C0 is T0 and it is in L0-1 so it also needs to go to L0-0 to be loaded with P1.

ACTION: load-truck P1 T2 L2-1
ACTION: drive-truck T2 L2-1 L2-0 C2
ACTION: unload-truck P1 T2 L2-0
ACTION: fly-airplane A0 L1-0 L2-0
ACTION: load-airplane P1 A0 L2-0
ACTION: fly-airplane A0 L2-0 L0-0
ACTION: unload-airplane P1 A0 L0-0
ACTION: drive-truck T0 L0-1 L0-0 C0
ACTION: load-truck P1 T0 L0-0
ACTION: drive-truck T0 L0-0 L0-1 C0
ACTION: unload-truck P1 T0 L0-1
STATE_JSON: ```json
{"at": {"A0": "L0-0", "A1": "L1-0", "P0": "L2-0", "P1": "L0-1", "P2": "L1-0", "P3": "L0-0", "P4": "L2-0", "P5": "L2-0", "T0": "L0-1", "T1": "L1-0", "T2": "L2-0"}, "in": {"P0": null, "P1": null, "P2": null, "P3": null, "P4": null, "P5": null}, "goal_satisfied": ["P0 at L2-0", "P1 at L0-1", "P2 at L1-0"], "goal_remaining": ["P3 at L2-0", "P4 at L1-0", "P5 at L1-1"]}
```

Now, let's continue with P3. P3 needs to go from L0-0 to L2-0. The airplane A0 is already at L0-0, so we can directly load it to take it to L2-0 which is the final destination.

ACTION: load-airplane P3 A0 L0-0
ACTION: fly-airplane A0 L0-0 L2-0
ACTION: unload-airplane P3 A0 L2-0
STATE_JSON: ```json
{"at": {"A0": "L2-0", "A1": "L1-0", "P0": "L2-0", "P1": "L0-1", "P2": "L1-0", "P3": "L2-0", "P4": "L2-0", "P5": "L2-0", "T0": "L0-1", "T1": "L1-0", "T2": "L2-0"}, "in": {"P0": null, "P1": null, "P2": null, "P3": null, "P4": null, "P5": null}, "goal_satisfied": ["P0 at L2-0", "P1 at L0-1", "P2 at L1-0", "P3 at L2-0"], "goal_remaining": ["P4 at L1-0", "P5 at L1-1"]}
```

Now, let's continue with P4. P4 needs to go from L2-0 to L1-0. The airplane A0 is already at L2-0, so we can directly load it to take it to L1-0 which is the final destination.

ACTION: load-airplane P4 A0 L2-0
ACTION: fly-airplane A0 L2-0 L1-0
ACTION: unload-airplane P4 A0 L1-0
STATE_JSON: ```json
{"at": {"A0": "L1-0", "A1": "L1-0", "P0": "L2-0", "P1": "L0-1", "P2": "L1-0", "P3": "L2-0", "P4": "L1-0", "P5": "L2-0", "T0": "L0-1", "T1": "L1-0", "T2": "L2-0"}, "in": {"P0": null, "P1": null, "P2": null, "P3": null, "P4": null, "P5": null}, "goal_satisfied": ["P0 at L2-0", "P1 at L0-1", "P2 at L1-0", "P3 at L2-0", "P4 at L1-0"], "goal_remaining": ["P5 at L1-1"]}
```

Finally, let's move P5. P5 needs to go from L2-0 to L1-1. The airplane A0 is at L1-0, so we need to fly it to L2-0 to load P5. Then, we need to fly it back to L1-0 and use the truck T1 to take it to L1-1. The truck in C1 is at L1-0, so it can directly be used to take P5 to the final destination L1-1.

ACTION: fly-airplane A0 L1-0 L2-0
ACTION: load-airplane P5 A0 L2-0
ACTION: fly-airplane A0 L2-0 L1-0
ACTION: unload-airplane P5 A0 L1-0
ACTION: load-truck P5 T1 L1-0
ACTION: drive-truck T1 L1-0 L1-1 C1
ACTION: unload-truck P5 T1 L1-1
STATE_JSON: ```json
{"at": {"A0": "L1-0", "A1": "L1-0", "P0": "L2-0", "P1": "L0-1", "P2": "L1-0", "P3": "L2-0", "P4": "L1-0", "P5": "L1-1", "T0": "L0-1", "T1": "L1-1", "T2": "L2-0"}, "in": {"P0": null, "P1": null, "P2": null, "P3": null, "P4": null, "P5": null}, "goal_satisfied": ["P0 at L2-0", "P1 at L0-1", "P2 at L1-0", "P3 at L2-0", "P4 at L1-0", "P5 at L1-1"], "goal_remaining": []}
```

All packages are at their goal locations.

Plan summary:
1. drive-truck T1 L1-0 L1-1 C1
2. load-truck P2 T1 L1-1
3. drive-truck T1 L1-1 L1-0 C1
4. unload-truck P2 T1 L1-0
5. load-truck P1 T2 L2-1
6. drive-truck T2 L2-1 L2-0 C2
7. unload-truck P1 T2 L2-0
8. fly-airplane A0 L1-0 L2-0
9. load-airplane P1 A0 L2-0
10. fly-airplane A0 L2-0 L0-0
11. unload-airplane P1 A0 L0-0
12. drive-truck T0 L0-1 L0-0 C0
13. load-truck P1 T0 L0-0
14. drive-truck T0 L0-0 L0-1 C0
15. unload-truck P1 T0 L0-1
16. load-airplane P3 A0 L0-0
17. fly-airplane A0 L0-0 L2-0
18. unload-airplane P3 A0 L2-0
19. load-airplane P4 A0 L2-0
20. fly-airplane A0 L2-0 L1-0
21. unload-airplane P4 A0 L1-0
22. fly-airplane A0 L1-0 L2-0
23. load-airplane P5 A0 L2-0
24. fly-airplane A0 L2-0 L1-0
25. unload-airplane P5 A0 L1-0
26. load-truck P5 T1 L1-0
27. drive-truck T1 L1-0 L1-1 C1
28. unload-truck P5 T1 L1-1"""


# ---------------------------------------------------------------------------
# Example 2  (Paper B.4 Ex2)
# 3 cities, 2 airplanes, 3 trucks, 5 packages, 21 actions
#
# Initial:
#   A0 at L2-0, A1 at L2-0
#   T0 at L0-0, T1 at L1-0, T2 at L2-0
#   P0 at L0-0, P1 at L2-0, P2 at L2-0, P3 at L1-2, P4 at L0-0
# Goal: P0@L0-0(noop), P1@L0-0, P2@L2-0(noop), P3@L2-1, P4@L2-2
#
# Checkpoints (phase boundaries):
#   After step  3: P1 delivered to L0-0
#   After step 14: P3 delivered to L2-1
#   After step 21: P4 delivered to L2-2  (final)
# ---------------------------------------------------------------------------

LOGISTICS_JSON_EXAMPLE_2 = """\
Initial state (JSON):
{"at": {"A0": "L2-0", "A1": "L2-0", "P0": "L0-0", "P1": "L2-0", "P2": "L2-0", "P3": "L1-2", "P4": "L0-0", "T0": "L0-0", "T1": "L1-0", "T2": "L2-0"}, "in": {"P0": null, "P1": null, "P2": null, "P3": null, "P4": null}}

Goal state (JSON):
{"at": {"P0": "L0-0", "P1": "L0-0", "P2": "L2-0", "P3": "L2-1", "P4": "L2-2"}, "in": {"P0": null, "P1": null, "P2": null, "P3": null, "P4": null}}

STATE_JSON: ```json
{"at": {"A0": "L2-0", "A1": "L2-0", "P0": "L0-0", "P1": "L2-0", "P2": "L2-0", "P3": "L1-2", "P4": "L0-0", "T0": "L0-0", "T1": "L1-0", "T2": "L2-0"}, "in": {"P0": null, "P1": null, "P2": null, "P3": null, "P4": null}}
```

There are three cities since only locations talked about are L0, L1 and L2. We have two airplanes A0 and A1, and both are in L2-0. Let's first examine which packages need to moved to other cities.

P0: L0 -> L0 (no)
P1: L2 -> L0 (yes)
P2: L2 -> L2 (no)
P3: L1 -> L2 (yes)
P4: L0 -> L2 (yes)

The packages P0 and P2 do not need to move. P1 needs to go from L2-0 to L0-0. Both are airports so A0 can directly fly P1 there. We should also start moving T1 toward P3 at L1-2 at the same time.

ACTION: load-airplane P1 A0 L2-0
ACTION: fly-airplane A0 L2-0 L0-0
ACTION: unload-airplane P1 A0 L0-0
STATE_JSON: ```json
{"at": {"A0": "L0-0", "A1": "L2-0", "P0": "L0-0", "P1": "L0-0", "P2": "L2-0", "P3": "L1-2", "P4": "L0-0", "T0": "L0-0", "T1": "L1-0", "T2": "L2-0"}, "in": {"P0": null, "P1": null, "P2": null, "P3": null, "P4": null}, "goal_satisfied": ["P0 at L0-0", "P1 at L0-0", "P2 at L2-0"], "goal_remaining": ["P3 at L2-1", "P4 at L2-2"]}
```

Now, let's continue with P3. P3 needs to go from L1-2 to L2-1. The truck T1 is now at L1-0 and needs to go to L1-2 to pick up P3. We need to bring P3 to L1-0 first, then fly it to L2-0 via airplane A1, then use truck T2 to deliver it to L2-1. Airplane A1 is at L2-0, so it needs to fly to L1-0. Truck T2 is at L2-0.

ACTION: drive-truck T1 L1-0 L1-2 C1
ACTION: load-truck P3 T1 L1-2
ACTION: drive-truck T1 L1-2 L1-0 C1
ACTION: unload-truck P3 T1 L1-0
ACTION: fly-airplane A1 L2-0 L1-0
ACTION: load-airplane P3 A1 L1-0
ACTION: fly-airplane A1 L1-0 L2-0
ACTION: unload-airplane P3 A1 L2-0
ACTION: load-truck P3 T2 L2-0
ACTION: drive-truck T2 L2-0 L2-1 C2
ACTION: unload-truck P3 T2 L2-1
STATE_JSON: ```json
{"at": {"A0": "L0-0", "A1": "L2-0", "P0": "L0-0", "P1": "L0-0", "P2": "L2-0", "P3": "L2-1", "P4": "L0-0", "T0": "L0-0", "T1": "L1-0", "T2": "L2-1"}, "in": {"P0": null, "P1": null, "P2": null, "P3": null, "P4": null}, "goal_satisfied": ["P0 at L0-0", "P1 at L0-0", "P2 at L2-0", "P3 at L2-1"], "goal_remaining": ["P4 at L2-2"]}
```

Finally, let's move P4. P4 needs to go from L0-0 to L2-2. The airplane A0 is already at L0-0, so we can directly load it to take it to L2-0. Then, we need to use the truck T2 to take it from L2-0 to L2-2. The truck in C2 is at L2-1, so it needs to go to L2-0 to be loaded with P4 to take it to the final destination L2-2.

ACTION: load-airplane P4 A0 L0-0
ACTION: fly-airplane A0 L0-0 L2-0
ACTION: unload-airplane P4 A0 L2-0
ACTION: drive-truck T2 L2-1 L2-0 C2
ACTION: load-truck P4 T2 L2-0
ACTION: drive-truck T2 L2-0 L2-2 C2
ACTION: unload-truck P4 T2 L2-2
STATE_JSON: ```json
{"at": {"A0": "L2-0", "A1": "L2-0", "P0": "L0-0", "P1": "L0-0", "P2": "L2-0", "P3": "L2-1", "P4": "L2-2", "T0": "L0-0", "T1": "L1-0", "T2": "L2-2"}, "in": {"P0": null, "P1": null, "P2": null, "P3": null, "P4": null}, "goal_satisfied": ["P0 at L0-0", "P1 at L0-0", "P2 at L2-0", "P3 at L2-1", "P4 at L2-2"], "goal_remaining": []}
```

All packages are at their goal locations.

Plan summary:
1. load-airplane P1 A0 L2-0
2. fly-airplane A0 L2-0 L0-0
3. unload-airplane P1 A0 L0-0
4. drive-truck T1 L1-0 L1-2 C1
5. load-truck P3 T1 L1-2
6. drive-truck T1 L1-2 L1-0 C1
7. unload-truck P3 T1 L1-0
8. fly-airplane A1 L2-0 L1-0
9. load-airplane P3 A1 L1-0
10. fly-airplane A1 L1-0 L2-0
11. unload-airplane P3 A1 L2-0
12. load-truck P3 T2 L2-0
13. drive-truck T2 L2-0 L2-1 C2
14. unload-truck P3 T2 L2-1
15. load-airplane P4 A0 L0-0
16. fly-airplane A0 L0-0 L2-0
17. unload-airplane P4 A0 L2-0
18. drive-truck T2 L2-1 L2-0 C2
19. load-truck P4 T2 L2-0
20. drive-truck T2 L2-0 L2-2 C2
21. unload-truck P4 T2 L2-2"""


# ---------------------------------------------------------------------------
# Example 3  (Paper B.4 Ex3)
# 2 cities, 1 airplane, 2 trucks, 4 packages, 30 actions
#
# Initial:
#   A0 at L1-0
#   T0 at L0-0, T1 at L1-1
#   P0 at L1-0, P1 at L1-1, P2 at L0-0, P3 at L0-1
# Goal: P0@L0-1, P1@L0-1, P2@L1-0, P3@L1-1
#
# Checkpoints (phase boundaries):
#   After step  6: P0 delivered to L0-1
#   After step 17: P1 delivered to L0-1
#   After step 20: P2 delivered to L1-0
#   After step 30: P3 delivered to L1-1  (final)
# ---------------------------------------------------------------------------

LOGISTICS_JSON_EXAMPLE_3 = """\
Initial state (JSON):
{"at": {"A0": "L1-0", "P0": "L1-0", "P1": "L1-1", "P2": "L0-0", "P3": "L0-1", "T0": "L0-0", "T1": "L1-1"}, "in": {"P0": null, "P1": null, "P2": null, "P3": null}}

Goal state (JSON):
{"at": {"P0": "L0-1", "P1": "L0-1", "P2": "L1-0", "P3": "L1-1"}, "in": {"P0": null, "P1": null, "P2": null, "P3": null}}

STATE_JSON: ```json
{"at": {"A0": "L1-0", "P0": "L1-0", "P1": "L1-1", "P2": "L0-0", "P3": "L0-1", "T0": "L0-0", "T1": "L1-1"}, "in": {"P0": null, "P1": null, "P2": null, "P3": null}}
```

There are two cities since only locations talked about are L0 and L1. We have one airplane A0, and it is in L1-0. Let's first examine which packages need to moved to other cities.

P0: L1 -> L0 (yes)
P1: L1 -> L0 (yes)
P2: L0 -> L1 (yes)
P3: L0 -> L1 (yes)

All packages need to be transferred via planes to other cities. Let's transfer P0 first. P0 needs to go from L1-0 to L0-1. The airplane A0 is already at L1-0, so we can directly load it to take it to L0-0. Then, we need to use the truck T0 to take it from L0-0 to L0-1. The truck in C0 is T0 and it's at L0-0. So, the package P0 can be directly loaded to T0 to be taken to the final destination L0-1.

ACTION: load-airplane P0 A0 L1-0
ACTION: fly-airplane A0 L1-0 L0-0
ACTION: unload-airplane P0 A0 L0-0
ACTION: load-truck P0 T0 L0-0
ACTION: drive-truck T0 L0-0 L0-1 C0
ACTION: unload-truck P0 T0 L0-1
STATE_JSON: ```json
{"at": {"A0": "L0-0", "P0": "L0-1", "P1": "L1-1", "P2": "L0-0", "P3": "L0-1", "T0": "L0-1", "T1": "L1-1"}, "in": {"P0": null, "P1": null, "P2": null, "P3": null}, "goal_satisfied": ["P0 at L0-1"], "goal_remaining": ["P1 at L0-1", "P2 at L1-0", "P3 at L1-1"]}
```

Now, let's continue with P1. P1 needs to go from L1-1 to L0-1. The truck in C1 is T1 and it's in L1-1 as the package, so we can directly load the package to T1. The airplane A0 is at L0-0, so we need to fly it to L1-0 to load P1. Then, we need to fly it back to L0-0 and use the truck T0 to take it to L0-1. The truck in C0 is T0 and it's in L0-1. So, it needs to be moved from L0-1 to L0-0 to load P1 and to be driven back to L0-1 which is the target destination for P1.

ACTION: load-truck P1 T1 L1-1
ACTION: drive-truck T1 L1-1 L1-0 C1
ACTION: unload-truck P1 T1 L1-0
ACTION: fly-airplane A0 L0-0 L1-0
ACTION: load-airplane P1 A0 L1-0
ACTION: fly-airplane A0 L1-0 L0-0
ACTION: unload-airplane P1 A0 L0-0
ACTION: drive-truck T0 L0-1 L0-0 C0
ACTION: load-truck P1 T0 L0-0
ACTION: drive-truck T0 L0-0 L0-1 C0
ACTION: unload-truck P1 T0 L0-1
STATE_JSON: ```json
{"at": {"A0": "L0-0", "P0": "L0-1", "P1": "L0-1", "P2": "L0-0", "P3": "L0-1", "T0": "L0-1", "T1": "L1-0"}, "in": {"P0": null, "P1": null, "P2": null, "P3": null}, "goal_satisfied": ["P0 at L0-1", "P1 at L0-1"], "goal_remaining": ["P2 at L1-0", "P3 at L1-1"]}
```

Now, let's continue with P2. P2 needs to go from L0-0 to L1-0. The airplane A0 is already at L0-0, so we can directly load it to take it to L1-0 which is the final destination.

ACTION: load-airplane P2 A0 L0-0
ACTION: fly-airplane A0 L0-0 L1-0
ACTION: unload-airplane P2 A0 L1-0
STATE_JSON: ```json
{"at": {"A0": "L1-0", "P0": "L0-1", "P1": "L0-1", "P2": "L1-0", "P3": "L0-1", "T0": "L0-1", "T1": "L1-0"}, "in": {"P0": null, "P1": null, "P2": null, "P3": null}, "goal_satisfied": ["P0 at L0-1", "P1 at L0-1", "P2 at L1-0"], "goal_remaining": ["P3 at L1-1"]}
```

Finally, let's move P3. P3 needs to go from L0-1 to L1-1. The truck in C0 is T0 and it's in L0-1 already. So, we can directly load P3 to T0 to transfer it to L0-0. The airplane A0 is in L1-0, so it needs to be transferred to L0-0 to load P3 to transfer it to L1-0. The truck in C1 is T1 and it's already in L1-0, so we can directly load P3 to it to carry it to the target destination L1-1.

ACTION: load-truck P3 T0 L0-1
ACTION: drive-truck T0 L0-1 L0-0 C0
ACTION: unload-truck P3 T0 L0-0
ACTION: fly-airplane A0 L1-0 L0-0
ACTION: load-airplane P3 A0 L0-0
ACTION: fly-airplane A0 L0-0 L1-0
ACTION: unload-airplane P3 A0 L1-0
ACTION: load-truck P3 T1 L1-0
ACTION: drive-truck T1 L1-0 L1-1 C1
ACTION: unload-truck P3 T1 L1-1
STATE_JSON: ```json
{"at": {"A0": "L1-0", "P0": "L0-1", "P1": "L0-1", "P2": "L1-0", "P3": "L1-1", "T0": "L0-0", "T1": "L1-1"}, "in": {"P0": null, "P1": null, "P2": null, "P3": null}, "goal_satisfied": ["P0 at L0-1", "P1 at L0-1", "P2 at L1-0", "P3 at L1-1"], "goal_remaining": []}
```

All packages are at their goal locations.

Plan summary:
1. load-airplane P0 A0 L1-0
2. fly-airplane A0 L1-0 L0-0
3. unload-airplane P0 A0 L0-0
4. load-truck P0 T0 L0-0
5. drive-truck T0 L0-0 L0-1 C0
6. unload-truck P0 T0 L0-1
7. load-truck P1 T1 L1-1
8. drive-truck T1 L1-1 L1-0 C1
9. unload-truck P1 T1 L1-0
10. fly-airplane A0 L0-0 L1-0
11. load-airplane P1 A0 L1-0
12. fly-airplane A0 L1-0 L0-0
13. unload-airplane P1 A0 L0-0
14. drive-truck T0 L0-1 L0-0 C0
15. load-truck P1 T0 L0-0
16. drive-truck T0 L0-0 L0-1 C0
17. unload-truck P1 T0 L0-1
18. load-airplane P2 A0 L0-0
19. fly-airplane A0 L0-0 L1-0
20. unload-airplane P2 A0 L1-0
21. load-truck P3 T0 L0-1
22. drive-truck T0 L0-1 L0-0 C0
23. unload-truck P3 T0 L0-0
24. fly-airplane A0 L1-0 L0-0
25. load-airplane P3 A0 L0-0
26. fly-airplane A0 L0-0 L1-0
27. unload-airplane P3 A0 L1-0
28. load-truck P3 T1 L1-0
29. drive-truck T1 L1-0 L1-1 C1
30. unload-truck P3 T1 L1-1"""


def get_logistics_json_examples() -> list:
    return [LOGISTICS_JSON_EXAMPLE_1, LOGISTICS_JSON_EXAMPLE_2, LOGISTICS_JSON_EXAMPLE_3]


def get_logistics_json_goal_examples() -> list:
    return get_logistics_json_examples()
