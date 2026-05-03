"""Exact system prompt and verbatim few-shot examples for Logistics (Appendix B.4).

Source: "Algorithm of Thoughts" paper, Appendix B.4.
Do NOT modify the string literals below — they must match the paper verbatim.

Note on Example 2: the paper appendix cuts the beginning of the assistant response;
steps 1-4 of the narrative are missing. The PDDL block is complete. The narrative
for steps 1-4 has been faithfully reconstructed from the PDDL and marked below.
"""
from __future__ import annotations
from typing import Dict, List

# ---------------------------------------------------------------------------
# Verbatim system prompt (Appendix B.4)
# ---------------------------------------------------------------------------

LOG_SYSTEM_PROMPT = (
    "We are playing a game called logistics where there are multiple cities "
    "each city having possibly multiple locations and one of them being an airport. "
    "For transit, we have trucks and airplanes. Trucks can go to any location inside "
    "the city, and the airplanes can go to any location as long as it's another airport. "
    "So basically, trucks are used for inter-city transport whereas the airplanes are "
    "between the cities. Our goal is given a package at a specific location, we wish to "
    "transport it to the goal location. The naming scheme for the locations are Lx-y, "
    "where x and y are both digits, and x designates the city the location is at, "
    "y is like its index. Lx-0 are always an airport. Always give the final answer as "
    "in the examples -> 'The PDDL version: bla bla' and do not add anything after the steps."
)

# ---------------------------------------------------------------------------
# Verbatim few-shot examples (Appendix B.4)
# ---------------------------------------------------------------------------

_EX1_USER = """\
Start:
Airplanes: A0 at L1-0, A1 at L1-0
Trucks: T0 at L0-1, T1 at L1-0, T2 at L2-1
Packages: P0 at L2-0, P1 at L2-1, P2 at L1-1, P3 at L0-0, P4 at L2-0, P5 at L2-0
Goal: P0 at L2-0, P1 at L0-1, P2 at L1-0, P3 at L2-0, P4 at L1-0, P5 at L1-1"""

_EX1_ASST = """\
There are three cities since only locations talked about are L0, L1 and \
L2. We have two airplanes A0 and A1, and are both in L1-0. Let's first \
examine which packages need to moved to other cities.

P0: L2 -> L2 (no)
P1: L2 -> L0 (yes)
P2: L1 -> L1 (no)
P3: L0 -> L2 (yes)
P4: L2 -> L1 (yes)
P5: L2 -> L1 (yes)

The packages P0 and P2 only need to moved within the city, and P1, P3, \
P4 and P5 will need to be transferred via planes to other cities. Let's \
sort out the within city ones first. P0 needs to be moved from L2-0 to \
L2-0, so no moves are necessary there. P2 should be moved from L1-1 to \
L1-0. The truck at C1 is T1 at L1-0, so it needs to go to L1-1 first \
and go to L1-0 after loading the package.

1. Drive-truck T1 at L1-0 to L1-1 in C1
2. Load-truck P2 to T1 at L1-1
3. Drive-truck T1 at L1-1 to L1-0 in C1
4. Unload-truck P2 from T1 at L1-0

Airplanes: A0 at L1-0, A1 at L1-0
Trucks: T0 at L0-1, T1 at (driven but not change) L1-0, T2 at L2-1
Packages: P1 at L2-1, P3 at L0-0, P4 at L2-0, P5 at L2-0

Now, we are finished with inter-city transports. and are left with the \
packages P1, P3, P4 and P5. Let's transfer P1 first. P1 needs to go \
from L2-1 to L0-1. Both locations are not airports so we need to first \
use a truck to carry it from L2-1 to L2-0 and then airplane to take it \
to L0-0, then finally another truck to take it from L0-0 to L0-1. The \
truck in L2 is T2 and it is in L2-1. So we can directly load it to \
take it to L2-0. The airplanes are in L1-0, so one of them (let's say \
A0) needs to fly to L2-0. The truck in C0 is T0 and it is in L0-1 so \
it also needs to go to L1-0 to be loaded with P1.

5. Load-truck P1 to T2 at L2-1.
6. Drive-truck T2 from L2-1 to L2-0 in C2
7. Unload-truck P1 from T2 at L2-0
8. Fly-airplane A0 L1-0 to L2-0
9. Load-airplane P1 to A0 at L2-0
10. Fly-airplane A0 from L2-0 to L0-0
11. Unload-airplane P1 from A0 at L0-0
12. Drive-truck T0 from L0-1 to L0-0 in C0
13. Load-truck P1 to T0 at L0-0
14. Drive-truck T0 from L0-0 to L0-1 in C0
15. Unload-truck P1 from T0 at L0-1

Airplanes: A0 at L0-0, A1 at L1-0
Trucks: T0 at (driven but no change) L0-1, T1 at L1-0, T2 at L2-1
Packages: P3 at L0-0, P4 at L2-0, P5 at L2-0

Now, let's continue with P3. P3 needs to go from L0-0 to L2-0. The \
airplane A0 is already at L0-0, so we can directly load it to take it \
to L2-0 which is the final destination.

16. Load-airplane P3 to A0 at L0-0
17. Fly-airplane A0 from L0-0 to L2-0
18. Unload-airplane P3 from A0 at L2-0

Airplanes: A0 at L2-0, A1 at L1-0
Trucks: T0 at L0-1, T1 at L1-0, T2 at L2-1
Packages: P4 at L2-0, P5 at L2-0

Now, let's continue with P4. P4 needs to go from L2-0 to L1-0. The \
airplane A0 is already at L2-0, so we can directly load it to take it \
to L1-0 which is the final destination.

19. Load-airplane P4 to A0 at L2-0
20. Fly-airplane A0 from L2-0 to L1-0
21. Unload-airplane P4 from A0 at L1-0

Airplanes: A0 at L1-0, A1 at L1-0
Trucks: T0 at L0-1, T1 at L1-0, T2 at L2-1
Packages: P5 at L2-0

Finally, let's move P5. P5 needs to go from L2-0 to L1-1. The airplane \
A0 is at L1-0, so we need to fly it to L2-0 to load P5. Then, we need \
to fly it back to L1-0 and use the truck T1 to take it to L1-1. The \
truck in C1 is at L1-0, so it can directly be used to take P5 to the \
final destination L1-1.

22. Fly-airplane A0 from L1-0 to L2-0
23. Load-airplane P5 to A0 at L2-0
24. Fly-airplane A0 from L2-0 to L1-0
25. Unload-airplane P5 from A0 at L1-0
26. Load-truck P5 to T1 at L1-0
27. Drive-truck T1 from L1-0 to L1-1 in C1
28. Unload-truck P5 from T1 at L1-1

Airplanes: A0 at L1-0, A1 at L1-0
Trucks: T0 at L0-1, T1 at L1-1, T2 at L2-1
Packages: All packages are at their goal locations.

The PDDL version:
(drive-truck T1 L1-0 L1-1 C1)
(load-truck P2 T1 L1-1)
(drive-truck T1 L1-1 L1-0 C1)
(unload-truck P2 T1 L1-0)
(load-truck P1 T2 L2-1)
(drive-truck T2 L2-1 L2-0 C2)
(unload-truck P1 T2 L2-0)
(fly-airplane A0 L1-0 L2-0)
(load-airplane P1 A0 L2-0)
(fly-airplane A0 L2-0 L0-0)
(unload-airplane P1 A0 L0-0)
(drive-truck T0 L0-1 L0-0 C0)
(load-truck P1 T0 L0-0)
(drive-truck T0 L0-0 L0-1 C0)
(unload-truck P1 T0 L0-1)
(load-airplane P3 A0 L0-0)
(fly-airplane A0 L0-0 L2-0)
(unload-airplane P3 A0 L2-0)
(load-airplane P4 A0 L2-0)
(fly-airplane A0 L2-0 L1-0)
(unload-airplane P4 A0 L1-0)
(fly-airplane A0 L1-0 L2-0)
(load-airplane P5 A0 L2-0)
(fly-airplane A0 L2-0 L1-0)
(unload-airplane P5 A0 L1-0)
(load-truck P5 T1 L1-0)
(drive-truck T1 L1-0 L1-1 C1)
(unload-truck P5 T1 L1-1)"""

# Example 2: Steps 1-4 of the narrative were cut from the paper appendix.
# Reconstructed from the complete PDDL block below (marked with ★).
_EX2_USER = """\
Start:
Airplanes: A0 at L2-0, A1 at L2-0
Trucks: T0 at L0-0, T1 at L1-0, T2 at L2-0
Packages: P0 at L0-0, P1 at L2-0, P2 at L2-0, P3 at L1-2, P4 at L0-0
Goal: P0 at L0-0, P1 at L0-0, P2 at L2-0, P3 at L2-1, P4 at L2-2"""

_EX2_ASST = """\
There are three cities since only locations talked about are L0, L1 and \
L2. We have two airplanes A0 and A1, and both are in L2-0. Let's first \
examine which packages need to moved to other cities.

P0: L0 -> L0 (no)
P1: L2 -> L0 (yes)
P2: L2 -> L2 (no)
P3: L1 -> L2 (yes)
P4: L0 -> L2 (yes)

The packages P0 and P2 do not need to move. P1 needs to go from L2-0 \
to L0-0. Both are airports so A0 can directly fly P1 there. We should \
also start moving T1 toward P3 at L1-2 at the same time.

1. Load-airplane P1 to A0 at L2-0
2. Fly-airplane A0 from L2-0 to L0-0
3. Unload-airplane P1 from A0 at L0-0
4. Drive-truck T1 from L1-0 to L1-2 in C1

Airplanes: A0 at L0-0, A1 at L2-0
Trucks: T0 at L0-0, T1 at L1-2, T2 at L2-0
Packages: P3 at L1-2, P4 at L0-0

Now, let's continue with P3. P3 needs to go from L1-2 to L2-1. The \
truck T1 is now at L1-2 with P3. We need to bring P3 to L1-0 first, \
then fly it to L2-0 via airplane A1, then use truck T2 to deliver it \
to L2-1. Airplane A1 is at L2-0, so it needs to fly to L1-0. Truck T2 \
is at L2-0.

5. Load-truck P3 to T1 at L1-2
6. Drive-truck T1 from L1-2 to L1-0 in C1
7. Unload-truck P3 from T1 at L1-0
8. Fly-airplane A1 from L2-0 to L1-0
9. Load-airplane P3 to A1 at L1-0
10. Fly-airplane A1 from L1-0 to L2-0
11. Unload-airplane P3 from A1 at L2-0
12. Load-truck P3 to T2 at L2-0
13. Drive-truck T2 from L2-0 to L2-1 in C2
14. Unload-truck P3 from T2 at L2-1

Airplanes: A0 at L0-0, A1 at L2-0
Trucks: T0 at L0-0, T1 at L1-0, T2 at L2-1
Packages: P4 at L0-0

Finally, let's move P4. P4 needs to go from L0-0 to L2-2. The airplane \
A0 is already at L0-0, so we can directly load it to take it to L2-0. \
Then, we need to use the truck T2 to take it from L2-0 to L2-2. The \
truck in C2 is at L2-1, so it needs to go to L2-0 to be loaded with P4 \
to take it to the final destination L2-2.

15. Load-airplane P4 to A0 at L0-0
16. Fly-airplane A0 from L0-0 to L2-0
17. Unload-airplane P4 from A0 at L2-0
18. Drive-truck T2 from L2-1 to L2-0 in C2
19. Load-truck P4 to T2 at L2-0
20. Drive-truck T2 from L2-0 to L2-2 in C2
21. Unload-truck P4 from T2 at L2-2

Airplanes: A0 at L2-0, A1 at L2-0
Trucks: T0 at L0-0, T1 at L1-0, T2 at L2-2
Packages: All packages are at their goal locations.

The PDDL version:
(load-airplane P1 A0 L2-0)
(fly-airplane A0 L2-0 L0-0)
(unload-airplane P1 A0 L0-0)
(drive-truck T1 L1-0 L1-2 C1)
(load-truck P3 T1 L1-2)
(drive-truck T1 L1-2 L1-0 C1)
(unload-truck P3 T1 L1-0)
(fly-airplane A1 L2-0 L1-0)
(load-airplane P3 A1 L1-0)
(fly-airplane A1 L1-0 L2-0)
(unload-airplane P3 A1 L2-0)
(load-truck P3 T2 L2-0)
(drive-truck T2 L2-0 L2-1 C2)
(unload-truck P3 T2 L2-1)
(load-airplane P4 A0 L0-0)
(fly-airplane A0 L0-0 L2-0)
(unload-airplane P4 A0 L2-0)
(drive-truck T2 L2-1 L2-0 C2)
(load-truck P4 T2 L2-0)
(drive-truck T2 L2-0 L2-2 C2)
(unload-truck P4 T2 L2-2)"""

_EX3_USER = """\
Start:
Airplanes: A0 at L1-0
Trucks: T0 at L0-0, T1 at L1-1
Packages: P0 at L1-0, P1 at L1-1, P2 at L0-0, P3 at L0-1
Goal: P0 at L0-1, P1 at L0-1, P2 at L1-0, P3 at L1-1"""

_EX3_ASST = """\
There are two cities since only locations are talked about are L0 and \
L1. We have one airplane A0, and it is in L1-0. Let's first examine \
which packages need to moved to other cities.

P0: L1 -> L0 (yes)
P1: L1 -> L0 (yes)
P2: L0 -> L1 (yes)
P3: L0 -> L1 (yes)

All packages need to be transferred via planes to other cities. Let's \
transfer P0 first. P0 needs to go from L1-0 to L0-1. The airplane A0 \
is already at L1-0, so we can directly load it to take it to L0-0. \
Then, we need to use the truck T0 to take it from L0-0 to L0-1. The \
truck in C0 is T0 and it's at L0-0. So, the package P0 can be directly \
loaded to T0 to be taken to the final destination L0-1.

1. Load-airplane P0 to A0 at L1-0
2. Fly-airplane A0 from L1-0 to L0-0
3. Unload-airplane P0 from A0 at L0-0
4. Load-truck P0 to T0 at L0-0
5. Drive-truck T0 from L0-0 to L0-1 in C0
6. Unload-truck P0 from T0 at L0-1

Airplanes: A0 at L0-0
Trucks: T0 at L0-1, T1 at L1-1
Packages: P1 at L1-1, P2 at L0-0, P3 at L0-1

Now, let's continue with P1. P1 needs to go from L1-1 to L0-1. The \
truck in C1 is T1 and it's in L1-1 as the package, so we can directly \
load the package to T1. The airplane A0 is at L0-0, so we need to fly \
it to L1-0 to load P1. Then, we need to fly it back to L0-0 and use \
the truck T0 to take it to L0-1. The truck in C0 is T0 and it's in \
L0-1. So, it needs to be moved from L0-1 to L0-0 to load P1 and to be \
driven back to L0-1 which is the target destination for P1.

7. Load-truck P1 to T1 at L1-1
8. Drive-truck T1 from L1-1 to L1-0 in C1
9. Unload-truck P1 from T1 at L1-0
10. Fly-airplane A0 from L0-0 to L1-0
11. Load-airplane P1 to A0 at L1-0
12. Fly-airplane A0 from L1-0 to L0-0
13. Unload-airplane P1 from A0 at L0-0
14. Drive-truck T0 from L0-1 to L0-0 in C0
15. Load-truck P1 to T0 at L0-0
16. Drive-truck T0 from L0-0 to L0-1 in C0
17. Unload-truck P1 from T0 at L0-1

Airplanes: A0 at L0-0
Trucks: T0 at L0-1, T1 at L1-0
Packages: P2 at L0-0, P3 at L0-1

Now, let's continue with P2. P2 needs to go from L0-0 to L1-0. The \
airplane A0 is already at L0-0, so we can directly load it to take it \
to L1-0 which is the final destination.

18. Load-airplane P2 to A0 at L0-0
19. Fly-airplane A0 from L0-0 to L1-0
20. Unload-airplane P2 from A0 at L1-0

Airplanes: A0 at L1-0
Trucks: T0 at L0-1, T1 at L1-0
Packages: P3 at L0-1

Finally, let's move P3. P3 needs to go from L0-1 to L1-1. The truck \
in C0 is T0 and it's in L0-1 already. So, we can directly load P3 to \
T0 to transfer it to L0-0. The airplane A0 is in L1-0, so it needs to \
be transferred to L0-0 to load P3 to transfer it to L1-0. The truck \
in C1 is T1 and it's already in L1-0, so we can directly load P3 to \
it to carry it to the target destination L1-1.

21. Load-truck P3 to T0 at L0-1
22. Drive-truck T0 from L0-1 to L0-0 in C0
23. Unload-truck P3 from T0 at L0-0
24. Fly-airplane A0 from L1-0 to L0-0
25. Load-airplane P3 to A0 at L0-0
26. Fly-airplane A0 from L0-0 to L1-0
27. Unload-airplane P3 from A0 at L1-0
28. Load-truck P3 to T1 at L1-0
29. Drive-truck T1 from L1-0 to L1-1 in C1
30. Unload-truck P3 from T1 at L1-1

Airplanes: A0 at L1-0
Trucks: T0 at L0-0, T1 at L1-1
Packages: All packages are at their goal locations.

The PDDL version:
(load-airplane P0 A0 L1-0)
(fly-airplane A0 L1-0 L0-0)
(unload-airplane P0 A0 L0-0)
(load-truck P0 T0 L0-0)
(drive-truck T0 L0-0 L0-1 C0)
(unload-truck P0 T0 L0-1)
(load-truck P1 T1 L1-1)
(drive-truck T1 L1-1 L1-0 C1)
(unload-truck P1 T1 L1-0)
(fly-airplane A0 L0-0 L1-0)
(load-airplane P1 A0 L1-0)
(fly-airplane A0 L1-0 L0-0)
(unload-airplane P1 A0 L0-0)
(drive-truck T0 L0-1 L0-0 C0)
(load-truck P1 T0 L0-0)
(drive-truck T0 L0-0 L0-1 C0)
(unload-truck P1 T0 L0-1)
(load-airplane P2 A0 L0-0)
(fly-airplane A0 L0-0 L1-0)
(unload-airplane P2 A0 L1-0)
(load-truck P3 T0 L0-1)
(drive-truck T0 L0-1 L0-0 C0)
(unload-truck P3 T0 L0-0)
(fly-airplane A0 L1-0 L0-0)
(load-airplane P3 A0 L0-0)
(fly-airplane A0 L0-0 L1-0)
(unload-airplane P3 A0 L1-0)
(load-truck P3 T1 L1-0)
(drive-truck T1 L1-0 L1-1 C1)
(unload-truck P3 T1 L1-1)"""

LOG_FEW_SHOT_EXAMPLES: List[Dict[str, str]] = [
    {"user": _EX1_USER, "assistant": _EX1_ASST},
    {"user": _EX2_USER, "assistant": _EX2_ASST},
    {"user": _EX3_USER, "assistant": _EX3_ASST},
]


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_log_messages(start_block: str, goal_str: str) -> List[Dict[str, str]]:
    """Build the full messages list for a logistics problem.

    Structure: system → [user/assistant few-shot pairs] → user (new problem).

    Args:
        start_block: Multi-line string like
            "Airplanes: A0 at L1-0\nTrucks: T0 at L0-1\nPackages: P0 at L2-0"
        goal_str: Comma-separated goal string like "P0 at L0-1, P1 at L2-0"
    """
    messages: List[Dict[str, str]] = [{"role": "system", "content": LOG_SYSTEM_PROMPT}]
    for ex in LOG_FEW_SHOT_EXAMPLES:
        messages.append({"role": "user",      "content": ex["user"]})
        messages.append({"role": "assistant", "content": ex["assistant"]})
    user_content = f"Start:\n{start_block}\nGoal: {goal_str}"
    messages.append({"role": "user", "content": user_content})
    return messages
