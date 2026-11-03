from src.eval.logistics_aot_eval import evaluate_logistics_aot_output
from src.eval.logistics_json_structured_eval import (
    evaluate_logistics_json_structured_output,
)
from src.eval.logistics_json_goal_eval import (
    evaluate_logistics_json_goal_output,
)


# -------------------------
# Shared initial states
# -------------------------

CITY_MAP = {
    "L0-0": "C0",
    "L0-1": "C0",
    "L1-0": "C1",
}

INITIAL_STATE_TEXT = (
    "Package P0 is at L0-0. "
    "Truck T0 is at L0-0. "
    "Airplane A0 is at L1-0."
)

INITIAL_STATE_JSON = {
    "at": {
        "P0": "L0-0",
        "T0": "L0-0",
        "A0": "L1-0",
    },
    "in": {
        "P0": None,
    },
}


# -------------------------
# Helper printer
# -------------------------

def print_result(title, result):
    print(f"\n=== {title} ===")
    print("Step errors:", result.step_errors)
    print("Depth errors:", result.depth_errors)
    print("First error step:", result.first_error_step)

    for i, d in enumerate(result.details):
        print(f"\nStep {i+1}:")
        print("Error types:", d["error_types"])
        print("Match:", d["match"])
        print("Missing:", d.get("missing_predicates"))
        print("Extra:", d.get("extra_predicates"))


# -------------------------
# AoT TEST
# -------------------------

def run_aot_test():
    raw_output = (
        "ACTION: load-truck P0 T0 L0-0\n"
        "STATE 1: Package P0 is in Truck T0. Truck T0 is at L0-0. Airplane A0 is at L1-0.\n\n"
        "ACTION: unload-truck P0 T0 L0-0\n"
        "STATE 2: Package P0 is at L0-0. Truck T0 is at L0-0. Airplane A0 is at L1-0.\n"
    )

    result = evaluate_logistics_aot_output(
        INITIAL_STATE_TEXT,
        CITY_MAP,
        raw_output,
    )

    print_result("AoT Perfect Trace", result)


# -------------------------
# JSON STRUCTURED TEST
# -------------------------

def run_json_structured_test():
    raw_output = (
        "ACTION: load-truck P0 T0 L0-0\n"
        "REASONING: valid\n"
        "STATE_JSON:\n"
        "{\n"
        '  "at": {"P0": null, "T0": "L0-0", "A0": "L1-0"},\n'
        '  "in": {"P0": "T0"}\n'
        "}\n\n"
        "ACTION: unload-truck P0 T0 L0-0\n"
        "REASONING: valid\n"
        "STATE_JSON:\n"
        "{\n"
        '  "at": {"P0": "L0-0", "T0": "L0-0", "A0": "L1-0"},\n'
        '  "in": {"P0": null}\n'
        "}"
    )

    result = evaluate_logistics_json_structured_output(
        INITIAL_STATE_JSON,
        CITY_MAP,
        raw_output,
    )

    print_result("JSON Structured Perfect Trace", result)


# -------------------------
# JSON GOAL TEST
# -------------------------

def run_json_goal_test():
    raw_output = (
        "ACTION: load-truck P0 T0 L0-0\n"
        "REASONING: valid\n"
        "STATE_JSON:\n"
        "{\n"
        '  "at": {"P0": null, "T0": "L0-0", "A0": "L1-0"},\n'
        '  "in": {"P0": "T0"},\n'
        '  "goal_satisfied": [],\n'
        '  "goal_remaining": ["P0 at L1-0"]\n'
        "}\n\n"
        "ACTION: unload-truck P0 T0 L0-0\n"
        "REASONING: valid\n"
        "STATE_JSON:\n"
        "{\n"
        '  "at": {"P0": "L0-0", "T0": "L0-0", "A0": "L1-0"},\n'
        '  "in": {"P0": null},\n'
        '  "goal_satisfied": [],\n'
        '  "goal_remaining": ["P0 at L1-0"]\n'
        "}"
    )

    result = evaluate_logistics_json_goal_output(
        INITIAL_STATE_JSON,
        CITY_MAP,
        raw_output,
    )

    print_result("JSON Goal Perfect Trace", result)


# -------------------------
# MAIN
# -------------------------

if __name__ == "__main__":
    print("\nRunning Logistics Evaluator Tests...")

    run_aot_test()
    run_json_structured_test()
    run_json_goal_test()

    print("\nAll tests executed.\n")