from src.eval.blocksworld_aot_eval import evaluate_blocksworld_aot_output
from src.eval.blocksworld_json_structured_eval import (
    evaluate_blocksworld_json_structured_output,
)
from src.eval.blocksworld_json_goal_eval import (
    evaluate_blocksworld_json_goal_output,
)


# -------------------------
# Shared initial states
# -------------------------

INITIAL_STATE_TEXT = (
    "Block A is on Block B. "
    "Block B is on the table. "
    "Block C is on the table. "
    "Block A is clear. "
    "Block C is clear. "
    "The hand is empty."
)

INITIAL_STATE_JSON = {
    "on": {
        "A": "B",
        "B": "table",
        "C": "table",
    },
    "clear": ["A", "C"],
    "holding": None,
    "handempty": True,
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
        "ACTION: Unstack A from B\n"
        "STATE 1: Block B is on the table. Block C is on the table. "
        "Block B is clear. Block C is clear. The hand is holding Block A.\n\n"
        "ACTION: Put down A\n"
        "STATE 2: Block A is on the table. Block B is on the table. "
        "Block C is on the table. Block A is clear. Block B is clear. "
        "Block C is clear. The hand is empty.\n"
    )

    result = evaluate_blocksworld_aot_output(
        INITIAL_STATE_TEXT,
        raw_output,
    )

    print_result("AoT Perfect Trace", result)


# -------------------------
# JSON STRUCTURED TEST
# -------------------------

def run_json_structured_test():
    raw_output = (
        "ACTION: Unstack A from B\n"
        "REASONING: valid\n"
        "STATE_JSON:\n"
        "{\n"
        '  "on": {"B": "table", "C": "table"},\n'
        '  "clear": ["B", "C"],\n'
        '  "holding": "A",\n'
        '  "handempty": false\n'
        "}\n\n"
        "ACTION: Put down A\n"
        "REASONING: valid\n"
        "STATE_JSON:\n"
        "{\n"
        '  "on": {"A": "table", "B": "table", "C": "table"},\n'
        '  "clear": ["A", "B", "C"],\n'
        '  "holding": null,\n'
        '  "handempty": true\n'
        "}"
    )

    result = evaluate_blocksworld_json_structured_output(
        INITIAL_STATE_JSON,
        raw_output,
    )

    print_result("JSON Structured Perfect Trace", result)


# -------------------------
# JSON GOAL TEST
# -------------------------

def run_json_goal_test():
    raw_output = (
        "ACTION: Unstack A from B\n"
        "REASONING: valid\n"
        "STATE_JSON:\n"
        "{\n"
        '  "on": {"B": "table", "C": "table"},\n'
        '  "clear": ["B", "C"],\n'
        '  "holding": "A",\n'
        '  "handempty": false,\n'
        '  "goal_satisfied": [],\n'
        '  "goal_remaining": ["A on table"]\n'
        "}\n\n"
        "ACTION: Put down A\n"
        "REASONING: valid\n"
        "STATE_JSON:\n"
        "{\n"
        '  "on": {"A": "table", "B": "table", "C": "table"},\n'
        '  "clear": ["A", "B", "C"],\n'
        '  "holding": null,\n'
        '  "handempty": true,\n'
        '  "goal_satisfied": ["A on table"],\n'
        '  "goal_remaining": []\n'
        "}"
    )

    result = evaluate_blocksworld_json_goal_output(
        INITIAL_STATE_JSON,
        raw_output,
    )

    print_result("JSON Goal Perfect Trace", result)


# -------------------------
# MAIN
# -------------------------

if __name__ == "__main__":
    print("\nRunning Blocksworld Evaluator Tests...")

    run_aot_test()
    run_json_structured_test()
    run_json_goal_test()

    print("\nAll tests executed.\n")