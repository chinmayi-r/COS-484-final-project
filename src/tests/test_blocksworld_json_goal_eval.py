from src.eval.blocksworld_json_goal_eval import evaluate_blocksworld_json_goal_output


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


def test_json_goal_perfect_trace():
    raw_output = (
        "ACTION: Unstack A from B\n"
        "REASONING: A is on B and clear, so I can unstack it.\n"
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
        "REASONING: Put A on the table.\n"
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

    assert result.step_errors == [0, 0]
    assert result.depth_errors == [0, 0]
    assert result.first_error_step is None


def test_json_goal_missing_goal_fields():
    raw_output = (
        "ACTION: Unstack A from B\n"
        "REASONING: A is on B and clear, so I can unstack it.\n"
        "STATE_JSON:\n"
        "{\n"
        '  "on": {"B": "table", "C": "table"},\n'
        '  "clear": ["B", "C"],\n'
        '  "holding": "A",\n'
        '  "handempty": false\n'
        "}"
    )

    result = evaluate_blocksworld_json_goal_output(
        INITIAL_STATE_JSON,
        raw_output,
    )

    assert result.step_errors == [1]
    assert result.first_error_step == 1
    assert "goal_field_failure" in result.details[0]["error_types"]


def test_json_goal_omission_error():
    raw_output = (
        "ACTION: Unstack A from B\n"
        "REASONING: A is on B and clear, so I can unstack it.\n"
        "STATE_JSON:\n"
        "{\n"
        '  "on": {"B": "table", "C": "table"},\n'
        '  "clear": ["C"],\n'
        '  "holding": "A",\n'
        '  "handempty": false,\n'
        '  "goal_satisfied": [],\n'
        '  "goal_remaining": ["A on table"]\n'
        "}\n\n"
        "ACTION: Put down A\n"
        "REASONING: Put A on the table.\n"
        "STATE_JSON:\n"
        "{\n"
        '  "on": {"A": "table", "B": "table", "C": "table"},\n'
        '  "clear": ["A", "C"],\n'
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

    assert result.step_errors == [1, 1]
    assert result.first_error_step == 1

    assert "omission" in result.details[0]["error_types"]
    assert "accumulation" in result.details[1]["error_types"]


def test_json_goal_hallucination_error():
    raw_output = (
        "ACTION: Unstack A from B\n"
        "REASONING: A is on B and clear, so I can unstack it.\n"
        "STATE_JSON:\n"
        "{\n"
        '  "on": {"B": "table", "C": "table"},\n'
        '  "clear": ["B"],\n'
        '  "holding": "A",\n'
        '  "handempty": false,\n'
        '  "goal_satisfied": [],\n'
        '  "goal_remaining": ["A on table"]\n'
        "}\n\n"
        "ACTION: Put down A\n"
        "REASONING: Put A on the table.\n"
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

    assert result.step_errors == [1, 0]
    assert result.first_error_step == 1

    assert "hallucination" in result.details[0]["error_types"]