from src.eval.blocksworld_aot_eval import evaluate_blocksworld_aot_output


INITIAL_STATE_TEXT = (
    "Block A is on Block B. "
    "Block B is on the table. "
    "Block C is on the table. "
    "Block A is clear. "
    "Block C is clear. "
    "The hand is empty."
)


def test_aot_perfect_trace():
    raw_output = (
        "ACTION: Unstack A from B\n"
        "STATE 1: Block B is on the table. Block C is on the table. "
        "Block B is clear. Block C is clear. The hand is holding Block A.\n\n"
        "ACTION: Put down A\n"
        "STATE 2: Block A is on the table. Block B is on the table. "
        "Block C is on the table. Block A is clear. Block B is clear. "
        "Block C is clear. The hand is empty.\n\n"
        "Plan summary:\n"
        "Done"
    )

    result = evaluate_blocksworld_aot_output(INITIAL_STATE_TEXT, raw_output)

    assert result.step_errors == [0, 0]
    assert result.depth_errors == [0, 0]
    assert result.first_error_step is None


def test_aot_omission_error():
    raw_output = (
        "ACTION: Unstack A from B\n"
        "STATE 1: Block B is on the table. Block C is on the table. "
        "Block C is clear. The hand is holding Block A.\n\n"
        "ACTION: Put down A\n"
        "STATE 2: Block A is on the table. Block B is on the table. "
        "Block C is on the table. Block A is clear. Block C is clear. "
        "The hand is empty.\n\n"
        "Plan summary:\n"
        "Done"
    )

    result = evaluate_blocksworld_aot_output(INITIAL_STATE_TEXT, raw_output)

    assert result.step_errors == [1, 1]
    assert result.depth_errors == [1, 1]
    assert result.first_error_step == 1

    assert "omission" in result.details[0]["error_types"]
    assert "accumulation" in result.details[1]["error_types"]


def test_aot_hallucination_error():
    raw_output = (
        "ACTION: Unstack A from B\n"
        "STATE 1: Block B is on the table. Block C is on the table. "
        "Block B is clear. The hand is holding Block A.\n\n"
        "ACTION: Put down A\n"
        "STATE 2: Block A is on the table. Block B is on the table. "
        "Block C is on the table. Block A is clear. Block B is clear. "
        "Block C is clear. The hand is empty.\n\n"
        "Plan summary:\n"
        "Done"
    )

    result = evaluate_blocksworld_aot_output(INITIAL_STATE_TEXT, raw_output)

    assert result.step_errors == [1, 0]
    assert result.depth_errors == [1, 1]
    assert result.first_error_step == 1

    assert "hallucination" in result.details[0]["error_types"]


def test_aot_parse_failure():
    raw_output = (
        "ACTION: Unstack A from B\n"
        "STATE 1: this is not a valid blocksworld state\n\n"
        "Plan summary:\n"
        "Done"
    )

    result = evaluate_blocksworld_aot_output(INITIAL_STATE_TEXT, raw_output)

    assert result.step_errors == [1]
    assert result.depth_errors == [1]
    assert result.first_error_step == 1
    assert "state_parse_failure" in result.details[0]["error_types"]