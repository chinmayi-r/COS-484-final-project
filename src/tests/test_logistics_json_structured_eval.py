from src.eval.logistics_json_structured_eval import (
    evaluate_logistics_json_structured_output,
)


CITY_MAP = {
    "L0-0": "C0",
    "L0-1": "C0",
    "L1-0": "C1",
}

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


def test_json_structured_perfect_trace():
    raw_output = (
        "ACTION: load-truck P0 T0 L0-0\n"
        "REASONING: P0 and T0 are both at L0-0.\n"
        "STATE_JSON:\n"
        "{\n"
        '  "at": {"P0": null, "T0": "L0-0", "A0": "L1-0"},\n'
        '  "in": {"P0": "T0"}\n'
        "}\n\n"
        "ACTION: unload-truck P0 T0 L0-0\n"
        "REASONING: unload P0 back to the same location.\n"
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

    assert result.step_errors == [0, 0]
    assert result.depth_errors == [0, 0]
    assert result.first_error_step is None


def test_json_structured_omission_error():
    raw_output = (
        "ACTION: load-truck P0 T0 L0-0\n"
        "REASONING: P0 and T0 are both at L0-0.\n"
        "STATE_JSON:\n"
        "{\n"
        '  "at": {"T0": "L0-0", "A0": "L1-0"},\n'
        '  "in": {"P0": null}\n'
        "}\n\n"
        "ACTION: unload-truck P0 T0 L0-0\n"
        "REASONING: unload P0 back to the same location.\n"
        "STATE_JSON:\n"
        "{\n"
        '  "at": {"T0": "L0-0", "A0": "L1-0"},\n'
        '  "in": {"P0": null}\n'
        "}"
    )

    result = evaluate_logistics_json_structured_output(
        INITIAL_STATE_JSON,
        CITY_MAP,
        raw_output,
    )

    assert result.step_errors == [1, 1]
    assert result.depth_errors == [1, 1]
    assert result.first_error_step == 1

    assert "omission" in result.details[0]["error_types"]
    assert "accumulation" in result.details[1]["error_types"]


def test_json_structured_hallucination_error():
    raw_output = (
        "ACTION: load-truck P0 T0 L0-0\n"
        "REASONING: P0 and T0 are both at L0-0.\n"
        "STATE_JSON:\n"
        "{\n"
        '  "at": {"P0": null, "T0": "L0-0", "A0": "L1-0", "T1": "L0-1"},\n'
        '  "in": {"P0": "T0"}\n'
        "}\n\n"
        "ACTION: unload-truck P0 T0 L0-0\n"
        "REASONING: unload P0 back to the same location.\n"
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

    assert result.step_errors == [1, 0]
    assert result.depth_errors == [1, 1]
    assert result.first_error_step == 1

    assert "hallucination" in result.details[0]["error_types"]


def test_json_structured_parse_failure():
    raw_output = (
        "ACTION: load-truck P0 T0 L0-0\n"
        "REASONING: P0 and T0 are both at L0-0.\n"
        "STATE_JSON:\n"
        "{\n"
        '  "at": {"P0": null, "T0": "L0-0", "A0": "L1-0"},\n'
        '  "in": {"P0": "T0"},\n'
    )

    result = evaluate_logistics_json_structured_output(
        INITIAL_STATE_JSON,
        CITY_MAP,
        raw_output,
    )

    assert result.step_errors == [1]
    assert result.depth_errors == [1]
    assert result.first_error_step == 1
    assert "state_parse_failure" in result.details[0]["error_types"]