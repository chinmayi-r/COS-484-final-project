from src.eval.logistics_aot_eval import evaluate_logistics_aot_output


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


def test_aot_perfect_trace():
    raw_output = (
        "ACTION: load-truck P0 T0 L0-0\n"
        "STATE 1: Package P0 is in Truck T0. Truck T0 is at L0-0. Airplane A0 is at L1-0.\n\n"
        "ACTION: unload-truck P0 T0 L0-0\n"
        "STATE 2: Package P0 is at L0-0. Truck T0 is at L0-0. Airplane A0 is at L1-0.\n\n"
        "Plan summary:\n"
        "Done"
    )

    result = evaluate_logistics_aot_output(
        INITIAL_STATE_TEXT,
        CITY_MAP,
        raw_output,
    )

    assert result.step_errors == [0, 0]
    assert result.depth_errors == [0, 0]
    assert result.first_error_step is None


def test_aot_omission_error():
    raw_output = (
        "ACTION: load-truck P0 T0 L0-0\n"
        "STATE 1: Truck T0 is at L0-0. Airplane A0 is at L1-0.\n\n"
        "ACTION: unload-truck P0 T0 L0-0\n"
        "STATE 2: Truck T0 is at L0-0. Airplane A0 is at L1-0.\n\n"
        "Plan summary:\n"
        "Done"
    )

    result = evaluate_logistics_aot_output(
        INITIAL_STATE_TEXT,
        CITY_MAP,
        raw_output,
    )

    assert result.step_errors == [1, 1]
    assert result.depth_errors == [1, 1]
    assert result.first_error_step == 1

    assert "omission" in result.details[0]["error_types"]
    assert "accumulation" in result.details[1]["error_types"]


def test_aot_hallucination_error():
    raw_output = (
        "ACTION: load-truck P0 T0 L0-0\n"
        "STATE 1: Package P0 is in Truck T0. Truck T0 is at L0-0. Airplane A0 is at L1-0. Truck T1 is at L0-1.\n\n"
        "ACTION: unload-truck P0 T0 L0-0\n"
        "STATE 2: Package P0 is at L0-0. Truck T0 is at L0-0. Airplane A0 is at L1-0.\n\n"
        "Plan summary:\n"
        "Done"
    )

    result = evaluate_logistics_aot_output(
        INITIAL_STATE_TEXT,
        CITY_MAP,
        raw_output,
    )

    assert result.step_errors == [1, 0]
    assert result.depth_errors == [1, 1]
    assert result.first_error_step == 1

    assert "hallucination" in result.details[0]["error_types"]


def test_aot_parse_failure():
    raw_output = (
        "ACTION: load-truck P0 T0 L0-0\n"
        "STATE 1: this is not a valid logistics state\n\n"
        "Plan summary:\n"
        "Done"
    )

    result = evaluate_logistics_aot_output(
        INITIAL_STATE_TEXT,
        CITY_MAP,
        raw_output,
    )

    assert result.step_errors == [1]
    assert result.depth_errors == [1]
    assert result.first_error_step == 1
    assert "state_parse_failure" in result.details[0]["error_types"]