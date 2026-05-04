"""Focused tests for ALFWorld AoT+ state-tracking diagnostics.

Run from repo root:
    PYTHONPATH=. python -m unittest -v tests.test_alfworld_error_eval

These tests check two big things:

1. State diagnostic logic:
   After each ACTION, the evaluator predicts what the next state should be.
   Then it compares that expected state to the STATE written in the trace.

2. Trace parsing / branch selection:
   The trace may contain many attempted branches.
   The parser should keep only the final committed branch.
"""

import os
import sys
import textwrap
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.alfworld_error_eval import (
    build_alias_maps,
    clean_alias_base,
    extract_actions_and_states,
    episode_diagnostics,
    parse_state_header,
    prefix_chain,
    replace_ids,
)


# ------------------------------------------------------------------
# Small starting states used by many tests.
# These are intentionally tiny so each test is easy to reason about.
# ------------------------------------------------------------------

PICKUP_INIT = (
    "Agent A is at Kitchen, "
    "Drawer D1 is at Kitchen, "
    "Drawer D1 is openable, "
    "Drawer D1 is opened, "
    "Apple AP1 is in Drawer D1, "
    "Apple AP1 is pickupable, "
    "Agent A is holding nothing"
)

MOVE_INIT = (
    "Agent A is at LivingRoom, "
    "CounterTop CT1 is at Kitchen, "
    "CounterTop CT2 is at Bathroom, "
    "Agent A is holding nothing"
)

OPEN_INIT = (
    "Agent A is at Kitchen, "
    "Drawer D1 is at Kitchen, "
    "Drawer D1 is openable, "
    "Agent A is holding nothing"
)

PUT_INIT = (
    "Agent A is at Kitchen, "
    "CounterTop CT1 is at Kitchen, "
    "CounterTop CT1 can contain Apple, "
    "Agent A is holding Apple AP1"
)


class TestALFWorldAliasLayer(unittest.TestCase):
    """Short-ID aliases for ALFWorld PDDL object/location strings."""

    def test_clean_alias_base_locations_and_types(self):
        self.assertEqual(clean_alias_base("loc_bar__minus_00"), "loc")
        self.assertEqual(clean_alias_base("loc_other"), "loc")
        self.assertEqual(clean_alias_base("AppleType"), "Apple")
        self.assertEqual(clean_alias_base("DiningTableType"), "DiningTable")

    def test_apple_long_id_maps_to_apple_1(self):
        apple_long = "Apple_bar__minus_00_dot_99_bar__plus_00_dot_95"
        objects_by_type = {"AppleType": [apple_long]}
        atoms = [
            ["atLocation", "agent1", "loc_bar__minus_01"],
            ["objectType", apple_long, "AppleType"],
        ]
        o2a, a2o = build_alias_maps(objects_by_type, atoms)
        self.assertEqual(o2a[apple_long], "Apple_1")
        self.assertEqual(a2o["Apple_1"], apple_long)

    def test_dining_table_long_id_maps_to_dining_table_1(self):
        dt_long = "DiningTable_bar__minus_00_dot_12"
        objects_by_type = {"DiningTableType": [dt_long]}
        atoms = [
            ["atLocation", "agent1", "loc_bar__a"],
            ["receptacleType", dt_long, "DiningTableType"],
        ]
        o2a, _ = build_alias_maps(objects_by_type, atoms)
        self.assertEqual(o2a[dt_long], "DiningTable_1")

    def test_replace_ids_round_trip(self):
        apple_long = "Apple_bar__minus_00_dot_99"
        objects_by_type = {"AppleType": [apple_long]}
        atoms = [
            ["atLocation", "agent1", "loc_bar__x"],
            ["objectType", apple_long, "AppleType"],
        ]
        o2a, a2o = build_alias_maps(objects_by_type, atoms)
        nl = f"Agent agent1 is at loc_bar__x, Apple {apple_long} is pickupable"
        aliased = replace_ids(nl, o2a)
        self.assertIn("Apple_1", aliased)
        self.assertNotIn(apple_long, aliased)
        restored = replace_ids(aliased, a2o)
        self.assertEqual(restored, nl)

    def test_replace_ids_longest_first_safe(self):
        """Shorter keys must not truncate longer aliases (e.g. Apple_1 vs Apple_11)."""
        alias_to_orig = {"Apple_11": "LONG_ELEVEN", "Apple_1": "LONG_ONE"}
        text = "STATE 1: Apple_11 is pickupable, Apple_1 is sliced"
        out = replace_ids(text, alias_to_orig)
        self.assertIn("LONG_ELEVEN", out)
        self.assertIn("LONG_ONE", out)
        self.assertNotIn("Apple_11", out)
        self.assertNotIn("Apple_1", out)


class TestALFWorldErrorDiagnostics(unittest.TestCase):
    def run_trace(self, initial_state, trace, dynamic_predicates=None):
        """Parse a trace, run diagnostics, and return all useful outputs.

        This keeps each test short.
        """
        actions, states = extract_actions_and_states(textwrap.dedent(trace).strip())
        per_step, accumulation = episode_diagnostics(
            actions,
            states,
            initial_state,
            dynamic_predicates=dynamic_predicates,
        )
        return actions, states, per_step, accumulation

    def assert_no_errors(self, step, msg=None):
        """Check that one step is completely correct."""
        self.assertFalse(step["propagation_error"], msg=msg)
        self.assertFalse(step["accumulation_error"], msg=msg)
        self.assertFalse(step["omission_error"], msg=msg)
        self.assertFalse(step["hallucination_error"], msg=msg)
        self.assertEqual(step["omission_facts"], [], msg=msg)
        self.assertEqual(step["hallucination_facts"], [], msg=msg)
        self.assertEqual(step["divergence_facts"], [], msg=msg)

    def assert_only_omission(self, step, expected_facts):
        """Check that the model forgot facts, but did not invent facts."""
        self.assertTrue(step["omission_error"])
        self.assertFalse(step["hallucination_error"])
        self.assertTrue(set(expected_facts).issubset(set(step["omission_facts"])))

    def assert_only_hallucination(self, step, expected_facts):
        """Check that the model invented facts, but did not forget facts."""
        self.assertFalse(step["omission_error"])
        self.assertTrue(step["hallucination_error"])
        self.assertTrue(set(expected_facts).issubset(set(step["hallucination_facts"])))

    # ------------------------------------------------------------------
    # Correct traces.
    # These should produce no diagnostic errors.
    # ------------------------------------------------------------------

    def test_clean_pickup_has_no_errors(self):
        """A correct pickup should have no errors.

        Before: AP1 is in D1 and the agent holds nothing.
        Action: pick up AP1.
        After: the agent holds AP1 and AP1 is no longer in D1.
        """
        _, _, per_step, accumulation = self.run_trace(
            PICKUP_INIT,
            """
            ACTION: Pick up object AP1 from receptacle D1 at location Kitchen
            STATE 1: Agent A is at Kitchen, Agent A is holding Apple AP1, Drawer D1 is at Kitchen, Drawer D1 is openable, Drawer D1 is opened
            """,
        )

        self.assertEqual(len(per_step), 1)
        self.assert_no_errors(per_step[0])
        self.assertEqual(accumulation, [])

    def test_clean_open_receptacle_has_no_errors(self):
        """A correct open action should add the opened fact."""
        _, _, per_step, accumulation = self.run_trace(
            OPEN_INIT,
            """
            ACTION: Open receptacle D1 at location Kitchen
            STATE 1: Agent A is at Kitchen, Drawer D1 is at Kitchen, Drawer D1 is openable, Drawer D1 is opened, Agent A is holding nothing
            """,
        )

        self.assertEqual(len(per_step), 1)
        self.assert_no_errors(per_step[0])
        self.assertEqual(accumulation, [])

    def test_clean_two_step_move_then_pickup_has_no_errors(self):
        """A correct two-step path should have no errors.

        Step 1: move from LivingRoom to Kitchen.
        Step 2: pick up AP1 from D1.
        """
        initial = (
            "Agent A is at LivingRoom, "
            "Drawer D1 is at Kitchen, "
            "Drawer D1 is openable, "
            "Drawer D1 is opened, "
            "Apple AP1 is in Drawer D1, "
            "Apple AP1 is pickupable, "
            "Agent A is holding nothing"
        )
        _, _, per_step, accumulation = self.run_trace(
            initial,
            """
            ACTION: Go from location LivingRoom to location Kitchen
            STATE 1: Agent A is at Kitchen, Drawer D1 is at Kitchen, Drawer D1 is openable, Drawer D1 is opened, Apple AP1 is in Drawer D1, Apple AP1 is pickupable, Agent A is holding nothing

            ACTION: Pick up object AP1 from receptacle D1 at location Kitchen
            STATE 1.1: Agent A is at Kitchen, Agent A is holding Apple AP1, Drawer D1 is at Kitchen, Drawer D1 is openable, Drawer D1 is opened
            """,
        )

        self.assertEqual(len(per_step), 2)
        for step in per_step:
            self.assert_no_errors(step)
        self.assertEqual(accumulation, [])

    # ------------------------------------------------------------------
    # Omission tests.
    # Omission means: the expected fact is missing from the written state.
    # ------------------------------------------------------------------

    def test_omission_pickup_missing_holding_effects(self):
        """The model says pickup happened, but still says the agent holds nothing.

        Expected after pickup:
        - A holds AP1.
        - A holds some object.
        - AP1 is no longer in D1.

        The written state forgot these changes, so this is an omission error.
        """
        _, _, per_step, _ = self.run_trace(
            PICKUP_INIT,
            """
            ACTION: Pick up object AP1 from receptacle D1 at location Kitchen
            STATE 1: Agent A is at Kitchen, Agent A is holding nothing, Drawer D1 is at Kitchen, Drawer D1 is openable, Drawer D1 is opened
            """,
            dynamic_predicates=["holds", "holdsAny", "inReceptacle", "objectAtLocation"],
        )

        self.assertEqual(len(per_step), 1)
        self.assert_only_omission(
            per_step[0],
            {
                ("holds", "A", "AP1"),
                ("holdsAny", "A"),
            },
        )

    def test_omission_open_missing_opened_fact(self):
        """The model opens D1, but the next state does not say D1 is opened."""
        _, _, per_step, _ = self.run_trace(
            OPEN_INIT,
            """
            ACTION: Open receptacle D1 at location Kitchen
            STATE 1: Agent A is at Kitchen, Drawer D1 is at Kitchen, Drawer D1 is openable, Agent A is holding nothing
            """,
            dynamic_predicates=["opened"],
        )

        self.assertEqual(len(per_step), 1)
        self.assert_only_omission(per_step[0], {("opened", "D1")})

    def test_omission_move_missing_new_location(self):
        """The model moves to Kitchen, but the state still says LivingRoom."""
        _, _, per_step, _ = self.run_trace(
            MOVE_INIT,
            """
            ACTION: Go from location LivingRoom to location Kitchen
            STATE 1: Agent A is at LivingRoom, CounterTop CT1 is at Kitchen, CounterTop CT2 is at Bathroom, Agent A is holding nothing
            """,
            dynamic_predicates=["atLocation"],
        )

        self.assertEqual(len(per_step), 1)
        self.assert_only_omission(per_step[0], {("atLocation", "A", "Kitchen")})

    # ------------------------------------------------------------------
    # Hallucination tests.
    # Hallucination means: the written state invented an unexpected fact.
    # ------------------------------------------------------------------

    def test_hallucination_pickup_invents_sliced_fact(self):
        """Pickup should not make the apple sliced.

        The pickup itself is correct, but the state also invents "AP1 is sliced."
        """
        _, _, per_step, _ = self.run_trace(
            PICKUP_INIT,
            """
            ACTION: Pick up object AP1 from receptacle D1 at location Kitchen
            STATE 1: Agent A is at Kitchen, Agent A is holding Apple AP1, Drawer D1 is at Kitchen, Drawer D1 is openable, Drawer D1 is opened, Apple AP1 is sliced
            """,
            dynamic_predicates=["holds", "holdsAny", "inReceptacle", "objectAtLocation", "isSliced"],
        )

        self.assertEqual(len(per_step), 1)
        self.assert_only_hallucination(per_step[0], {("isSliced", "AP1")})

    def test_hallucination_open_invents_holding_fact(self):
        """Opening a drawer should not make the agent hold an apple."""
        _, _, per_step, _ = self.run_trace(
            OPEN_INIT,
            """
            ACTION: Open receptacle D1 at location Kitchen
            STATE 1: Agent A is at Kitchen, Agent A is holding Apple AP1, Drawer D1 is at Kitchen, Drawer D1 is openable, Drawer D1 is opened
            """,
            dynamic_predicates=["opened", "holds", "holdsAny"],
        )

        self.assertEqual(len(per_step), 1)
        self.assert_only_hallucination(per_step[0], {("holds", "A", "AP1")})

    # ------------------------------------------------------------------
    # Mixed error tests.
    # Mixed means: the state both forgets expected facts and invents wrong facts.
    # ------------------------------------------------------------------

    def test_wrong_move_destination_is_omission_and_hallucination(self):
        """Moving to the wrong place creates two errors.

        Action says: move to Kitchen.
        State says: agent is at Bathroom.

        Missing expected fact: agent at Kitchen.
        Invented wrong fact: agent at Bathroom.
        """
        _, _, per_step, _ = self.run_trace(
            MOVE_INIT,
            """
            ACTION: Go from location LivingRoom to location Kitchen
            STATE 1: Agent A is at Bathroom, CounterTop CT1 is at Kitchen, CounterTop CT2 is at Bathroom, Agent A is holding nothing
            """,
            dynamic_predicates=["atLocation"],
        )

        self.assertEqual(len(per_step), 1)
        step = per_step[0]
        self.assertTrue(step["omission_error"])
        self.assertTrue(step["hallucination_error"])
        self.assertIn(("atLocation", "A", "Kitchen"), step["omission_facts"])
        self.assertIn(("atLocation", "A", "Bathroom"), step["hallucination_facts"])

    def test_pickup_missing_holds_and_inventing_sliced_is_mixed(self):
        """This pickup state forgets one real effect and invents one fake effect.

        Missing: A should hold AP1.
        Invented: AP1 should not become sliced.
        """
        _, _, per_step, _ = self.run_trace(
            PICKUP_INIT,
            """
            ACTION: Pick up object AP1 from receptacle D1 at location Kitchen
            STATE 1: Agent A is at Kitchen, Agent A is holding nothing, Drawer D1 is at Kitchen, Drawer D1 is openable, Drawer D1 is opened, Apple AP1 is sliced
            """,
            dynamic_predicates=["holds", "holdsAny", "inReceptacle", "objectAtLocation", "isSliced"],
        )

        self.assertEqual(len(per_step), 1)
        step = per_step[0]
        self.assertTrue(step["omission_error"])
        self.assertTrue(step["hallucination_error"])
        self.assertIn(("holds", "A", "AP1"), step["omission_facts"])
        self.assertIn(("isSliced", "AP1"), step["hallucination_facts"])

    # ------------------------------------------------------------------
    # Propagation and accumulation tests.
    # These are about whether wrong state facts persist over time.
    # ------------------------------------------------------------------

    def test_propagation_marks_first_divergent_step(self):
        """The first wrong step is a propagation error, not accumulation yet.

        There is only one step.
        So the wrong Bathroom location first appears here.
        """
        _, _, per_step, accumulation = self.run_trace(
            MOVE_INIT,
            """
            ACTION: Go from location LivingRoom to location Kitchen
            STATE 1: Agent A is at Bathroom, CounterTop CT1 is at Kitchen, CounterTop CT2 is at Bathroom, Agent A is holding nothing
            """,
            dynamic_predicates=["atLocation"],
        )

        self.assertEqual(len(per_step), 1)
        self.assertTrue(per_step[0]["propagation_error"])
        self.assertFalse(per_step[0]["accumulation_error"])
        self.assertEqual(accumulation, [])

    def test_accumulation_marks_persistent_divergence_after_first_step(self):
        """A wrong fact that survives into the next step becomes accumulation.

        Step 1 wrongly says the agent is at Bathroom.
        Step 2 still says the agent is at Bathroom.
        The repeated wrong fact is accumulation.
        """
        _, _, per_step, accumulation = self.run_trace(
            MOVE_INIT,
            """
            ACTION: Go from location LivingRoom to location Kitchen
            STATE 1: Agent A is at Bathroom, CounterTop CT1 is at Kitchen, CounterTop CT2 is at Bathroom, Agent A is holding nothing

            ACTION: Go from location Kitchen to location Kitchen
            STATE 1.1: Agent A is at Bathroom, CounterTop CT1 is at Kitchen, CounterTop CT2 is at Bathroom, Agent A is holding nothing
            """,
            dynamic_predicates=["atLocation"],
        )

        self.assertEqual(len(per_step), 2)
        self.assertTrue(per_step[0]["propagation_error"])
        self.assertFalse(per_step[0]["accumulation_error"])
        self.assertTrue(per_step[1]["accumulation_error"])
        self.assertTrue(accumulation)
        self.assertIn(("atLocation", "A", "Bathroom"), accumulation)
        
    # ------------------------------------------------------------------
    # Extra domain/action coverage.
    # These cover the ALFWorld actions not tested above.
    # ------------------------------------------------------------------

    def test_clean_put_into_receptacle_has_no_errors(self):
        """A valid put should move the held object into the receptacle."""
        _, _, per_step, accumulation = self.run_trace(
            PUT_INIT,
            """
            ACTION: Put object AP1 into receptacle CT1 at location Kitchen
            STATE 1: Agent A is at Kitchen, Agent A is holding nothing, CounterTop CT1 is at Kitchen, Apple AP1 is in CounterTop CT1
            """,
            dynamic_predicates=["holds", "holdsAny", "inReceptacle", "objectAtLocation"],
        )

        self.assertEqual(len(per_step), 1)
        self.assert_no_errors(per_step[0])
        self.assertEqual(accumulation, [])

    def test_invalid_put_when_not_holding_object_noops(self):
        """Putting an object should do nothing if the agent is not holding it."""
        initial = (
            "Agent A is at Kitchen, "
            "CounterTop CT1 is at Kitchen, "
            "Apple AP1 is in CounterTop CT1, "
            "Agent A is holding nothing"
        )

        _, _, per_step, accumulation = self.run_trace(
            initial,
            """
            ACTION: Put object AP1 into receptacle CT1 at location Kitchen
            STATE 1: Agent A is at Kitchen, Agent A is holding nothing, CounterTop CT1 is at Kitchen, Apple AP1 is in CounterTop CT1
            """,
            dynamic_predicates=["holds", "holdsAny", "inReceptacle", "objectAtLocation"],
        )

        self.assertEqual(len(per_step), 1)
        self.assert_no_errors(per_step[0])
        self.assertEqual(accumulation, [])

    def test_clean_close_receptacle_has_no_errors(self):
        """A correct close action should remove the opened fact."""
        initial = (
            "Agent A is at Kitchen, "
            "Drawer D1 is at Kitchen, "
            "Drawer D1 is openable, "
            "Drawer D1 is opened, "
            "Agent A is holding nothing"
        )

        _, _, per_step, accumulation = self.run_trace(
            initial,
            """
            ACTION: Close receptacle D1 at location Kitchen
            STATE 1: Agent A is at Kitchen, Drawer D1 is at Kitchen, Drawer D1 is openable, Agent A is holding nothing
            """,
            dynamic_predicates=["opened"],
        )

        self.assertEqual(len(per_step), 1)
        self.assert_no_errors(per_step[0])
        self.assertEqual(accumulation, [])

    def test_invalid_pickup_from_closed_receptacle_noops(self):
        """Pickup from a closed openable receptacle should do nothing."""
        initial = (
            "Agent A is at Kitchen, "
            "Drawer D1 is at Kitchen, "
            "Drawer D1 is openable, "
            "Apple AP1 is in Drawer D1, "
            "Apple AP1 is pickupable, "
            "Agent A is holding nothing"
        )

        _, _, per_step, accumulation = self.run_trace(
            initial,
            """
            ACTION: Pick up object AP1 from receptacle D1 at location Kitchen
            STATE 1: Agent A is at Kitchen, Agent A is holding nothing, Drawer D1 is at Kitchen, Drawer D1 is openable, Apple AP1 is in Drawer D1, Apple AP1 is pickupable
            """,
            dynamic_predicates=["holds", "holdsAny", "inReceptacle", "objectAtLocation"],
        )

        self.assertEqual(len(per_step), 1)
        self.assert_no_errors(per_step[0])
        self.assertEqual(accumulation, [])

    def test_clean_object_using_sink_has_no_errors(self):
        """Cleaning a held cleanable object at a sink should add clean."""
        initial = (
            "Agent A is at Kitchen, "
            "SinkBasin SB1 is at Kitchen, "
            "Mug M1 is cleanable, "
            "Agent A is holding Mug M1"
        )

        _, _, per_step, accumulation = self.run_trace(
            initial,
            """
            ACTION: Clean object M1 using receptacle SB1 at location Kitchen
            STATE 1: Agent A is at Kitchen, Agent A is holding Mug M1, SinkBasin SB1 is at Kitchen, Mug M1 is cleanable, Mug M1 is clean
            """,
            dynamic_predicates=["isClean"],
        )

        self.assertEqual(len(per_step), 1)
        self.assert_no_errors(per_step[0])
        self.assertEqual(accumulation, [])

    def test_heat_object_using_microwave_has_no_errors(self):
        """Heating a held heatable object in a microwave should add hot and remove cool."""
        initial = (
            "Agent A is at Kitchen, "
            "Microwave MW1 is at Kitchen, "
            "Mug M1 is heatable, "
            "Mug M1 is cool, "
            "Agent A is holding Mug M1"
        )

        _, _, per_step, accumulation = self.run_trace(
            initial,
            """
            ACTION: Heat object M1 using receptacle MW1 at location Kitchen
            STATE 1: Agent A is at Kitchen, Agent A is holding Mug M1, Microwave MW1 is at Kitchen, Mug M1 is heatable, Mug M1 is hot
            """,
            dynamic_predicates=["isHot", "isCool"],
        )

        self.assertEqual(len(per_step), 1)
        self.assert_no_errors(per_step[0])
        self.assertEqual(accumulation, [])

    def test_cool_object_using_fridge_has_no_errors(self):
        """Cooling a held coolable object in a fridge should add cool and remove hot."""
        initial = (
            "Agent A is at Kitchen, "
            "Fridge F1 is at Kitchen, "
            "Mug M1 is coolable, "
            "Mug M1 is hot, "
            "Agent A is holding Mug M1"
        )

        _, _, per_step, accumulation = self.run_trace(
            initial,
            """
            ACTION: Cool object M1 using receptacle F1 at location Kitchen
            STATE 1: Agent A is at Kitchen, Agent A is holding Mug M1, Fridge F1 is at Kitchen, Mug M1 is coolable, Mug M1 is cool
            """,
            dynamic_predicates=["isHot", "isCool"],
        )

        self.assertEqual(len(per_step), 1)
        self.assert_no_errors(per_step[0])
        self.assertEqual(accumulation, [])

    def test_toggle_object_flips_on_and_marks_toggled(self):
        """Toggling a toggleable object should turn it on and mark it toggled."""
        initial = (
            "Agent A is at Kitchen, "
            "Table T1 is at Kitchen, "
            "Lamp L1 is in Table T1, "
            "Lamp L1 is toggleable, "
            "Agent A is holding nothing"
        )

        _, _, per_step, accumulation = self.run_trace(
            initial,
            """
            ACTION: Toggle object L1 in receptacle T1 at location Kitchen
            STATE 1: Agent A is at Kitchen, Agent A is holding nothing, Table T1 is at Kitchen, Lamp L1 is in Table T1, Lamp L1 is toggleable, Lamp L1 is on, Lamp L1 is toggled
            """,
            dynamic_predicates=["isOn", "isToggled"],
        )

        self.assertEqual(len(per_step), 1)
        self.assert_no_errors(per_step[0])
        self.assertEqual(accumulation, [])

    def test_slice_object_with_knife_has_no_errors(self):
        """Slicing a sliceable object while holding a knife should add sliced."""
        initial = (
            "Agent A is at Kitchen, "
            "Apple AP1 is at Kitchen, "
            "Apple AP1 is sliceable, "
            "Agent A is holding Knife K1"
        )

        _, _, per_step, accumulation = self.run_trace(
            initial,
            """
            ACTION: Slice object AP1 at location Kitchen using tool K1
            STATE 1: Agent A is at Kitchen, Agent A is holding Knife K1, Apple AP1 is at Kitchen, Apple AP1 is sliceable, Apple AP1 is sliced
            """,
            dynamic_predicates=["isSliced"],
        )

        self.assertEqual(len(per_step), 1)
        self.assert_no_errors(per_step[0])
        self.assertEqual(accumulation, [])

    def test_unknown_action_format_noops(self):
        """An action with the wrong wording should not change the state."""
        _, _, per_step, accumulation = self.run_trace(
            MOVE_INIT,
            """
            ACTION: Walk from LivingRoom into Kitchen
            STATE 1: Agent A is at LivingRoom, CounterTop CT1 is at Kitchen, CounterTop CT2 is at Bathroom, Agent A is holding nothing
            """,
            dynamic_predicates=["atLocation"],
        )

        self.assertEqual(len(per_step), 1)
        self.assert_no_errors(per_step[0])
        self.assertEqual(accumulation, [])

    # ------------------------------------------------------------------
    # Extra evaluator edge cases.
    # These check special behavior in episode_diagnostics.
    # ------------------------------------------------------------------

    def test_empty_dynamic_predicates_disables_delta_errors_but_keeps_divergence(self):
        """Empty dynamic_predicates should suppress omission/hallucination but keep divergence."""
        _, _, per_step, _ = self.run_trace(
            MOVE_INIT,
            """
            ACTION: Go from location LivingRoom to location Kitchen
            STATE 1: Agent A is at Bathroom, CounterTop CT1 is at Kitchen, CounterTop CT2 is at Bathroom, Agent A is holding nothing
            """,
            dynamic_predicates=[],
        )

        self.assertEqual(len(per_step), 1)
        self.assertFalse(per_step[0]["omission_error"])
        self.assertFalse(per_step[0]["hallucination_error"])
        self.assertTrue(per_step[0]["propagation_error"])
        self.assertTrue(per_step[0]["divergence_facts"])

    def test_episode_diagnostics_only_scores_aligned_action_state_pairs(self):
        """If actions and states have different lengths, only paired steps are scored."""
        actions = [
            "Go from location LivingRoom to location Kitchen",
            "Go from location Kitchen to location Bathroom",
        ]
        states = [
            "Agent A is at Kitchen, CounterTop CT1 is at Kitchen, CounterTop CT2 is at Bathroom, Agent A is holding nothing"
        ]

        per_step, accumulation = episode_diagnostics(
            actions,
            states,
            MOVE_INIT,
            dynamic_predicates=["atLocation"],
        )

        self.assertEqual(len(per_step), 1)
        self.assert_no_errors(per_step[0])
        self.assertEqual(accumulation, [])

    # ------------------------------------------------------------------
    # Basic parser tests.
    # These check that ACTION / STATE extraction works.
    # ------------------------------------------------------------------

    def test_parser_ignores_commentary_and_reads_action_state_pair(self):
        """The parser should ignore normal prose around ACTION and STATE lines."""
        actions, states, per_step, _ = self.run_trace(
            PICKUP_INIT,
            """
            This is commentary and should be ignored.
            ACTION: Pick up object AP1 from receptacle D1 at location Kitchen
            More commentary.
            STATE 1: Agent A is at Kitchen, Agent A is holding Apple AP1, Drawer D1 is at Kitchen, Drawer D1 is openable, Drawer D1 is opened
            """,
        )

        self.assertEqual(actions, ["Pick up object AP1 from receptacle D1 at location Kitchen"])
        self.assertEqual(len(states), 1)
        self.assertEqual(len(per_step), 1)
        self.assert_no_errors(per_step[0])

    def test_parser_uses_last_state_in_same_action_segment(self):
        """If one action has multiple STATE lines, use the last STATE.

        This supports traces where the model first writes a rough state,
        then writes a refined/corrected state for the same action.
        """
        actions, states = extract_actions_and_states(
            textwrap.dedent(
                """
                ACTION: Go from location LivingRoom to location Kitchen
                STATE 1: first memo
                STATE 1.1: refined memo
                """
            ).strip()
        )

        self.assertEqual(actions, ["Go from location LivingRoom to location Kitchen"])
        self.assertEqual(states, ["refined memo"])

    def test_parser_selects_last_dotted_leaf_branch(self):
        """The parser should choose the deepest final branch, not every branch.

        Trace shape:

        A1 -> STATE 1
        A2 -> STATE 1.1

        A3 -> STATE 2
        A4 -> STATE 2.1
        A5 -> STATE 2.1.1

        The last leaf is 2.1.1, so the committed path is:
        2 -> 2.1 -> 2.1.1.
        Therefore the parser should return A3, A4, A5.
        """
        trace = textwrap.dedent(
            """
            ACTION: Branch one first action
            STATE 1: b1
            ACTION: Branch one continuation
            STATE 1.1: b11

            ACTION: Branch two first action
            STATE 2: b2
            ACTION: Branch two continuation
            STATE 2.1: b21
            ACTION: Branch two leaf
            STATE 2.1.1: b211
            """
        ).strip()

        actions, states = extract_actions_and_states(trace)
        self.assertEqual(
            actions,
            [
                "Branch two first action",
                "Branch two continuation",
                "Branch two leaf",
            ],
        )
        self.assertEqual(states, ["b2", "b21", "b211"])

    # ------------------------------------------------------------------
    # Longer branch parser tests.
    # These test the exact kind of tree numbering you described.
    # ------------------------------------------------------------------

    def test_parser_complex_branch_keeps_final_committed_path(self):
        """Complex branch example using A1, A2, etc. as action names.

        The written trace is in generation order:

        state 1
        A2
        state 1.1
        A3
        state 1.1.1
        A4
        state 1.1.1.1

        A5
        state 1.1.2
        A6
        state 1.1.2.1
        A7
        state 1.1.2.2

        A8
        state 1.1.2.1.1

        The final leaf is 1.1.2.1.1.
        Its state path is:
        1 -> 1.1 -> 1.1.2 -> 1.1.2.1 -> 1.1.2.1.1

        The actions attached to those states are:
        A1 -> A2 -> A5 -> A6 -> A8
        """
        trace = textwrap.dedent(
            """
            ACTION: A1
            STATE 1: state 1

            ACTION: A2
            STATE 1.1: state 1.1

            ACTION: A3
            STATE 1.1.1: state 1.1.1

            ACTION: A4
            STATE 1.1.1.1: state 1.1.1.1

            ACTION: A5
            STATE 1.1.2: state 1.1.2

            ACTION: A6
            STATE 1.1.2.1: state 1.1.2.1

            ACTION: A7
            STATE 1.1.2.2: state 1.1.2.2

            ACTION: A8
            STATE 1.1.2.1.1: state 1.1.2.1.1
            """
        ).strip()

        actions, states = extract_actions_and_states(trace)

        self.assertEqual(actions, ["A1", "A2", "A5", "A6", "A8"])
        self.assertEqual(
            states,
            [
                "state 1",
                "state 1.1",
                "state 1.1.2",
                "state 1.1.2.1",
                "state 1.1.2.1.1",
            ],
        )

    def test_parser_complex_branch_ignores_dead_sibling_leaf(self):
        """The parser should not keep a sibling branch that is not on the final path.

        Here, 1.1.2.2 is a sibling of 1.1.2.1.
        The final leaf is 1.1.2.1.1.
        So A7 / state 1.1.2.2 should be ignored.
        """
        trace = textwrap.dedent(
            """
            ACTION: A1
            STATE 1: state 1

            ACTION: A2
            STATE 1.1: state 1.1

            ACTION: A5
            STATE 1.1.2: state 1.1.2

            ACTION: A6
            STATE 1.1.2.1: state 1.1.2.1

            ACTION: A7
            STATE 1.1.2.2: state 1.1.2.2

            ACTION: A8
            STATE 1.1.2.1.1: state 1.1.2.1.1
            """
        ).strip()

        actions, states = extract_actions_and_states(trace)

        self.assertNotIn("A7", actions)
        self.assertNotIn("state 1.1.2.2", states)
        self.assertEqual(actions, ["A1", "A2", "A5", "A6", "A8"])

    def test_parser_complex_branch_can_switch_to_later_top_level_branch(self):
        """A later top-level branch should replace an earlier top-level branch.

        First the trace explores branch 1:
        1 -> 1.1 -> 1.1.1

        Then it backtracks all the way and explores branch 2:
        2 -> 2.1

        The last leaf is 2.1.
        So the parser should return only A4 and A5.
        """
        trace = textwrap.dedent(
            """
            ACTION: A1
            STATE 1: state 1

            ACTION: A2
            STATE 1.1: state 1.1

            ACTION: A3
            STATE 1.1.1: state 1.1.1

            ACTION: A4
            STATE 2: state 2

            ACTION: A5
            STATE 2.1: state 2.1
            """
        ).strip()

        actions, states = extract_actions_and_states(trace)

        self.assertEqual(actions, ["A4", "A5"])
        self.assertEqual(states, ["state 2", "state 2.1"])

    def test_parser_complex_branch_uses_last_deepest_leaf_even_if_shorter_than_old_branch(self):
        """The latest committed leaf wins, even if an older branch was deeper.

        Older branch:
        1 -> 1.1 -> 1.1.1 -> 1.1.1.1

        Later branch:
        1 -> 1.2

        The last leaf is 1.2, so the committed path is:
        1 -> 1.2
        """
        trace = textwrap.dedent(
            """
            ACTION: A1
            STATE 1: state 1

            ACTION: A2
            STATE 1.1: state 1.1

            ACTION: A3
            STATE 1.1.1: state 1.1.1

            ACTION: A4
            STATE 1.1.1.1: state 1.1.1.1

            ACTION: A5
            STATE 1.2: state 1.2
            """
        ).strip()

        actions, states = extract_actions_and_states(trace)

        self.assertEqual(actions, ["A1", "A5"])
        self.assertEqual(states, ["state 1", "state 1.2"])

    # ------------------------------------------------------------------
    # Helper tests.
    # These test tiny helper functions directly.
    # ------------------------------------------------------------------

    def test_prefix_chain_helper(self):
        """prefix_chain returns every prefix of a state label.

        Example:
        3.1.1 becomes:
        3, 3.1, 3.1.1
        """
        self.assertEqual(prefix_chain((3, 1, 1)), [(3,), (3, 1), (3, 1, 1)])

    def test_parse_state_header_helper(self):
        """parse_state_header splits the label from the state text.

        Input:
        STATE 2.3.4: hello world

        Output:
        label = (2, 3, 4)
        state text = hello world
        """
        label, state_text = parse_state_header("STATE 2.3.4: hello world")
        self.assertEqual(label, (2, 3, 4))
        self.assertEqual(state_text, "hello world")


if __name__ == "__main__":
    unittest.main()
