# alfworld.py
# Author(s): Evan Soper
# ALFWorld domain implementation for AoT+ and state memoization variants.

import re

class ALFWorldState:
    """Mutable ALFWorld state with PDDL-aligned predicates."""

    def __init__(self):
        # core placement/location
        self.at_location = {}
        self.receptacle_at_location = {}
        self.object_at_location = {}
        self.in_receptacle = {}
        self.in_receptacle_object = {}

        # receptacle/object property
        self.openable = set()
        self.opened = set()
        self.is_receptacle_object = set()
        self.is_receptacle_object_full = set()
        self.was_in_receptacle = set()
        self.checked = set()
        self.examined = set()
        self.receptacle_type = {}
        self.can_contain = set()
        self.object_type = {}
        self.full = set()

        # agent "holding"
        self.holds = {}
        self.holds_any = set()
        self.holds_any_receptacle_object = set()

        # object state
        self.is_clean = set()
        self.cleanable = set()
        self.is_hot = set()
        self.heatable = set()
        self.is_cool = set()
        self.coolable = set()
        self.pickupable = set()
        self.moveable = set()
        self.toggleable = set()
        self.is_on = set()
        self.is_toggled = set()
        self.sliceable = set()
        self.is_sliced = set()

    def _agent_holding(self, agent):
        return self.holds.get(agent)

    def _can_access_in_receptacle(self, obj, receptacle):
        if self.in_receptacle.get(obj) != receptacle:
            return False
        if receptacle not in self.openable:
            return True
        return receptacle in self.opened

    def look(self, agent, location):
        # Preconditions: (atLocation ?a ?l)
        if self.at_location.get(agent) != location:
            return False
        # Effects: (checked ?l)
        self.checked.add(location)
        return True

    def inventory(self, agent):
        # Preconditions: none
        # Effects: (checked ?a)
        self.checked.add(agent)
        return True

    def examine_receptacle(self, agent, receptacle):
        # Preconditions: agent and receptacle at same location
        agent_loc = self.at_location.get(agent)
        if not agent_loc:
            return False
        if self.receptacle_at_location.get(receptacle) != agent_loc:
            return False
        # Effects: (checked ?r)
        self.checked.add(receptacle)
        return True

    def examine_object(self, agent, obj):
        # Preconditions:
        # (or
        #   (exists (... ?r) (atLocation ?a ?l) (receptacleAtLocation ?r ?l)
        #                    (inReceptacle ?o ?r) and receptacle open if openable)
        #   (holds ?a ?o))
        if self._agent_holding(agent) == obj:
            self.checked.add(obj)
            return True

        agent_loc = self.at_location.get(agent)
        if not agent_loc:
            return False

        recep = self.in_receptacle.get(obj)
        if not recep:
            return False
        if self.receptacle_at_location.get(recep) != agent_loc:
            return False
        if not self._can_access_in_receptacle(obj, recep):
            return False

        # Effects: (checked ?o)
        self.checked.add(obj)
        return True

    def goto_location(self, agent, start_location, end_location, receptacle_at_destination):
        # Preconditions: at start and provided receptacle at destination
        if self.at_location.get(agent) != start_location:
            return False
        if self.receptacle_at_location.get(receptacle_at_destination) != end_location:
            return False

        # Effects: move agent to destination
        self.at_location[agent] = end_location
        return True

    def open_object(self, agent, location, receptacle):
        # Preconditions from OpenObject
        if receptacle not in self.openable:
            return False
        if self.at_location.get(agent) != location:
            return False
        if self.receptacle_at_location.get(receptacle) != location:
            return False
        if receptacle in self.opened:
            return False

        # Effects: opened + checked
        self.opened.add(receptacle)
        self.checked.add(receptacle)
        return True

    def close_object(self, agent, location, receptacle):
        # Preconditions from CloseObject
        if receptacle not in self.openable:
            return False
        if self.at_location.get(agent) != location:
            return False
        if self.receptacle_at_location.get(receptacle) != location:
            return False
        if receptacle not in self.opened:
            return False

        # Effects: remove opened
        self.opened.remove(receptacle)
        return True

    def pickup_object(self, agent, location, obj, receptacle):
        # Preconditions from PickupObject
        if obj not in self.pickupable:
            return False
        if self.at_location.get(agent) != location:
            return False
        if self.receptacle_at_location.get(receptacle) != location:
            return False
        if self.in_receptacle.get(obj) != receptacle:
            return False
        if agent in self.holds_any:
            return False
        if receptacle in self.openable and receptacle not in self.opened:
            return False

        # Effects
        del self.in_receptacle[obj]
        self.holds[agent] = obj
        self.holds_any.add(agent)
        if self.object_at_location.get(obj) == location:
            del self.object_at_location[obj]
        return True

    def put_object(self, agent, location, obj, receptacle, object_type, receptacle_type):
        # Preconditions from PutObject
        if self.holds.get(agent) != obj:
            return False
        if self.at_location.get(agent) != location:
            return False
        if self.receptacle_at_location.get(receptacle) != location:
            return False
        if receptacle in self.openable and receptacle not in self.opened:
            return False
        if self.object_type.get(obj) != object_type:
            return False
        if self.receptacle_type.get(receptacle) != receptacle_type:
            return False
        if (receptacle_type, object_type) not in self.can_contain:
            return False

        # Effects
        self.in_receptacle[obj] = receptacle
        self.object_at_location[obj] = location
        del self.holds[agent]
        self.holds_any.discard(agent)
        return True

    def clean_object(self, agent, location, receptacle, obj):
        # Preconditions from CleanObject
        if obj not in self.cleanable:
            return False
        if self.receptacle_type.get(receptacle) not in {"SinkType", "SinkBasinType"}:
            return False
        if self.at_location.get(agent) != location:
            return False
        if self.receptacle_at_location.get(receptacle) != location:
            return False
        if self.holds.get(agent) != obj:
            return False

        # Effects
        self.is_clean.add(obj)
        return True

    def heat_object(self, agent, location, receptacle, obj):
        # Preconditions from HeatObject
        if obj not in self.heatable:
            return False
        if self.receptacle_type.get(receptacle) != "MicrowaveType":
            return False
        if self.at_location.get(agent) != location:
            return False
        if self.receptacle_at_location.get(receptacle) != location:
            return False
        if self.holds.get(agent) != obj:
            return False

        # Effects
        self.is_hot.add(obj)
        self.is_cool.discard(obj)
        return True

    def cool_object(self, agent, location, receptacle, obj):
        # Preconditions from CoolObject
        if obj not in self.coolable:
            return False
        if self.receptacle_type.get(receptacle) != "FridgeType":
            return False
        if self.at_location.get(agent) != location:
            return False
        if self.receptacle_at_location.get(receptacle) != location:
            return False
        if self.holds.get(agent) != obj:
            return False

        # Effects
        self.is_cool.add(obj)
        self.is_hot.discard(obj)
        return True

    def toggle_object(self, agent, location, obj, receptacle):
        # Preconditions from ToggleObject
        if obj not in self.toggleable:
            return False
        if self.at_location.get(agent) != location:
            return False
        if self.receptacle_at_location.get(receptacle) != location:
            return False
        if self.in_receptacle.get(obj) != receptacle:
            return False

        # Effects: flip isOn, add isToggled
        if obj in self.is_on:
            self.is_on.remove(obj)
        else:
            self.is_on.add(obj)
        self.is_toggled.add(obj)
        return True

    def slice_object(self, agent, location, cuttable_object, knife_object):
        # Preconditions from SliceObject
        if cuttable_object not in self.sliceable:
            return False
        knife_type = self.object_type.get(knife_object)
        if knife_type not in {"KnifeType", "ButterKnifeType"}:
            return False
        if self.at_location.get(agent) != location:
            return False
        if self.object_at_location.get(cuttable_object) != location:
            return False
        if self.holds.get(agent) != knife_object:
            return False

        # Effects
        self.is_sliced.add(cuttable_object)
        return True

    def __str__(self):
        facts = []

        for agent, location in sorted(self.at_location.items()):
            facts.append("atLocation({0}, {1})".format(agent, location))
        for obj, recep in sorted(self.in_receptacle.items()):
            facts.append("inReceptacle({0}, {1})".format(obj, recep))
        for agent, obj in sorted(self.holds.items()):
            facts.append("holds({0}, {1})".format(agent, obj))

        return ", ".join(facts)

    def is_goal_state(self, goal_state):
        # Goal check uses subset semantics: every fact in goal_state must hold here.
        dict_fields = [
            "at_location",
            "receptacle_at_location",
            "object_at_location",
            "in_receptacle",
            "in_receptacle_object",
            "receptacle_type",
            "object_type",
            "holds",
        ]
        set_fields = [
            "openable",
            "opened",
            "is_receptacle_object",
            "is_receptacle_object_full",
            "was_in_receptacle",
            "checked",
            "examined",
            "can_contain",
            "holds_any",
            "holds_any_receptacle_object",
            "full",
            "is_clean",
            "cleanable",
            "is_hot",
            "heatable",
            "is_cool",
            "coolable",
            "pickupable",
            "moveable",
            "toggleable",
            "is_on",
            "is_toggled",
            "sliceable",
            "is_sliced",
        ]

        for field in dict_fields:
            mine = getattr(self, field)
            target = getattr(goal_state, field)
            for key, value in target.items():
                if mine.get(key) != value:
                    return False

        for field in set_fields:
            mine = getattr(self, field)
            target = getattr(goal_state, field)
            if not target.issubset(mine):
                return False

        return True

class ALFWorldDomain:
    """Implementation of the ALFWorld planning domain."""

    @staticmethod
    def get_domain_description():
        """Get the description of the ALFWorld domain for the prompt."""
        return """
# ALFWorld Planning Domain

## Description:
The ALFWorld domain involves household task planning in indoor environments.
An agent navigates between locations, interacts with receptacles, and manipulates objects to satisfy task goals.
Objects may need to be moved, placed, cleaned, heated, cooled, toggled, or sliced depending on the goal.

## Receptacle Types:
- ArmChairType
- BathtubBasinType
- BedType
- CabinetType
- CartType
- CoffeeMachineType
- CoffeeTableType
- CounterTopType
- DeskType
- DiningTableType
- DrawerType
- DresserType
- FridgeType
- GarbageCanType
- HandTowelHolderType
- LaundryHamperType
- MicrowaveType
- OttomanType
- SafeType
- ShelfType
- SideTableType
- SinkBasinType
- SofaType
- StoveBurnerType
- TVStandType
- ToasterType
- ToiletPaperHangerType
- ToiletType
- TowelHolderType

## Rules:
1. The agent can only act at its current location.
2. A receptacle can be opened only if it is openable and currently closed.
3. A receptacle can be closed only if it is openable and currently opened.
4. To pick up an object from a receptacle, the agent must be at the receptacle's location and have empty hands.
5. If a receptacle is openable, it must be opened before accessing objects inside it.
6. To place an object in a receptacle, the agent must be holding that object and the receptacle must be compatible with the object type.
7. Cleaning is valid only for cleanable objects and sink/sink-basin receptacles.
8. Heating is valid only for heatable objects and microwave receptacles.
9. Cooling is valid only for coolable objects and fridge receptacles.
10. Toggling is valid only for toggleable objects located in a receptacle at the agent's location.
11. Slicing is valid only for sliceable objects when the agent holds a knife or butter knife.

## Actions:
- Look around the current location
- Check inventory
- Examine a receptacle
- Examine an object
- Go from one location to another
- Open a receptacle
- Close a receptacle
- Pick up an object from a receptacle
- Put an object into a receptacle
- Clean an object
- Heat an object
- Cool an object
- Toggle an object on/off
- Slice an object with a knife-like tool

## Action Format:
- Look at location L
- Check inventory for agent A
- Examine receptacle R
- Examine object O
- Go from location L1 to location L2
- Open receptacle R at location L
- Close receptacle R at location L
- Pick up object O from receptacle R at location L
- Put object O into receptacle R at location L
- Clean object O using receptacle R at location L
- Heat object O using receptacle R at location L
- Cool object O using receptacle R at location L
- Toggle object O in receptacle R at location L
- Slice object O at location L using tool K

## State Representation:
The state is represented by facts about:
- where the agent is,
- where receptacles are,
- where objects are (including in which receptacle),
- whether receptacles are open/closed,
- what the agent is holding,
- object properties and statuses (clean/hot/cool/on/toggled/sliced),
- object/receptacle type compatibility constraints.
"""

    @staticmethod
    def _apply_fact(state, predicate, args):
        if predicate == "atLocation" and len(args) == 2:
            state.at_location[args[0]] = args[1]
        elif predicate == "receptacleAtLocation" and len(args) == 2:
            state.receptacle_at_location[args[0]] = args[1]
        elif predicate == "objectAtLocation" and len(args) == 2:
            state.object_at_location[args[0]] = args[1]
        elif predicate == "openable" and len(args) == 1:
            state.openable.add(args[0])
        elif predicate == "opened" and len(args) == 1:
            state.opened.add(args[0])
        elif predicate == "inReceptacle" and len(args) == 2:
            state.in_receptacle[args[0]] = args[1]
        elif predicate == "isReceptacleObject" and len(args) == 1:
            state.is_receptacle_object.add(args[0])
        elif predicate == "inReceptacleObject" and len(args) == 2:
            state.in_receptacle_object[args[0]] = args[1]
        elif predicate == "isReceptacleObjectFull" and len(args) == 1:
            state.is_receptacle_object_full.add(args[0])
        elif predicate == "wasInReceptacle" and len(args) == 2:
            state.was_in_receptacle.add((args[0], args[1]))
        elif predicate == "checked" and len(args) == 1:
            state.checked.add(args[0])
        elif predicate == "examined" and len(args) == 1:
            state.examined.add(args[0])
        elif predicate == "receptacleType" and len(args) == 2:
            state.receptacle_type[args[0]] = args[1]
        elif predicate == "canContain" and len(args) == 2:
            state.can_contain.add((args[0], args[1]))
        elif predicate == "objectType" and len(args) == 2:
            state.object_type[args[0]] = args[1]
        elif predicate == "holds" and len(args) == 2:
            state.holds[args[0]] = args[1]
        elif predicate == "holdsAny" and len(args) == 1:
            state.holds_any.add(args[0])
        elif predicate == "holdsAnyReceptacleObject" and len(args) == 1:
            state.holds_any_receptacle_object.add(args[0])
        elif predicate == "full" and len(args) == 1:
            state.full.add(args[0])
        elif predicate == "isClean" and len(args) == 1:
            state.is_clean.add(args[0])
        elif predicate == "cleanable" and len(args) == 1:
            state.cleanable.add(args[0])
        elif predicate == "isHot" and len(args) == 1:
            state.is_hot.add(args[0])
        elif predicate == "heatable" and len(args) == 1:
            state.heatable.add(args[0])
        elif predicate == "isCool" and len(args) == 1:
            state.is_cool.add(args[0])
        elif predicate == "coolable" and len(args) == 1:
            state.coolable.add(args[0])
        elif predicate == "pickupable" and len(args) == 1:
            state.pickupable.add(args[0])
        elif predicate == "moveable" and len(args) == 1:
            state.moveable.add(args[0])
        elif predicate == "toggleable" and len(args) == 1:
            state.toggleable.add(args[0])
        elif predicate == "isOn" and len(args) == 1:
            state.is_on.add(args[0])
        elif predicate == "isToggled" and len(args) == 1:
            state.is_toggled.add(args[0])
        elif predicate == "sliceable" and len(args) == 1:
            state.sliceable.add(args[0])
        elif predicate == "isSliced" and len(args) == 1:
            state.is_sliced.add(args[0])

    @staticmethod
    def parse_state_description(description):
        """Parse ALFWorld/PDDL-style fact descriptions into an ALFWorldState."""
        state = ALFWorldState()

        known_predicates = {
            "atLocation", "receptacleAtLocation", "objectAtLocation", "openable", "opened",
            "inReceptacle", "isReceptacleObject", "inReceptacleObject", "isReceptacleObjectFull",
            "wasInReceptacle", "checked", "examined", "receptacleType", "canContain", "objectType",
            "holds", "holdsAny", "holdsAnyReceptacleObject", "full", "isClean", "cleanable",
            "isHot", "heatable", "isCool", "coolable", "pickupable", "moveable", "toggleable",
            "isOn", "isToggled", "sliceable", "isSliced",
        }

        for atom in re.findall(r"\(([^()]+)\)", description):
            pieces = atom.split()
            if not pieces:
                continue

            predicate = pieces[0]
            args = pieces[1:]

            if predicate not in known_predicates:
                continue

            # Skip quantified-variable atoms from generic logical templates.
            if any(token.startswith("?") for token in args):
                continue

            ALFWorldDomain._apply_fact(state, predicate, args)

        return state

    @staticmethod
    def generate_example_problem():
        """Generate an example ALFWorld problem with solution and random trajectories."""
        initial_state = (
            "Agent A is at Kitchen, "
            "CounterTop T is at Kitchen, "
            "SinkBasin S is at Kitchen, "
            "Microwave M is at Kitchen, "
            "Fridge F is at Kitchen, "
            "Mug U is in CounterTop T, "
            "Mug U is cleanable, heatable, and coolable, "
            "Agent A is holding nothing"
        )

        goal_state = "Mug U is clean and Mug U is cool"

        solution = [
            "Pick up object Mug U from receptacle CounterTop T at location Kitchen",
            "Clean object Mug U using receptacle SinkBasin S at location Kitchen",
            "Heat object Mug U using receptacle Microwave M at location Kitchen",
            "Cool object Mug U using receptacle Fridge F at location Kitchen"
        ]

        random_trajectory1 = [
            "Pick up object Mug U from receptacle CounterTop T at location Kitchen",
            "Cool object Mug U using receptacle Microwave M at location Kitchen",
            "Put object Mug U into receptacle CounterTop T at location Kitchen"
        ]

        random_trajectory2 = [
            "Look at location Kitchen",
            "Check inventory for agent A",
            "Pick up object Mug U from receptacle CounterTop T at location Kitchen",
            "Clean object Mug U using receptacle Fridge F at location Kitchen"
        ]

        return initial_state, goal_state, solution, [random_trajectory1, random_trajectory2]
