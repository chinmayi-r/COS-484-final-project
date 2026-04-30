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
    def parse_state_description(description):
        """Parse a natural-language ALFWorld state description."""
        state = ALFWorldState()

        # Agent location:
        # "Agent A is at Kitchen"
        for match in re.finditer(r"Agent ([A-Za-z0-9_]+) is at ([A-Za-z0-9_]+)", description):
            agent = match.group(1)
            location = match.group(2)
            state.at_location[agent] = location

        # Receptacle location:
        # "CounterTop T is at Kitchen"
        receptacle_pattern = (
            r"(CounterTop|SinkBasin|Microwave|Fridge|Cabinet|Drawer|Shelf|Table|Desk|Dresser|"
            r"CoffeeTable|DiningTable|SideTable|Sofa|Bed|ArmChair|GarbageCan|Toaster|"
            r"StoveBurner|CoffeeMachine|BathtubBasin|Toilet|TVStand|Safe) "
            r"([A-Za-z0-9_]+) is at ([A-Za-z0-9_]+)"
        )
        for match in re.finditer(receptacle_pattern, description):
            receptacle_type = match.group(1) + "Type"
            receptacle = match.group(2)
            location = match.group(3)
            state.receptacle_at_location[receptacle] = location
            state.receptacle_type[receptacle] = receptacle_type

        # Object inside receptacle:
        # "Mug U is in CounterTop T"
        object_in_receptacle_pattern = (
            r"([A-Za-z]+) ([A-Za-z0-9_]+) is in "
            r"(CounterTop|SinkBasin|Microwave|Fridge|Cabinet|Drawer|Shelf|Table|Desk|Dresser|"
            r"CoffeeTable|DiningTable|SideTable|Sofa|Bed|ArmChair|GarbageCan|Toaster|"
            r"StoveBurner|CoffeeMachine|BathtubBasin|Toilet|TVStand|Safe) "
            r"([A-Za-z0-9_]+)"
        )
        for match in re.finditer(object_in_receptacle_pattern, description):
            object_type = match.group(1) + "Type"
            obj = match.group(2)
            receptacle = match.group(4)

            state.object_type[obj] = object_type
            state.in_receptacle[obj] = receptacle

            if receptacle in state.receptacle_at_location:
                state.object_at_location[obj] = state.receptacle_at_location[receptacle]

        # Object directly at location:
        # "Knife K is at Kitchen"
        object_at_location_pattern = r"([A-Za-z]+) ([A-Za-z0-9_]+) is at ([A-Za-z0-9_]+)"
        for match in re.finditer(object_at_location_pattern, description):
            obj_type_name = match.group(1)
            obj = match.group(2)
            location = match.group(3)

            # Avoid reparsing agent/receptacle facts as objects.
            if obj_type_name in {
                "Agent", "CounterTop", "SinkBasin", "Microwave", "Fridge", "Cabinet",
                "Drawer", "Shelf", "Table", "Desk", "Dresser", "CoffeeTable",
                "DiningTable", "SideTable", "Sofa", "Bed", "ArmChair", "GarbageCan",
                "Toaster", "StoveBurner", "CoffeeMachine", "BathtubBasin", "Toilet",
                "TVStand", "Safe"
            }:
                continue

            state.object_type[obj] = obj_type_name + "Type"
            state.object_at_location[obj] = location

        # Openable / opened:
        # "Fridge F is openable"
        # "Fridge F is opened"
        for match in re.finditer(r"[A-Za-z]+ ([A-Za-z0-9_]+) is openable", description):
            state.openable.add(match.group(1))

        for match in re.finditer(r"[A-Za-z]+ ([A-Za-z0-9_]+) is opened", description):
            state.opened.add(match.group(1))

        # Object capabilities/statuses:
        capability_map = {
            "cleanable": state.cleanable,
            "heatable": state.heatable,
            "coolable": state.coolable,
            "pickupable": state.pickupable,
            "moveable": state.moveable,
            "toggleable": state.toggleable,
            "sliceable": state.sliceable,
            "clean": state.is_clean,
            "hot": state.is_hot,
            "cool": state.is_cool,
            "on": state.is_on,
            "toggled": state.is_toggled,
            "sliced": state.is_sliced,
        }

        for word, target_set in capability_map.items():
            pattern = r"[A-Za-z]+ ([A-Za-z0-9_]+) is " + word
            for match in re.finditer(pattern, description):
                target_set.add(match.group(1))

        # Handles compact form:
        # "Mug U is cleanable, heatable, and coolable"
        compact_property_pattern = r"[A-Za-z]+ ([A-Za-z0-9_]+) is ([^.]+?)(?:,|$)"
        for match in re.finditer(compact_property_pattern, description):
            obj = match.group(1)
            props = match.group(2).replace("and", ",")
            for prop in props.split(","):
                prop = prop.strip()
                if prop in capability_map:
                    capability_map[prop].add(obj)

        # Holding:
        # "Agent A is holding Mug U"
        for match in re.finditer(r"Agent ([A-Za-z0-9_]+) is holding [A-Za-z]+ ([A-Za-z0-9_]+)", description):
            agent = match.group(1)
            obj = match.group(2)
            state.holds[agent] = obj
            state.holds_any.add(agent)

        # Default compatibility constraints needed for example planning.
        # These mirror the canContain predicate role in the PDDL.
        common_receptacle_types = set(state.receptacle_type.values())
        common_object_types = set(state.object_type.values())
        for rt in common_receptacle_types:
            for ot in common_object_types:
                state.can_contain.add((rt, ot))

        return state

    @staticmethod
    def generate_example_problem():
        """Generate a natural-language ALFWorld example problem.

        This example is intentionally more ALFWorld-like than a minimal toy task:
        the agent must navigate, open a receptacle, retrieve a tool, manipulate an
        object through multiple state changes, and place it in a goal receptacle.
        """

        initial_state = (
            "Agent A is at LivingRoom, "

            # Kitchen receptacles
            "CounterTop CT1 is at Kitchen, "
            "SinkBasin SB1 is at Kitchen, "
            "Fridge F1 is at Kitchen, "
            "Cabinet C1 is at Kitchen, "
            "Drawer D1 is at Kitchen, "
            "DiningTable DT1 is at Kitchen, "

            # Living room distractor receptacles
            "CoffeeTable COF1 is at LivingRoom, "
            "Sofa S1 is at LivingRoom, "

            # Bathroom distractor receptacles
            "SinkBasin SB2 is at Bathroom, "
            "Cabinet C2 is at Bathroom, "

            # Openable/closed/open facts
            "Fridge F1 is openable, "
            "Fridge F1 is opened, "
            "Cabinet C1 is openable, "
            "Drawer D1 is openable, "
            "Cabinet C2 is openable, "

            # Relevant objects
            "Apple AP1 is in CounterTop CT1, "
            "Knife K1 is in Drawer D1, "

            # Distractor objects
            "Mug M1 is in CoffeeTable COF1, "
            "SoapBottle SOAP1 is in Cabinet C2, "
            "Book B1 is in Sofa S1, "

            # Object capabilities
            "Apple AP1 is pickupable, "
            "Apple AP1 is cleanable, "
            "Apple AP1 is coolable, "
            "Apple AP1 is sliceable, "
            "Knife K1 is pickupable, "
            "Mug M1 is pickupable, "
            "Mug M1 is cleanable, "
            "SoapBottle SOAP1 is pickupable, "
            "Book B1 is pickupable, "

            # Agent starts empty-handed
            "Agent A is holding nothing"
        )

        goal_state = (
            "Apple AP1 is sliced, "
            "Apple AP1 is clean, "
            "Apple AP1 is cool, "
            "Apple AP1 is in DiningTable DT1"
        )

        solution = [
            "Go from location LivingRoom to location Kitchen",
            "Open receptacle Drawer D1 at location Kitchen",
            "Pick up object Knife K1 from receptacle Drawer D1 at location Kitchen",
            "Slice object Apple AP1 at location Kitchen using tool K1",
            "Put object Knife K1 into receptacle CounterTop CT1 at location Kitchen",
            "Pick up object Apple AP1 from receptacle CounterTop CT1 at location Kitchen",
            "Clean object Apple AP1 using receptacle SinkBasin SB1 at location Kitchen",
            "Cool object Apple AP1 using receptacle Fridge F1 at location Kitchen",
            "Put object Apple AP1 into receptacle DiningTable DT1 at location Kitchen"
        ]

        random_trajectory1 = [
            "Look at location LivingRoom",
            "Examine receptacle CoffeeTable COF1",
            "Pick up object Mug M1 from receptacle CoffeeTable COF1 at location LivingRoom",
            "Go from location LivingRoom to location Kitchen",
            "Clean object Mug M1 using receptacle SinkBasin SB1 at location Kitchen"
        ]

        random_trajectory2 = [
            "Go from location LivingRoom to location Kitchen",
            "Pick up object Knife K1 from receptacle Drawer D1 at location Kitchen",
            "Slice object Apple AP1 at location Kitchen using tool K1"
        ]

        random_trajectory3 = [
            "Go from location LivingRoom to location Bathroom",
            "Open receptacle Cabinet C2 at location Bathroom",
            "Pick up object SoapBottle SOAP1 from receptacle Cabinet C2 at location Bathroom",
            "Go from location Bathroom to location Kitchen",
            "Cool object SoapBottle SOAP1 using receptacle Fridge F1 at location Kitchen"
        ]

        return initial_state, goal_state, solution, [
            random_trajectory1,
            random_trajectory2,
            random_trajectory3,
        ]
