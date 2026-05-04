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

    def goto_location(self, agent, start_location, end_location):
        if self.at_location.get(agent) != start_location:
            return False

        known_locations = set(self.receptacle_at_location.values())
        if end_location not in known_locations:
            return False

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

        # Effects: opened
        self.opened.add(receptacle)
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

    def put_object(self, agent, location, obj, receptacle):
        if self.holds.get(agent) != obj:
            return False
        if self.at_location.get(agent) != location:
            return False
        if self.receptacle_at_location.get(receptacle) != location:
            return False
        if receptacle in self.openable and receptacle not in self.opened:
            return False

        object_type = self.object_type.get(obj)
        receptacle_type = self.receptacle_type.get(receptacle)

        if (receptacle_type, object_type) not in self.can_contain:
            return False

        self.in_receptacle[obj] = receptacle
        self.object_at_location[obj] = location
        del self.holds[agent]
        self.holds_any.discard(agent)
        return True

    def clean_object(self, agent, location, receptacle, obj):
        # Preconditions from CleanObject
        if obj not in self.cleanable:
            return False
        if self.receptacle_type.get(receptacle) not in {"Sink", "SinkBasin"}:
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
        if self.receptacle_type.get(receptacle) != "Microwave":
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
        if self.receptacle_type.get(receptacle) != "Fridge":
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
        if knife_type not in {"Knife", "ButterKnife"}:
            return False
        if self.at_location.get(agent) != location:
            return False
        # Cuttable object may be tracked as on the floor (`objectAtLocation`) or only
        # as inside a receptacle at this location (`inReceptacle` + receptacle placement).
        on_floor = self.object_at_location.get(cuttable_object) == location
        in_rec_at_loc = False
        rec = self.in_receptacle.get(cuttable_object)
        if rec is not None and self.receptacle_at_location.get(rec) == location:
            in_rec_at_loc = True
        if not (on_floor or in_rec_at_loc):
            return False
        if self.holds.get(agent) != knife_object:
            return False

        # Effects
        self.is_sliced.add(cuttable_object)
        return True

    def __str__(self):
        facts = []

        for agent, location in sorted(self.at_location.items()):
            facts.append(f"Agent {agent} is at {location}")

            if agent in self.holds:
                obj = self.holds[agent]
                obj_type = self.object_type.get(obj, "Object")
                facts.append(f"Agent {agent} is holding {obj_type} {obj}")
            else:
                facts.append(f"Agent {agent} is holding nothing")

        for receptacle, location in sorted(self.receptacle_at_location.items()):
            receptacle_type = self.receptacle_type.get(receptacle, "Receptacle")
            facts.append(f"{receptacle_type} {receptacle} is at {location}")

            if receptacle in self.openable:
                facts.append(f"{receptacle_type} {receptacle} is openable")

            if receptacle in self.opened:
                facts.append(f"{receptacle_type} {receptacle} is opened")

        for obj, receptacle in sorted(self.in_receptacle.items()):
            obj_type = self.object_type.get(obj, "Object")
            receptacle_type = self.receptacle_type.get(receptacle, "Receptacle")
            facts.append(f"{obj_type} {obj} is in {receptacle_type} {receptacle}")

        held_objects = set(self.holds.values())
        contained_objects = set(self.in_receptacle.keys())

        for obj, location in sorted(self.object_at_location.items()):
            if obj in held_objects or obj in contained_objects:
                continue

            obj_type = self.object_type.get(obj, "Object")
            facts.append(f"{obj_type} {obj} is at {location}")

        property_sets = [
            ("pickupable", self.pickupable),
            ("moveable", self.moveable),
            ("cleanable", self.cleanable),
            ("heatable", self.heatable),
            ("coolable", self.coolable),
            ("toggleable", self.toggleable),
            ("sliceable", self.sliceable),
            ("clean", self.is_clean),
            ("hot", self.is_hot),
            ("cool", self.is_cool),
            ("on", self.is_on),
            ("toggled", self.is_toggled),
            ("sliced", self.is_sliced),
        ]

        for property_name, objects in property_sets:
            for obj in sorted(objects):
                obj_type = self.object_type.get(obj, "Object")
                facts.append(f"{obj_type} {obj} is {property_name}")

        return ", ".join(facts)
    
    @staticmethod
    def apply_action(current_state_description: str, action_description: str) -> str:
        state = ALFWorldDomain.parse_state_description(current_state_description)
        action = action_description.strip()
        agent = next(iter(state.at_location))

        match = re.fullmatch(
            r"Go from location ([A-Za-z0-9_]+) to location ([A-Za-z0-9_]+)",
            action,
        )
        if match:
            start_location, end_location = match.groups()
            state.goto_location(agent, start_location, end_location)
            return str(state)

        match = re.fullmatch(
            r"Open receptacle ([A-Za-z0-9_]+) at location ([A-Za-z0-9_]+)",
            action,
        )
        if match:
            receptacle, location = match.groups()
            state.open_object(agent, location, receptacle)
            return str(state)

        match = re.fullmatch(
            r"Close receptacle ([A-Za-z0-9_]+) at location ([A-Za-z0-9_]+)",
            action,
        )
        if match:
            receptacle, location = match.groups()
            state.close_object(agent, location, receptacle)
            return str(state)

        match = re.fullmatch(
            r"Pick up object ([A-Za-z0-9_]+) from receptacle ([A-Za-z0-9_]+) at location ([A-Za-z0-9_]+)",
            action,
        )
        if match:
            obj, receptacle, location = match.groups()
            state.pickup_object(agent, location, obj, receptacle)
            return str(state)

        match = re.fullmatch(
            r"Put object ([A-Za-z0-9_]+) into receptacle ([A-Za-z0-9_]+) at location ([A-Za-z0-9_]+)",
            action,
        )
        if match:
            obj, receptacle, location = match.groups()
            state.put_object(agent, location, obj, receptacle)
            return str(state)

        match = re.fullmatch(
            r"Clean object ([A-Za-z0-9_]+) using receptacle ([A-Za-z0-9_]+) at location ([A-Za-z0-9_]+)",
            action,
        )
        if match:
            obj, receptacle, location = match.groups()
            state.clean_object(agent, location, receptacle, obj)
            return str(state)

        match = re.fullmatch(
            r"Heat object ([A-Za-z0-9_]+) using receptacle ([A-Za-z0-9_]+) at location ([A-Za-z0-9_]+)",
            action,
        )
        if match:
            obj, receptacle, location = match.groups()
            state.heat_object(agent, location, receptacle, obj)
            return str(state)

        match = re.fullmatch(
            r"Cool object ([A-Za-z0-9_]+) using receptacle ([A-Za-z0-9_]+) at location ([A-Za-z0-9_]+)",
            action,
        )
        if match:
            obj, receptacle, location = match.groups()
            state.cool_object(agent, location, receptacle, obj)
            return str(state)

        match = re.fullmatch(
            r"Toggle object ([A-Za-z0-9_]+) in receptacle ([A-Za-z0-9_]+) at location ([A-Za-z0-9_]+)",
            action,
        )
        if match:
            obj, receptacle, location = match.groups()
            state.toggle_object(agent, location, obj, receptacle)
            return str(state)

        match = re.fullmatch(
            r"Slice object ([A-Za-z0-9_]+) at location ([A-Za-z0-9_]+) using tool ([A-Za-z0-9_]+)",
            action,
        )
        if match:
            obj, location, tool = match.groups()
            state.slice_object(agent, location, obj, tool)
            return str(state)

        return str(state)

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
            receptacle_type = match.group(1)
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
            object_type = match.group(1)
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

            state.object_type[obj] = obj_type_name
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
            # Use a trailing word boundary so tokens like `coolable` do not match `cool`.
            pattern = r"[A-Za-z]+ ([A-Za-z0-9_]+) is " + re.escape(word) + r"\b"
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
        # "Agent A is holding Mug U" / "Agent A is holding Knife K1"
        for match in re.finditer(
            r"Agent ([A-Za-z0-9_]+) is holding ([A-Za-z]+) ([A-Za-z0-9_]+)", description
        ):
            agent = match.group(1)
            obj_type_name = match.group(2)
            obj = match.group(3)
            state.holds[agent] = obj
            state.holds_any.add(agent)
            state.object_type[obj] = obj_type_name

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
        """Generate a natural-language ALFWorld example problem."""

        initial_state = (
            "Agent A is at LivingRoom, "

            # Locations / receptacles
            "CounterTop CT1 is at Kitchen, "
            "Drawer D1 is at Kitchen, "
            "DiningTable DT1 is at Kitchen, "
            "CoffeeTable COF1 is at LivingRoom, "

            # Openable receptacle
            "Drawer D1 is openable, "

            # Relevant objects
            "Apple AP1 is in CounterTop CT1, "
            "Knife K1 is in Drawer D1, "

            # Distractor object
            "Mug M1 is in CoffeeTable COF1, "

            # Object capabilities
            "Apple AP1 is pickupable, "
            "Apple AP1 is sliceable, "
            "Knife K1 is pickupable, "
            "Mug M1 is pickupable, "

            # Agent starts empty-handed
            "Agent A is holding nothing"
        )

        goal_state = (
            "Apple AP1 is sliced, "
            "Apple AP1 is in DiningTable DT1"
        )

        solution = [
            "Go from location LivingRoom to location Kitchen",
            "Open receptacle D1 at location Kitchen",
            "Pick up object K1 from receptacle D1 at location Kitchen",
            "Slice object AP1 at location Kitchen using tool K1",
            "Put object K1 into receptacle CT1 at location Kitchen",
            "Pick up object AP1 from receptacle CT1 at location Kitchen",
            "Put object AP1 into receptacle DT1 at location Kitchen"
        ]

        random_trajectory1 = [
            "Pick up object M1 from receptacle COF1 at location LivingRoom",
            "Go from location LivingRoom to location Kitchen",
            "Put object M1 into receptacle DT1 at location Kitchen"
        ]

        random_trajectory2 = [
            "Go from location LivingRoom to location Kitchen",
            "Pick up object AP1 from receptacle CT1 at location Kitchen",
            "Put object AP1 into receptacle DT1 at location Kitchen"
        ]

        return initial_state, goal_state, solution, [
            random_trajectory1,
            random_trajectory2,
        ]
