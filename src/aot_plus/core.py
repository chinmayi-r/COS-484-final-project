"""Core implementation of the Algorithm of Thoughts Plus (AoT+) methodology."""
from typing import Dict, List, Optional, Any, Callable
import json
import re

from src.models.llm_client import get_llm_client


class AoTPlusPromptBuilder:
    """Builder for AoT+ prompts.

    Supports three modes:
        1. Natural language state memoization: use_json_state=False
        2. Structured JSON state memoization: use_json_state=True, use_goal_residual=False
        3. JSON + goal residual tracking: use_json_state=True, use_goal_residual=True
    """

    def __init__(
        self,
        domain_description: str,
        use_json_state: bool = False,
        use_goal_residual: bool = False,
        state_memoization_interval: int = 3,
        state_transition_fn: Optional[Callable[[str, str], str]] = None,
    ):
        self.domain_description = domain_description
        self.use_json_state = use_json_state
        self.use_goal_residual = use_goal_residual
        self.state_memoization_interval = state_memoization_interval
        self.state_transition_fn = state_transition_fn
        self.examples = []
        self.formatted_examples: List[str] = []

    def add_formatted_example(self, example_str: str) -> None:
        """Add a pre-formatted example string.

        This is most useful for JSON mode, where you may hand-write examples with
        STATE_JSON blocks.
        """
        self.formatted_examples.append(example_str)

    def add_example(
        self,
        initial_state: str,
        goal_state: str,
        solution_trajectory: List[str],
        random_trajectories: List[List[str]],
    ) -> None:
        """Add a few-shot example with both successful and random trajectories."""
        self.examples.append({
            "initial_state": initial_state,
            "goal_state": goal_state,
            "solution_trajectory": solution_trajectory,
            "random_trajectories": random_trajectories,
        })

    def _state_to_json_dict(self, state_description) -> Dict[str, Any]:
        """Convert a state object/string/dict into a JSON-serializable dict."""
        if hasattr(state_description, "to_json_dict"):
            return state_description.to_json_dict()

        if isinstance(state_description, dict):
            return dict(state_description)

        if isinstance(state_description, str):
            try:
                parsed = json.loads(state_description)
                if isinstance(parsed, dict):
                    return parsed
                return {"state": parsed}
            except json.JSONDecodeError:
                # Fallback: still valid JSON, but indicates this state was only text.
                return {"state_text": state_description}

        return {"state_text": str(state_description)}

    def _format_state_memoization(
        self,
        state_id: str,
        state_description,
        goal_state=None,
    ) -> str:
        """Format a state memoization entry.

        Natural-language mode:
            STATE 1: Agent A is at ...

        JSON mode:
            STATE_JSON:
            ```json
            {...}
            ```

        JSON + residual mode additionally includes:
            goal_satisfied
            goal_remaining
        """
        if self.use_json_state:
            state_dict = self._state_to_json_dict(state_description)

            if self.use_goal_residual:
                if goal_state is not None and hasattr(state_description, "compute_goal_diff"):
                    satisfied, remaining = state_description.compute_goal_diff(goal_state)
                else:
                    # For model-output prompting, the model will fill these fields.
                    # For generated examples, these remain empty unless the domain state
                    # object implements compute_goal_diff().
                    satisfied, remaining = [], []

                state_dict["goal_satisfied"] = satisfied
                state_dict["goal_remaining"] = remaining

            return (
                "STATE_JSON:\n"
                "```json\n"
                f"{json.dumps(state_dict, indent=2, sort_keys=True)}\n"
                "```\n"
            )

        return f"STATE {state_id}: {state_description}\n"

    def _make_state_id(self, parent_state_id: Optional[str], candidate_index: int) -> str:
        if parent_state_id is None:
            return str(candidate_index)
        return f"{parent_state_id}.{candidate_index}"

    def _format_example(self, example: Dict[str, Any]) -> str:
        """Format a complete example with failed branches first, then successful path."""
        initial_state = example["initial_state"]
        goal_state = example["goal_state"]
        solution = example["solution_trajectory"]
        random_paths = example["random_trajectories"]

        result = f"Initial state: {initial_state}\n"
        result += f"Goal state: {goal_state}\n\n"
        result += "I'll solve this step by step:\n\n"

        # 1. Show failed/random branches first.
        for branch_index, random_path in enumerate(random_paths, start=1):
            result += "Try a candidate path from the initial state.\n"

            branch_state = initial_state
            branch_state_id = None

            for i, action in enumerate(random_path):
                if i == 0:
                    state_id = str(branch_index)
                else:
                    state_id = self._make_state_id(branch_state_id, 1)

                result += f"ACTION: {action}\n"

                if self.state_transition_fn is not None:
                    branch_state = self.state_transition_fn(branch_state, action)

                result += self._format_state_memoization(state_id, branch_state, goal_state)
                branch_state_id = state_id

            result += f"STATE {branch_state_id} does not satisfy the goal. Backtrack to the initial state.\n\n"

        # 2. Then show the successful branch.
        success_branch_index = len(random_paths) + 1
        current_state = initial_state
        current_state_id = None

        result += "Try another candidate path from the initial state.\n"

        for i, action in enumerate(solution):
            if i == 0:
                state_id = str(success_branch_index)
            else:
                state_id = self._make_state_id(current_state_id, 1)

            result += f"ACTION: {action}\n"

            if self.state_transition_fn is not None:
                current_state = self.state_transition_fn(current_state, action)

            result += self._format_state_memoization(state_id, current_state, goal_state)
            current_state_id = state_id

        result += f"\nFinal check: {goal_state} has been achieved.\n"
        return result

    def build_prompt(self, initial_state: str, goal_state: str) -> str:
        """Build the complete AoT+ prompt.

        initial_state / goal_state are natural language in text mode and JSON strings
        in JSON modes.
        """
        if self.use_json_state:
            return self._build_json_prompt(initial_state, goal_state)
        return self._build_text_prompt(initial_state, goal_state)

    def _build_text_prompt(self, initial_state: str, goal_state: str) -> str:
        prompt = f"""# Algorithm of Thoughts Plus (AoT+) Planning System

## Domain Description:
{self.domain_description}
## Planning Task:
Given an initial state and a goal state, create a plan to transform the initial state into the goal state.
Use the Algorithm of Thoughts Plus approach to solve this planning problem.

### Key Instructions:
1. Think step by step to explore possible actions
2. Periodically track and restate the current state (using STATE notation)
3. Explore different paths when needed
4. Backtrack if a path doesn't seem promising
5. Build towards the goal state systematically
6. For each action you take, start the line with "ACTION:" followed by the action description
7. After each action, show the updated state using path-based STATE notation
8. STATE numbering is tree/path-based, not decimal counting.
9. A linear continuation always appends ".1" to the previous state label.
   Example: STATE 1 -> STATE 1.1 -> STATE 1.1.1 -> STATE 1.1.1.1.
10. A sibling branch increments only when trying a different action from the same parent state.
    Example: after STATE 1, a different second action would be STATE 1.2, not STATE 1.1.1.
11. STATE 2 means a completely different first action from the initial state.
12. Do not use STATE 1.2 to mean “the second time step.” Use STATE 1.1 for the next step after STATE 1, then STATE 1.1.1, and so on.
13. When backtracking, explicitly say which previous STATE or initial state you return to.

## Examples:
"""
        for example in self.examples:
            prompt += f"{self._format_example(example)}\n---\n\n"

        prompt += f"""## Problem to Solve:

Initial state: {initial_state}
Goal state: {goal_state}

Solve this step by step, tracking the state after each action. Remember to:
1. Prefix each action with "ACTION:"
2. Each ACTION should use object/receptacle IDs only, not type names.
For example, write “Pick up object AP1 from receptacle CT1,” not “Pick
up object Apple AP1 from receptacle CounterTop CT1.”
3. Update the state after each action using tree/path-based STATE notation:
   - next step in same path: append ".1"
   - alternative sibling action from same parent: increment the last number
   - different first action from initial state: use STATE 2, STATE 3, etc.
4. Do not number linear time steps as STATE 1.1, STATE 1.2, STATE 1.3.
5. After every ACTION, the STATE line must include the complete current dynamic world state, not only changed facts.
6. Include every dynamic fact that remains true:
   - agent location
   - what the agent is holding
   - every object currently in a receptacle
   - every object currently at a location
   - every opened receptacle
   - every goal-relevant object status, such as cool/hot/clean/sliced
7. Do not omit facts that are unchanged. If an Apple was in a DiningTable before and is still there, restate it.
8. Do not write "All else unchanged", "Agent at ...", "holding Apple ...", "Fridge is open", or "Apple removed from DiningTable".
9. Every included object/receptacle fact must include both the type word and the ID.
10. Use exact fact formats like:
   - Agent agent1 is at <location>
   - Agent agent1 is holding Apple <object_id>
   - Agent agent1 is holding nothing
   - Fridge <fridge_id> is opened
   - Apple <object_id> is cool
   - Apple <object_id> is in DiningTable <receptacle_id>
11. Each STATE must be written on one single line after the colon.
Do not use bullet points or multiline STATE blocks.
Correct:
STATE 1: Agent agent1 is at L, Agent agent1 is holding nothing
Incorrect:
STATE 1:
- Agent agent1 is at L
- Agent agent1 is holding nothing

Start your solution:
"""
        return prompt

    def _build_json_prompt(self, initial_state_json: str, goal_state_json: str) -> str:
        residual_rules = ""
        if self.use_goal_residual:
            residual_rules = """
7. Every STATE_JSON block must include these two extra fields:
   - "goal_satisfied": a short natural-language checklist of concrete goal conditions already true
   - "goal_remaining": a short natural-language checklist of concrete goal conditions not yet true
8. Do not copy symbolic existential variables like ?o or ?r into goal_satisfied/goal_remaining.
9. Instantiate the goal with concrete object/receptacle IDs when possible.
10. Example:
   "goal_satisfied": ["Apple_1 is cool"]
   "goal_remaining": ["Apple_1 must be placed in DiningTable_1"]
11. Use "goal_remaining" to decide what still needs to be done.
"""
        else:
            residual_rules = """
7. Do not include goal_satisfied or goal_remaining unless explicitly requested.
"""

        prompt = f"""# Algorithm of Thoughts Plus (AoT+) Planning System — JSON State Mode

## Domain Description:
{self.domain_description}

## Planning Task:
Given an initial state and a goal state (both in JSON), create a plan to transform the initial
state into the goal state.

### CRITICAL Output Format Rules:
1. For each action, output a line starting with "ACTION:" followed by the action description
2. Immediately after EVERY action (no exceptions), output a STATE_JSON block showing the complete updated world state.
3. Use this exact block format:
   STATE_JSON:
   ```json
   {{...}}
   ```
4. Every single action must be followed by exactly one STATE_JSON block.
5. Do not skip state updates for any action.
6. Every STATE_JSON block must be valid parseable JSON. Do not use comments or trailing commas.
{residual_rules}12. Explore different paths if needed; if you backtrack, still output ACTION + STATE_JSON for each move.
13. End with a "Plan summary:" section listing only the final successful actions.

## Examples:
"""
        for ex in self.formatted_examples:
            prompt += f"{ex}\n---\n\n"

        initial_state_block = self._ensure_initial_state_json_for_prompt(initial_state_json)

        prompt += f"""## Problem to Solve:

Initial state (JSON):
{initial_state_json}

Goal state (JSON):
{goal_state_json}

STATE_JSON:
```json
{initial_state_block}
```

Solve this step by step. Remember: after EVERY ACTION output exactly one STATE_JSON block.

Start your solution:
"""
        return prompt

    def _ensure_initial_state_json_for_prompt(self, initial_state_json: str) -> str:
        """Return a valid JSON string for the initial STATE_JSON block."""
        try:
            state_dict = json.loads(initial_state_json)

            if self.use_goal_residual:
                # Do NOT preserve stale/bad residual fields from the input JSON.
                # The model should recompute these after each action.
                state_dict["goal_satisfied"] = []
                state_dict["goal_remaining"] = [
                    "Goal not yet satisfied. Need to make some valid object/receptacle assignment satisfy the goal."
                ]

            return json.dumps(state_dict, indent=2, sort_keys=True)

        except json.JSONDecodeError:
            state_dict = {"state_text": initial_state_json}

            if self.use_goal_residual:
                state_dict["goal_satisfied"] = []
                state_dict["goal_remaining"] = [
                    "Goal not yet satisfied. Need to make some valid object/receptacle assignment satisfy the goal."
                ]

            return json.dumps(state_dict, indent=2, sort_keys=True)


class AoTPlus:
    """Main implementation of Algorithm of Thoughts Plus."""

    def __init__(
        self,
        domain_description: str,
        llm_provider: Optional[str] = None,
        use_json_state: bool = False,
        use_goal_residual: bool = False,
        state_memoization_interval: int = 3,
        state_transition_fn: Optional[Callable[[str, str], str]] = None,
    ):
        self.llm_client = get_llm_client(llm_provider)
        self.prompt_builder = AoTPlusPromptBuilder(
            domain_description,
            use_json_state=use_json_state,
            use_goal_residual=use_goal_residual,
            state_memoization_interval=state_memoization_interval,
            state_transition_fn=state_transition_fn,
        )

    def add_formatted_example(self, example_str: str) -> None:
        """Add a pre-formatted example."""
        self.prompt_builder.add_formatted_example(example_str)

    def add_example(
        self,
        initial_state: str,
        goal_state: str,
        solution_trajectory: List[str],
        random_trajectories: List[List[str]],
    ) -> None:
        """Add a few-shot example to the prompt builder."""
        self.prompt_builder.add_example(
            initial_state,
            goal_state,
            solution_trajectory,
            random_trajectories,
        )

    def solve(
        self,
        initial_state: str,
        goal_state: str,
        temperature: float = 0.0,
        max_tokens: int = 25000,
    ) -> str:
        """Solve a planning problem using AoT+."""
        prompt = self.prompt_builder.build_prompt(initial_state, goal_state)
        return self.llm_client.generate(prompt, temperature, max_tokens)

    def extract_plan(self, solution_text: str) -> List[str]:
        """Extract the final plan steps from the solution text."""
        steps = []

        # First try to extract from "Plan summary:" section.
        plan_summary_pattern = r"[Pp]lan [Ss]ummary:?(.*?)(?=$|\n\n)"
        plan_summary_match = re.search(plan_summary_pattern, solution_text, re.DOTALL)

        if plan_summary_match:
            plan_summary = plan_summary_match.group(1)
            numbered_step_pattern = r"(\d+)[\.\:\)]\s+(.*?)(?=\n\d+[\.\:\)]|\Z)"
            for match in re.finditer(numbered_step_pattern, plan_summary, re.DOTALL):
                step = match.group(2).strip()
                if step:
                    steps.append(step)

        # If no steps found from plan summary, look for ACTION: format.
        if not steps:
            action_pattern = r"ACTION:\s*(.*?)(?=STATE|ACTION|\n\n|$)"
            action_matches = re.finditer(action_pattern, solution_text)
            for match in action_matches:
                action = match.group(1).strip()
                if action and not action.lower().startswith("no action"):
                    steps.append(action)

        # If still no steps found, try Step format.
        if not steps:
            step_pattern = r"Step (\d+):\s*(.*?)(?=Step \d+:|$)"
            step_matches = re.finditer(step_pattern, solution_text, re.DOTALL)
            for match in step_matches:
                step = match.group(2).strip()
                if step and not any(x in step.lower() for x in ["let's try", "approach isn't", "backtrack"]):
                    steps.append(step)

        # If still no steps found, look for numbered actions outside of plan summary.
        if not steps:
            action_list_pattern = r"(?:^|\n)(\d+)[\.\:\)]\s+(.*?)(?=\n\d+[\.\:\)]|\Z)"
            for match in re.finditer(action_list_pattern, solution_text, re.DOTALL):
                step = match.group(2).strip()
                if step and not any(x in step.lower() for x in ["let's try", "approach isn't", "backtrack"]):
                    steps.append(step)

        return steps
