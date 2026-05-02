"""Core implementation of the Algorithm of Thoughts Plus (AoT+) methodology."""
from typing import Dict, List, Optional, Any, Callable
import re

# from src.models.llm_client import LLMClient, get_llm_client

class AoTPlusPromptBuilder:
    """Builder for AoT+ style prompts with state memoization and random trajectory augmentation."""

    def __init__(
        self,
        domain_description: str,
        use_json_state: bool = False,
        state_memoization_interval: int = 3,
        state_transition_fn: Optional[Callable[[str, str], str]] = None,
    ):
        self.domain_description = domain_description
        self.use_json_state = use_json_state
        self.state_memoization_interval = state_memoization_interval
        self.state_transition_fn = state_transition_fn
        self.examples = []
        self.formatted_examples: List[str] = []
    
    def add_formatted_example(self, example_str: str) -> None:
        """Add a pre-formatted example string (used for JSON mode where states are hardcoded)."""
        self.formatted_examples.append(example_str)

    def add_example(self, initial_state: str, goal_state: str, solution_trajectory: List[str],
                   random_trajectories: List[List[str]]) -> None:
        """Add a few-shot example with both successful and random trajectories."""
        self.examples.append({
            "initial_state": initial_state,
            "goal_state": goal_state,
            "solution_trajectory": solution_trajectory,
            "random_trajectories": random_trajectories
        })
    
    def _format_state_memoization(self, state_id: str, state_description: str) -> str:
        """Format a state memoization entry.

        In JSON mode: state_description must be a JSON string; wraps in STATE_JSON fenced block.
        In text mode: outputs the classic STATE N: <text> line.
        """
        if self.use_json_state:
            return f"STATE_JSON: ```json\n{state_description}\n```\n"
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

                result += self._format_state_memoization(state_id, branch_state)
                branch_state_id = state_id

            result += f"STATE {branch_state_id} does not satisfy the goal. Backtrack to the initial state.\n\n"

        # 2. Then show the successful branch.
        success_branch_index = len(random_paths) + 1
        current_state = initial_state
        current_state_id = None
        plan_steps = []

        result += "Try another candidate path from the initial state.\n"

        for i, action in enumerate(solution):
            if i == 0:
                state_id = str(success_branch_index)
            else:
                state_id = self._make_state_id(current_state_id, 1)

            result += f"ACTION: {action}\n"

            if self.state_transition_fn is not None:
                current_state = self.state_transition_fn(current_state, action)

            result += self._format_state_memoization(state_id, current_state)

            plan_steps.append(action)
            current_state_id = state_id

        result += f"\nFinal check: {goal_state} has been achieved.\n\n"

        result += "Plan summary:\n"
        for i, step in enumerate(plan_steps):
            result += f"{i + 1}. {step}\n"

        return result
    
    def build_prompt(self, initial_state: str, goal_state: str) -> str:
        """Build the complete AoT+ prompt for the given problem.

        initial_state / goal_state are JSON strings in JSON mode, natural language otherwise.
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
8. STATE 1 means the first candidate first action from the initial state
9. STATE 2 means the second candidate first action from the initial state
10. STATE 1.1 means the first continuation after STATE 1
11. STATE 1.2 means the second continuation after STATE 1
12. When backtracking, explicitly return to a previous STATE or to the initial state
14. Conclude with a "Plan summary:" section that lists only the final successful actions

## Examples:
"""
        for example in self.examples:
            prompt += f"{self._format_example(example)}\n---\n\n"

        prompt += f"""## Problem to Solve:

Initial state: {initial_state}
Goal state: {goal_state}

Solve this step by step, tracking the state after each action. Remember to:
1. Prefix each action with "ACTION:"
2. Update the state after each action using path-based STATE notation
3. Backtracking should return to a previous STATE or to the initial state
** 4. YOU MUST END with a "Plan summary:" section containing only the final successful actions

Start your solution:
"""
        return prompt

    def _build_json_prompt(self, initial_state_json: str, goal_state_json: str) -> str:
        prompt = f"""# Algorithm of Thoughts Plus (AoT+) Planning System — JSON State Mode

## Domain Description:
{self.domain_description}

## Planning Task:
Given an initial state and a goal state (both in JSON), create a plan to transform the initial
state into the goal state.

### CRITICAL Output Format Rules:
1. For each action, output a line starting with "ACTION:" followed by the action description
2. Immediately after EVERY action (no exceptions), output a STATE_JSON block showing the
   complete updated world state in this exact format:
   STATE_JSON: ```json
   {{...}}
   ```
3. Every single action must be followed by exactly one STATE_JSON block
4. Do not skip state updates for any action
5. Explore different paths if needed; if you backtrack, still output ACTION + STATE_JSON
   for each move (including the undo moves)
6. End with a "Plan summary:" section listing only the final successful actions

## Examples:
"""
        for ex in self.formatted_examples:
            prompt += f"{ex}\n---\n\n"

        prompt += f"""## Problem to Solve:

Initial state (JSON):
{initial_state_json}

Goal state (JSON):
{goal_state_json}

STATE_JSON: ```json
{initial_state_json}
```

Solve this step by step. Remember: after EVERY ACTION output exactly one STATE_JSON block.

Start your solution:
"""
        return prompt


class AoTPlus:
    """Main implementation of Algorithm of Thoughts Plus."""

    def __init__(
        self,
        domain_description: str,
        llm_provider: Optional[str] = None,
        use_json_state: bool = False,
        state_memoization_interval: int = 3,
        state_transition_fn: Optional[Callable[[str, str], str]] = None,
    ):
        # self.llm_client = get_llm_client(llm_provider)
        self.prompt_builder = AoTPlusPromptBuilder(
            domain_description,
            use_json_state=use_json_state,
            state_memoization_interval=state_memoization_interval,
            state_transition_fn=state_transition_fn,
        )

    def add_formatted_example(self, example_str: str) -> None:
        """Add a pre-formatted example (JSON mode)."""
        self.prompt_builder.add_formatted_example(example_str)

    def add_example(self, initial_state: str, goal_state: str, solution_trajectory: List[str],
                   random_trajectories: List[List[str]]) -> None:
        """Add a few-shot example to the prompt builder."""
        self.prompt_builder.add_example(
            initial_state, goal_state, solution_trajectory, random_trajectories
        )
        
    def solve(self, initial_state: str, goal_state: str, 
              temperature: float = 0.0, max_tokens: int = 3000) -> str:
        """Solve a planning problem using AoT+.
        
        Args:
            initial_state: Description of the initial state
            goal_state: Description of the goal state
            temperature: Sampling temperature for the LLM
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            The generated solution plan
        """
        prompt = self.prompt_builder.build_prompt(initial_state, goal_state)
        return self.llm_client.generate(prompt, temperature, max_tokens)
    
    def extract_plan(self, solution_text: str) -> List[str]:
        """Extract the final plan steps from the solution text.
        
        Args:
            solution_text: The raw output from the LLM
            
        Returns:
            List of plan steps
        """
        steps = []
        
        # First try to extract from "Plan summary:" section
        plan_summary_pattern = r"[Pp]lan [Ss]ummary:?(.*?)(?=$|\n\n)"
        plan_summary_match = re.search(plan_summary_pattern, solution_text, re.DOTALL)
        
        if plan_summary_match:
            # Extract steps from numbered list in plan summary
            plan_summary = plan_summary_match.group(1)
            numbered_step_pattern = r"(\d+)[\.:\)]\s+(.*?)(?=\n\d+[\.:\)]|\Z)"
            for match in re.finditer(numbered_step_pattern, plan_summary, re.DOTALL):
                step = match.group(2).strip()
                if step:
                    steps.append(step)
        
        # If no steps found from plan summary, look for ACTION: format
        if not steps:
            action_pattern = r"ACTION:\s*(.*?)(?=STATE|ACTION|\n\n|$)"
            action_matches = re.finditer(action_pattern, solution_text)
            for match in action_matches:
                action = match.group(1).strip()
                if action and not action.lower().startswith("no action"):
                    steps.append(action)
        
        # If still no steps found, try Step format
        if not steps:
            step_pattern = r"Step (\d+):\s*(.*?)(?=Step \d+:|$)"
            step_matches = re.finditer(step_pattern, solution_text, re.DOTALL)
            for match in step_matches:
                step = match.group(2).strip()
                if step and not any(x in step.lower() for x in ["let's try", "approach isn't", "backtrack"]):
                    steps.append(step)
        
        # If still no steps found, look for numbered actions outside of plan summary
        if not steps:
            # First check for a list of actions with ordinal numbers
            action_list_pattern = r"(?:^|\n)(\d+)[\.:\)]\s+(.*?)(?=\n\d+[\.:\)]|\Z)"
            for match in re.finditer(action_list_pattern, solution_text, re.DOTALL):
                step = match.group(2).strip()
                if step and not any(x in step.lower() for x in ["let's try", "approach isn't", "backtrack"]):
                    steps.append(step)
        
        return steps
