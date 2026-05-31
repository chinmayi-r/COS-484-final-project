"""Core implementation of the Algorithm of Thoughts Plus (AoT+) methodology."""
from typing import Dict, List, Optional, Any, Tuple
import re

from src.models.llm_client import LLMClient, get_llm_client

class AoTPlusPromptBuilder:
    """Builder for AoT+ style prompts with state memoization and random trajectory augmentation."""

    def __init__(self, domain_description: str, use_json_state: bool = False,
                 state_memoization_interval: int = 3, goal_tracking: bool = False):
        self.domain_description = domain_description
        self.use_json_state = use_json_state
        self.state_memoization_interval = state_memoization_interval
        self.goal_tracking = goal_tracking
        self.examples = []
        self.formatted_examples: List[str] = []
        self._formatted_examples_block: str = ""  # pre-built for JSON prompt reuse

    def add_formatted_example(self, example_str: str) -> None:
        self.formatted_examples.append(example_str)
        self._formatted_examples_block += f"{example_str}\n---\n\n"

    def _split_example_turns(self, example_str: str) -> tuple[str, str]:
        """Split a formatted example into (user_part, assistant_part).

        The user part is everything up to (but not including) the first STATE_JSON block.
        The assistant part is everything from STATE_JSON onward.
        """
        marker = "STATE_JSON:"
        idx = example_str.find(marker)
        if idx == -1:
            return example_str, ""
        return example_str[:idx].rstrip(), example_str[idx:]

    def build_messages(self, initial_state_json: str, goal_state_json: str,
                       extra_context: str = "") -> List[Dict[str, str]]:
        """Build a proper multi-turn messages list for the JSON extension.

        Structure mirrors the paper replication:
            system → [user/assistant per example] → user (new problem)

        This is structurally equivalent to build_bw_messages / build_log_messages
        used by the paper replication runner.
        """
        k = self.state_memoization_interval
        goal_instr = (
            " Each STATE_JSON must include 'goal_satisfied' (list of goal predicates "
            "already achieved) and 'goal_remaining' (list not yet achieved)."
            if self.goal_tracking else ""
        )
        ckpt_instr = (
            "After every action output one STATE_JSON block showing the complete current world state"
            if k == 1 else
            f"Every {k} actions output one STATE_JSON block showing the complete current world state"
        )
        system_content = (
            "You are a planning assistant. Solve planning problems step by step using "
            "the Algorithm of Thoughts approach demonstrated in the examples. "
            f"{ckpt_instr}; always output a STATE_JSON block after your final action.{goal_instr} "
            "End with a 'Plan summary:' numbered list of the committed actions only."
        )
        if self.domain_description:
            system_content += f"\n\nDomain: {self.domain_description}"

        messages: List[Dict[str, str]] = [{"role": "system", "content": system_content}]

        for ex in self.formatted_examples:
            user_part, asst_part = self._split_example_turns(ex)
            if user_part and asst_part:
                messages.append({"role": "user",      "content": user_part})
                messages.append({"role": "assistant", "content": asst_part})

        extra_section = f"\n{extra_context}\n" if extra_context else ""
        user_content = (
            f"{extra_section}Initial state (JSON):\n{initial_state_json}\n\n"
            f"Goal state (JSON):\n{goal_state_json}"
        )
        messages.append({"role": "user", "content": user_content})
        return messages

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
    
    def _interleave_trajectories(self, solution: List[str], random_paths: List[List[str]]) -> List[str]:
        """Interleave the solution and random trajectories for example generation.
        
        Args:
            solution: List of steps in the successful solution path
            random_paths: List of lists, where each inner list is a sequence of steps in a random path
            
        Returns:
            List of interleaved steps, showing exploration and backtracking
        """
        # Simple interleaving strategy - can be made more sophisticated
        interleaved = []
        solution_steps_used = 0
        
        # Add initial solution steps (up to half)
        mid_point = len(solution) // 2
        interleaved.extend(solution[:mid_point])
        solution_steps_used = mid_point
        
        # Add random exploration paths with backtracking
        for random_path in random_paths:
            interleaved.append("Let's try a different approach.")
            interleaved.extend(random_path)
            interleaved.append("This approach isn't working well. Let's backtrack.")
        
        # Complete with the rest of the solution
        interleaved.extend(solution[solution_steps_used:])
        
        return interleaved
    
    def _format_example(self, example: Dict[str, Any]) -> str:
        """Format a complete example with state memoization and interleaved trajectories.
        
        Args:
            example: Dictionary containing the example data
            
        Returns:
            Formatted example as a string
        """
        initial_state = example["initial_state"]
        goal_state = example["goal_state"]
        
        # Interleave solution and random trajectories
        steps = self._interleave_trajectories(
            example["solution_trajectory"], 
            example["random_trajectories"]
        )
        
        # Build the example with state memoization
        result = f"Initial state: {initial_state}\n"
        result += f"Goal state: {goal_state}\n\n"
        result += self._format_state_memoization("1.0", initial_state)
        result += "I'll solve this step by step:\n\n"
        
        current_state = initial_state
        plan_steps = []
        
        for i, step in enumerate(steps):
            if "backtrack" not in step.lower() and "approach" not in step.lower():
                # This is an actual action step
                result += f"ACTION: {step}\n"
                plan_steps.append(step)
            else:
                # This is a commentary or backtracking step
                result += f"{step}\n"
            
            # Add state memoization periodically
            if (i+1) % self.state_memoization_interval == 0:
                # If backtracking, use a different state id format
                if "backtrack" in step.lower():
                    state_id = f"{(i+1)//self.state_memoization_interval}.0"
                else:
                    state_id = f"{(i+1)//self.state_memoization_interval+1}.0"
                
                # This would be replaced with actual state computation
                current_state = f"Updated state after step {i+1}"
                result += self._format_state_memoization(state_id, current_state)
        
        result += f"\nFinal check: {goal_state} has been achieved.\n\n"
        
        # Add a plan summary section
        result += "Plan summary:\n"
        for i, step in enumerate(plan_steps):
            result += f"{i+1}. {step}\n"
            
        return result
    
    def build_prompt(self, initial_state: str, goal_state: str,
                    extra_context: str = "") -> str:
        """Build the complete AoT+ prompt for the given problem.

        initial_state / goal_state are JSON strings in JSON mode, natural language otherwise.
        extra_context is injected into the problem section (e.g. city maps for logistics).
        """
        if self.use_json_state:
            return self._build_json_prompt(initial_state, goal_state, extra_context)
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
7. After each action, show the new state using "STATE x.y:" notation
8. Conclude with a "Plan summary:" section that lists all the actions in numbered format

## Examples:
"""
        for example in self.examples:
            prompt += f"{self._format_example(example)}\n---\n\n"

        prompt += f"""## Problem to Solve:

Initial state: {initial_state}
Goal state: {goal_state}

STATE 0: {initial_state}

Solve this step by step, tracking the state periodically. Remember to:
1. Prefix each action with "ACTION:"
2. Update the state after significant changes using "STATE x.y:" notation
3. End with a "Plan summary:" section

Start your solution:
"""
        return prompt

    def _build_json_prompt(self, initial_state_json: str, goal_state_json: str,
                           extra_context: str = "") -> str:
        k = self.state_memoization_interval
        goal_rule = (
            f"\n7. Each STATE_JSON must include 'goal_satisfied' and 'goal_remaining' "
            "fields tracking which goal predicates are already achieved."
            if self.goal_tracking else ""
        )
        prompt = f"""# Algorithm of Thoughts Plus (AoT+) Planning System — JSON State Mode

## Domain Description:
{self.domain_description}

## Planning Task:
Given an initial state and a goal state (both in JSON), create a plan to transform the initial
state into the goal state.

### CRITICAL Output Format Rules:
1. For each action, output a line starting with "ACTION:" followed by the action description.
2. After every {k} actions, output one STATE_JSON block showing the complete updated world
   state in this exact format:
   STATE_JSON: ```json
   {{...}}
   ```
3. Always output a STATE_JSON block after your final action.
4. If you backtrack, output ACTION lines for the undo moves and include them in the checkpoint count.
5. End with a "Plan summary:" section listing only the final committed actions.
6. Before moving a block, verify it appears in the "clear" array of the current STATE_JSON.{goal_rule}

## Examples:
"""
        prompt += self._formatted_examples_block
        extra_section = f"\n{extra_context}\n" if extra_context else ""
        prompt += f"""## Problem to Solve:
{extra_section}
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

    def __init__(self, domain_description: str, llm_provider: Optional[str] = None,
                 use_json_state: bool = False, state_memoization_interval: int = 3,
                 goal_tracking: bool = False, model: Optional[str] = None):
        self.llm_client = get_llm_client(llm_provider, model=model)
        self.prompt_builder = AoTPlusPromptBuilder(
            domain_description,
            use_json_state=use_json_state,
            state_memoization_interval=state_memoization_interval,
            goal_tracking=goal_tracking,
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
              temperature: float = 0.0, max_tokens: int = 3000,
              extra_context: str = "") -> str:
        """Solve a planning problem using AoT+.

        Args:
            initial_state: Description of the initial state
            goal_state: Description of the goal state
            temperature: Sampling temperature for the LLM
            max_tokens: Maximum number of tokens to generate
            extra_context: Optional extra text injected into the problem section

        Returns:
            The generated solution plan
        """
        if self.prompt_builder.use_json_state:
            messages = self.prompt_builder.build_messages(initial_state, goal_state, extra_context)
            return self.llm_client.generate_messages(messages, temperature, max_tokens)
        prompt = self.prompt_builder.build_prompt(initial_state, goal_state, extra_context)
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