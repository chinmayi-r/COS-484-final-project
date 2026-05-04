"""Core implementation of the Algorithm of Thoughts Plus (AoT+) methodology."""
from typing import Dict, List, Optional, Any, Tuple
import re

from src.models.llm_client import LLMClient, get_llm_client

class AoTPlusPromptBuilder:
    """Builder for AoT+ style prompts with state memoization and random trajectory augmentation."""

    def __init__(self, domain_description: str, use_json_state: bool = False,
                 state_memoization_interval: int = 4):
        self.domain_description = domain_description
        self.use_json_state = use_json_state
        self.state_memoization_interval = state_memoization_interval
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
        n = self.state_memoization_interval
        system_content = (
            "You are a planning assistant. Solve planning problems step by step using "
            "the Algorithm of Thoughts approach demonstrated in the examples. "
            f"Output a STATE_JSON block every {n} actions (or at a natural phase boundary) "
            "and always after the final action. "
            "Each STATE_JSON must include 'goal_satisfied' (list of met goal predicates as strings) "
            "and 'goal_remaining' (list of unmet goal predicates as strings) fields. "
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
        """Interleave the solution and random trajectories for example generation."""
        interleaved = []
        solution_steps_used = 0

        mid_point = len(solution) // 2
        interleaved.extend(solution[:mid_point])
        solution_steps_used = mid_point

        for random_path in random_paths:
            interleaved.append("Let's try a different approach.")
            interleaved.extend(random_path)
            interleaved.append("This approach isn't working well. Let's backtrack.")

        interleaved.extend(solution[solution_steps_used:])

        return interleaved

    def _format_example(self, example: Dict[str, Any]) -> str:
        """Format a complete example with state memoization and interleaved trajectories."""
        initial_state = example["initial_state"]
        goal_state = example["goal_state"]

        steps = self._interleave_trajectories(
            example["solution_trajectory"],
            example["random_trajectories"]
        )

        result = f"Initial state: {initial_state}\n"
        result += f"Goal state: {goal_state}\n\n"
        result += self._format_state_memoization("1.0", initial_state)
        result += "I'll solve this step by step:\n\n"

        current_state = initial_state
        plan_steps = []

        for i, step in enumerate(steps):
            if "backtrack" not in step.lower() and "approach" not in step.lower():
                result += f"ACTION: {step}\n"
                plan_steps.append(step)
            else:
                result += f"{step}\n"

            if (i+1) % self.state_memoization_interval == 0:
                if "backtrack" in step.lower():
                    state_id = f"{(i+1)//self.state_memoization_interval}.0"
                else:
                    state_id = f"{(i+1)//self.state_memoization_interval+1}.0"

                current_state = f"Updated state after step {i+1}"
                result += self._format_state_memoization(state_id, current_state)

        result += f"\nFinal check: {goal_state} has been achieved.\n\n"

        result += "Plan summary:\n"
        for i, step in enumerate(plan_steps):
            result += f"{i+1}. {step}\n"

        return result

    def build_prompt(self, initial_state: str, goal_state: str,
                    extra_context: str = "") -> str:
        """Build the complete AoT+ prompt for the given problem."""
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
        n = self.state_memoization_interval
        prompt = f"""# Algorithm of Thoughts Plus (AoT+) Planning System — JSON State Mode

## Domain Description:
{self.domain_description}

## Planning Task:
Given an initial state and a goal state (both in JSON), create a plan to transform the initial
state into the goal state.

### Output Format Rules:
1. For each action, output a line starting with "ACTION:" followed by the action description
2. Every {n} actions (or at a natural phase boundary), output a STATE_JSON block in this format:
   STATE_JSON: ```json
   {{...}}
   ```
3. Always output a STATE_JSON block after the final action
4. Each STATE_JSON must include:
   - "goal_satisfied": list of goal predicates already met (e.g. ["P0 at L2-0", "B on A"])
   - "goal_remaining": list of unmet goal predicates
5. Explore different paths if needed; if you backtrack, record the undo actions too
6. End with a "Plan summary:" section listing only the final successful actions
7. Start with a goal analysis: for each entity, note current location vs goal and whether
   it needs to move (e.g. "P0: L2 -> L2 (no)" or "B: table -> A (yes)")
8. Before acting, verify preconditions from the most recent STATE_JSON

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

Solve this step by step. Start with a goal analysis, then execute actions. Output STATE_JSON
with goal_satisfied/goal_remaining every {n} actions or at phase boundaries.

Start your solution:
"""
        return prompt


class AoTPlus:
    """Main implementation of Algorithm of Thoughts Plus."""

    def __init__(self, domain_description: str, llm_provider: Optional[str] = None,
                 use_json_state: bool = False, state_memoization_interval: int = 4):
        self.llm_client = get_llm_client(llm_provider)
        self.prompt_builder = AoTPlusPromptBuilder(
            domain_description,
            use_json_state=use_json_state,
            state_memoization_interval=state_memoization_interval,
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
        """Solve a planning problem using AoT+."""
        if self.prompt_builder.use_json_state:
            messages = self.prompt_builder.build_messages(initial_state, goal_state, extra_context)
            return self.llm_client.generate_messages(messages, temperature, max_tokens)
        prompt = self.prompt_builder.build_prompt(initial_state, goal_state, extra_context)
        return self.llm_client.generate(prompt, temperature, max_tokens)

    def extract_plan(self, solution_text: str) -> List[str]:
        """Extract the final plan steps from the solution text."""
        steps = []

        plan_summary_pattern = r"[Pp]lan [Ss]ummary:?(.*?)(?=$|\n\n)"
        plan_summary_match = re.search(plan_summary_pattern, solution_text, re.DOTALL)

        if plan_summary_match:
            plan_summary = plan_summary_match.group(1)
            numbered_step_pattern = r"(\d+)[\.:\)]\s+(.*?)(?=\n\d+[\.:\)]|\Z)"
            for match in re.finditer(numbered_step_pattern, plan_summary, re.DOTALL):
                step = match.group(2).strip()
                if step:
                    steps.append(step)

        if not steps:
            action_pattern = r"ACTION:\s*(.*?)(?=STATE|ACTION|\n\n|$)"
            action_matches = re.finditer(action_pattern, solution_text)
            for match in action_matches:
                action = match.group(1).strip()
                if action and not action.lower().startswith("no action"):
                    steps.append(action)

        if not steps:
            step_pattern = r"Step (\d+):\s*(.*?)(?=Step \d+:|$)"
            step_matches = re.finditer(step_pattern, solution_text, re.DOTALL)
            for match in step_matches:
                step = match.group(2).strip()
                if step and not any(x in step.lower() for x in ["let's try", "approach isn't", "backtrack"]):
                    steps.append(step)

        if not steps:
            action_list_pattern = r"(?:^|\n)(\d+)[\.:\)]\s+(.*?)(?=\n\d+[\.:\)]|\Z)"
            for match in re.finditer(action_list_pattern, solution_text, re.DOTALL):
                step = match.group(2).strip()
                if step and not any(x in step.lower() for x in ["let's try", "approach isn't", "backtrack"]):
                    steps.append(step)

        return steps
