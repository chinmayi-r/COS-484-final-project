"""JSON-formatted prompt builders for AoT+ variants."""
from typing import Dict, List, Optional, Any
import json
import re

from src.models.llm_client import LLMClient, get_llm_client


class AoTPlusJSONPromptBuilder:
    """AoT+ prompt builder that represents state as JSON objects."""

    def __init__(self, domain_description: str):
        self.domain_description = domain_description
        self.examples = []

    def add_example(self, initial_state_json: Dict, goal_state_json: Dict,
                    solution_trajectory: List[Dict], random_trajectories: List[List[Dict]]) -> None:
        self.examples.append({
            "initial_state": initial_state_json,
            "goal_state": goal_state_json,
            "solution_trajectory": solution_trajectory,
            "random_trajectories": random_trajectories,
        })

    def _format_state_block(self, state_id: str, state_json: Dict) -> str:
        return f"STATE {state_id}:\n```json\n{json.dumps(state_json, indent=2)}\n```\n"

    def _format_example(self, example: Dict[str, Any]) -> str:
        initial = example["initial_state"]
        goal = example["goal_state"]
        solution = example["solution_trajectory"]
        randoms = example["random_trajectories"]

        result = "Starting State:\n"
        result += f"```json\n{json.dumps(initial, indent=2)}\n```\n"
        result += "Goal State:\n"
        result += f"```json\n{json.dumps(goal, indent=2)}\n```\n\n"
        result += self._format_state_block("0", initial)

        mid = len(solution) // 2
        for i, step in enumerate(solution[:mid]):
            result += f"ACTION: {step['action']}\n"
            result += self._format_state_block(str(i + 1), step["state"])

        for rand_path in randoms:
            result += "Backtracking to try a different path:\n"
            for j, step in enumerate(rand_path):
                result += f"ACTION: {step['action']}\n"
                result += self._format_state_block(f"b{j}", step["state"])
            result += "This path does not reach the goal. Backtracking to initial state.\n"
            result += self._format_state_block("0", initial)

        for i, step in enumerate(solution[mid:], start=mid):
            result += f"ACTION: {step['action']}\n"
            result += self._format_state_block(str(i + 1), step["state"])

        result += "\nExact solution:\n"
        for i, step in enumerate(solution, 1):
            result += f"{i}. {step['action']}\n"
        return result

    def build_prompt(self, initial_state_json: Dict, goal_state_json: Dict) -> str:
        prompt = f"""# AoT+ Planning with JSON State Representation

## Domain Description:
{self.domain_description}

### Instructions:
1. States are represented as JSON objects.
2. After each action prefix with "ACTION:", output the updated state as a JSON block labeled "STATE <id>:".
3. Explore alternatives and backtrack when a path fails.
4. End with a numbered "Exact solution:" list.

## Examples:
"""
        for ex in self.examples:
            prompt += self._format_example(ex) + "\n---\n\n"

        prompt += f"""## Problem to Solve:

Starting State:
```json
{json.dumps(initial_state_json, indent=2)}
```
Goal State:
```json
{json.dumps(goal_state_json, indent=2)}
```

STATE 0:
```json
{json.dumps(initial_state_json, indent=2)}
```

Solve step by step. Prefix each action with "ACTION:", output updated JSON state after each action, backtrack if needed, end with "Exact solution:".
"""
        return prompt


class AoTPlusJSONGoalTrackingPromptBuilder:
    """AoT+ prompt builder with JSON state AND per-step goal predicate tracking."""

    def __init__(self, domain_description: str):
        self.domain_description = domain_description
        self.examples = []

    def add_example(self, initial_state_json: Dict, goal_state_json: Dict,
                    solution_trajectory: List[Dict], random_trajectories: List[List[Dict]]) -> None:
        self.examples.append({
            "initial_state": initial_state_json,
            "goal_state": goal_state_json,
            "solution_trajectory": solution_trajectory,
            "random_trajectories": random_trajectories,
        })

    def _check_goal_progress(self, state: Dict, goal: Dict) -> Dict:
        """Compute which goal predicates are satisfied in state."""
        satisfied = {}
        unsatisfied = {}
        for key, val in goal.items():
            if isinstance(val, dict):
                sat = {k: state.get(key, {}).get(k) == v for k, v in val.items()}
                satisfied[key] = {k: v for k, v in sat.items() if v}
                unsatisfied[key] = {k: goal[key][k] for k, v in sat.items() if not v}
            elif isinstance(val, list):
                sat = [item for item in val if item in state.get(key, [])]
                unsat = [item for item in val if item not in state.get(key, [])]
                satisfied[key] = sat
                unsatisfied[key] = unsat
            else:
                if state.get(key) == val:
                    satisfied[key] = val
                else:
                    unsatisfied[key] = val
        return {"satisfied": satisfied, "unsatisfied": unsatisfied}

    def _format_step_block(self, step_id: str, state: Dict, goal: Dict) -> str:
        progress = self._check_goal_progress(state, goal)
        block = {
            "step": step_id,
            "state": state,
            "goal_progress": progress,
        }
        return f"```json\n{json.dumps(block, indent=2)}\n```\n"

    def _format_example(self, example: Dict[str, Any]) -> str:
        initial = example["initial_state"]
        goal = example["goal_state"]
        solution = example["solution_trajectory"]
        randoms = example["random_trajectories"]

        result = "Starting State:\n"
        result += f"```json\n{json.dumps(initial, indent=2)}\n```\n"
        result += "Goal State:\n"
        result += f"```json\n{json.dumps(goal, indent=2)}\n```\n\n"
        result += self._format_step_block("0", initial, goal)

        mid = len(solution) // 2
        step_counter = 1
        for step in solution[:mid]:
            result += f"{step_counter}. ACTION: {step['action']}\n"
            result += self._format_step_block(str(step_counter), step["state"], goal)
            step_counter += 1

        for r_idx, rand_path in enumerate(randoms, 1):
            result += f"Backtracking to {step_counter - mid}:\n"
            for j, step in enumerate(rand_path):
                result += f"{step_counter}.{j + 1} ACTION: {step['action']}\n"
                result += self._format_step_block(f"{step_counter}.{j + 1}", step["state"], goal)
            result += f"Backtracking to the initial state:\n"
            result += self._format_step_block("0", initial, goal)

        for step in solution[mid:]:
            result += f"{step_counter}. ACTION: {step['action']}\n"
            result += self._format_step_block(str(step_counter), step["state"], goal)
            step_counter += 1

        result += "\nExact solution:\n"
        for i, step in enumerate(solution, 1):
            result += f"{i}. {step['action']}\n"
        return result

    def build_prompt(self, initial_state_json: Dict, goal_state_json: Dict) -> str:
        prompt = f"""# AoT+ Planning with JSON State and Goal Tracking

## Domain Description:
{self.domain_description}

### Instructions:
1. States are represented as JSON objects.
2. After each action (prefixed "ACTION:"), output a JSON block containing:
   - "step": the hierarchical step ID (e.g. "1", "1.2", "1.2.1")
   - "state": the full updated state
   - "goal_progress": which goal predicates are "satisfied" vs "unsatisfied"
3. Explore alternatives with hierarchical numbering; backtrack when a path fails.
4. End with a numbered "Exact solution:" list.

## Examples:
"""
        for ex in self.examples:
            prompt += self._format_example(ex) + "\n---\n\n"

        init_progress = self._check_goal_progress(initial_state_json, goal_state_json)
        init_block = {
            "step": "0",
            "state": initial_state_json,
            "goal_progress": init_progress,
        }

        prompt += f"""## Problem to Solve:

Starting State:
```json
{json.dumps(initial_state_json, indent=2)}
```
Goal State:
```json
{json.dumps(goal_state_json, indent=2)}
```

```json
{json.dumps(init_block, indent=2)}
```

Solve step by step with hierarchical numbering, tracking goal progress at each step. End with "Exact solution:".
"""
        return prompt


class AoTPlusJSON:
    """AoT+ with JSON state representation."""

    def __init__(self, domain_description: str, llm_provider: Optional[str] = None):
        self.llm_client = get_llm_client(llm_provider)
        self.prompt_builder = AoTPlusJSONPromptBuilder(domain_description)

    def add_example(self, initial_state_json: Dict, goal_state_json: Dict,
                    solution_trajectory: List[Dict], random_trajectories: List[List[Dict]]) -> None:
        self.prompt_builder.add_example(
            initial_state_json, goal_state_json, solution_trajectory, random_trajectories
        )

    def solve(self, initial_state_json: Dict, goal_state_json: Dict,
              temperature: float = 0.0, max_tokens: int = 4000) -> str:
        prompt = self.prompt_builder.build_prompt(initial_state_json, goal_state_json)
        return self.llm_client.generate(prompt, temperature, max_tokens)

    def extract_plan(self, solution_text: str) -> List[str]:
        steps = []
        solution_match = re.search(r"[Ee]xact solution:?(.*?)(?:$)", solution_text, re.DOTALL)
        if solution_match:
            numbered_pattern = r"(\d+)[\.:\)]\s+(.*?)(?=\n\d+[\.:\)]|\Z)"
            for m in re.finditer(numbered_pattern, solution_match.group(1), re.DOTALL):
                step = m.group(2).strip()
                if step:
                    steps.append(step)
        if not steps:
            for m in re.finditer(r"ACTION:\s*(.*?)(?=ACTION:|STATE|\n\n|```|$)", solution_text, re.DOTALL):
                action = m.group(1).strip()
                if action:
                    steps.append(action)
        return steps


class AoTPlusJSONGoalTracking:
    """AoT+ with JSON state and explicit goal predicate tracking."""

    def __init__(self, domain_description: str, llm_provider: Optional[str] = None):
        self.llm_client = get_llm_client(llm_provider)
        self.prompt_builder = AoTPlusJSONGoalTrackingPromptBuilder(domain_description)

    def add_example(self, initial_state_json: Dict, goal_state_json: Dict,
                    solution_trajectory: List[Dict], random_trajectories: List[List[Dict]]) -> None:
        self.prompt_builder.add_example(
            initial_state_json, goal_state_json, solution_trajectory, random_trajectories
        )

    def solve(self, initial_state_json: Dict, goal_state_json: Dict,
              temperature: float = 0.0, max_tokens: int = 4000) -> str:
        prompt = self.prompt_builder.build_prompt(initial_state_json, goal_state_json)
        return self.llm_client.generate(prompt, temperature, max_tokens)

    def extract_plan(self, solution_text: str) -> List[str]:
        steps = []
        solution_match = re.search(r"[Ee]xact solution:?(.*?)(?:$)", solution_text, re.DOTALL)
        if solution_match:
            numbered_pattern = r"(\d+)[\.:\)]\s+(.*?)(?=\n\d+[\.:\)]|\Z)"
            for m in re.finditer(numbered_pattern, solution_match.group(1), re.DOTALL):
                step = m.group(2).strip()
                if step:
                    steps.append(step)
        if not steps:
            for m in re.finditer(r"ACTION:\s*(.*?)(?=ACTION:|```|\n\n|$)", solution_text, re.DOTALL):
                action = m.group(1).strip()
                if action:
                    steps.append(action)
        return steps
