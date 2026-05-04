"""Evaluate ALFWorld AoT+ traces for state-tracking errors."""

import argparse
import glob
import json
import os
import random
import re

from src.aot_plus.core import AoTPlusPromptBuilder
from src.domains.alfworld import ALFWorldDomain, ALFWorldState
from src.models.llm_client import get_llm_client

DYNAMIC_PREDICATES = {
    "atLocation",
    "objectAtLocation",
    "inReceptacle",
    "opened",
    "holds",
    "holdsAny",
    "isClean",
    "isHot",
    "isCool",
    "isOn",
    "isToggled",
    "isSliced",
}

CAPABILITY_WORDS = {
    "pickupable": "pickupable",
    "moveable": "moveable",
    "cleanable": "clean",
    "heatable": "heatable",
    "coolable": "coolable",
    "toggleable": "toggleable",
    "sliceable": "sliceable",
    "isClean": "clean",
    "isHot": "hot",
    "isCool": "cool",
    "isOn": "on",
    "isToggled": "toggled",
    "isSliced": "sliced",
}

STATE_HEADER_RE = re.compile(
    r"^STATE(?:\s+([0-9]+(?:\.[0-9]+)*))?\s*:\s*(.*)$",
    re.IGNORECASE,
)


def parse_sexpr(text):
    """Parse a Lisp/PDDL S-expression string into nested Python lists."""
    tokens = re.findall(r"\(|\)|[^\s()]+", text)
    idx = 0

    def parse_node():
        nonlocal idx
        if idx >= len(tokens):
            return None
        tok = tokens[idx]
        if tok != "(":
            idx += 1
            return tok

        idx += 1
        node = []
        while idx < len(tokens) and tokens[idx] != ")":
            node.append(parse_node())
        idx += 1
        return node

    return parse_node()


def extract_problem_sections(pddl_problem):
    """Return the objects, init, and goal blocks from a PDDL problem string."""
    patterns = {
        "objects": r"\(:objects(.*?)\)\s*\(:init",
        "init": r"\(:init(.*?)\)\s*\(:goal",
        "goal": r"\(:goal\s*(.*?)\)\s*\)\s*$",
    }
    matches = {name: re.search(pattern, pddl_problem, re.S) for name, pattern in patterns.items()}
    if not all(matches.values()):
        return None, None, None
    return matches["objects"].group(1), matches["init"].group(1), matches["goal"].group(1)


def parse_typed_objects(objects_block):
    """Parse PDDL typed object declarations into a type -> objects mapping."""
    objects_by_type = {}
    current_names = []
    tokens = objects_block.replace("\n", " ").split()

    i = 0
    while i < len(tokens):
        if tokens[i] == "-" and i + 1 < len(tokens):
            objects_by_type.setdefault(tokens[i + 1], []).extend(current_names)
            current_names = []
            i += 2
        else:
            current_names.append(tokens[i])
            i += 1
    return objects_by_type


def parse_init_atoms(init_block):
    """Extract predicate atoms from a PDDL :init block."""
    atoms = []
    for atom in re.findall(r"\(([^()]+)\)", init_block):
        parts = atom.split()
        if parts:
            atoms.append(parts)
    return atoms


def clean_type(type_name):
    """Remove ALFWorld's trailing `Type` suffix when present."""
    return type_name[:-4] if type_name.endswith("Type") else type_name


def init_atoms_to_nl(atoms, max_facts):
    """Convert parsed init atoms into a compact natural-language state memo."""
    facts = []
    by_pred = {}
    for pred, *args in atoms:
        by_pred.setdefault(pred, []).append(args)

    object_types = {
        obj: clean_type(obj_type)
        for obj, obj_type in by_pred.get("objectType", [])
        if obj and obj_type
    }
    receptacle_types = {
        recep: clean_type(recep_type)
        for recep, recep_type in by_pred.get("receptacleType", [])
        if recep and recep_type
    }

    def obj_label(obj):
        return f"{object_types.get(obj, 'Object')} {obj}"

    def recep_label(recep):
        return f"{receptacle_types.get(recep, 'Receptacle')} {recep}"

    agents = set()
    located_receptacles = set()
    contained_objects = set()
    held_objects = set()

    for agent, loc in by_pred.get("atLocation", []):
        agents.add(agent)
        facts.append(f"Agent {agent} is at {loc}")

    for recep, loc in by_pred.get("receptacleAtLocation", []):
        located_receptacles.add(recep)
        facts.append(f"{recep_label(recep)} is at {loc}")

    for pred in ("openable", "opened"):
        for (recep,) in by_pred.get(pred, []):
            if recep in located_receptacles:
                facts.append(f"{recep_label(recep)} is {pred}")

    for obj, recep in by_pred.get("inReceptacle", []):
        contained_objects.add(obj)
        facts.append(f"{obj_label(obj)} is in {recep_label(recep)}")

    for agent, obj in by_pred.get("holds", []):
        agents.add(agent)
        held_objects.add(obj)
        facts.append(f"Agent {agent} is holding {obj_label(obj)}")

    for obj, loc in by_pred.get("objectAtLocation", []):
        if obj not in contained_objects and obj not in held_objects:
            facts.append(f"{obj_label(obj)} is at {loc}")

    for pred, word in CAPABILITY_WORDS.items():
        for (obj,) in by_pred.get(pred, []):
            facts.append(f"{obj_label(obj)} is {word}")

    for agent in sorted(agents):
        if not any(fact.startswith(f"Agent {agent} is holding ") for fact in facts):
            facts.append(f"Agent {agent} is holding nothing")

    deduped = []
    seen = set()
    for fact in facts:
        if fact in seen:
            continue
        seen.add(fact)
        deduped.append(fact)
        if len(deduped) >= max_facts:
            break
    return ", ".join(deduped)


def goal_to_text(node):
    """Render a parsed PDDL goal formula as readable text for the prompt."""
    if not isinstance(node, list) or not node:
        return "No goal."

    head = node[0]
    if head in {"and", "or"}:
        separator = "; " if head == "and" else " or "
        return separator.join(filter(None, (goal_to_text(child) for child in node[1:])))
    if head == "exists":
        return f"There exists an assignment such that {goal_to_text(node[-1])}"
    if head == "not" and len(node) > 1:
        return f"not ({goal_to_text(node[1])})"
    return f"{head}({' '.join(node[1:])})"


def state_to_facts(state):
    """Flatten an ALFWorldState into comparable predicate tuples."""
    fact_sources = [
        ("atLocation", state.at_location.items()),
        ("receptacleAtLocation", state.receptacle_at_location.items()),
        ("objectAtLocation", state.object_at_location.items()),
        ("inReceptacle", state.in_receptacle.items()),
        ("holds", state.holds.items()),
        ("receptacleType", state.receptacle_type.items()),
        ("objectType", state.object_type.items()),
    ]
    set_sources = [
        ("openable", state.openable),
        ("opened", state.opened),
        ("holdsAny", state.holds_any),
        ("pickupable", state.pickupable),
        ("moveable", state.moveable),
        ("cleanable", state.cleanable),
        ("heatable", state.heatable),
        ("coolable", state.coolable),
        ("toggleable", state.toggleable),
        ("sliceable", state.sliceable),
        ("isClean", state.is_clean),
        ("isHot", state.is_hot),
        ("isCool", state.is_cool),
        ("isOn", state.is_on),
        ("isToggled", state.is_toggled),
        ("isSliced", state.is_sliced),
    ]

    facts = {(pred, *args) for pred, pairs in fact_sources for args in pairs}
    facts |= {(pred, item) for pred, items in set_sources for item in items}
    facts |= {("canContain", recep_type, obj_type) for recep_type, obj_type in state.can_contain}
    return facts


def diff_facts(prev_facts, curr_facts):
    """Return added and removed facts between two fact sets."""
    return {
        "added": sorted(curr_facts - prev_facts),
        "removed": sorted(prev_facts - curr_facts),
    }


def parse_state_header(line):
    """Parse a STATE header line into `(numeric_label, state_text)` if possible."""
    match = STATE_HEADER_RE.match(line.strip())
    if not match:
        return None
    label_raw, content = match.group(1), (match.group(2) or "").strip()
    label = tuple(int(x) for x in label_raw.split(".")) if label_raw else None
    return label, content


def prefix_chain(leaf_label):
    """Return all numeric prefixes from root to leaf, e.g. (3,), (3, 1)."""
    return [leaf_label[:i] for i in range(1, len(leaf_label) + 1)] if leaf_label else []


def extract_actions_and_states(solution_text):
    """Extract the committed ACTION/STATE chain from an AoT+ trace.

    The model may explore sibling branches such as STATE 1, STATE 2, and STATE 3,
    then continue only one branch with dotted labels like STATE 3.1.1. This function
    uses the deepest dotted state label as the successful leaf, reconstructs its
    numeric prefix chain, and returns only the actions/states on that branch.
    If the dotted chain is incomplete, it falls back to the full parsed trace.
    """
    actions = []
    states = []
    label_to_pair = {}
    leading_states = []
    pending_action = None
    state_buffer = []
    last_leaf_label = None

    def flush():
        nonlocal pending_action, state_buffer, last_leaf_label
        if not pending_action:
            return

        segment = leading_states + state_buffer
        leading_states.clear()
        state_buffer = []
        action = pending_action
        pending_action = None
        if not segment:
            return

        actions.append(action)
        states.append(segment[-1][1])
        for label, state_text in segment:
            if label is not None:
                label_to_pair[label] = (action, state_text)
        if segment[-1][0] is not None:
            last_leaf_label = segment[-1][0]

    for raw_line in solution_text.splitlines():
        line = raw_line.strip()
        if line.startswith("ACTION:"):
            action = line[len("ACTION:") :].strip()
            while action.upper().startswith("ACTION:"):
                action = action[len("ACTION:") :].lstrip(": ")
            if not action:
                continue
            if pending_action is not None:
                flush()
            pending_action = action
            state_buffer = []
            continue

        parsed_state = parse_state_header(line)
        if parsed_state is None:
            continue
        if pending_action is None:
            leading_states.append(parsed_state)
        else:
            state_buffer.append(parsed_state)

    if pending_action is not None:
        flush()

    if last_leaf_label is None or not label_to_pair:
        return actions, states

    committed_actions = []
    committed_states = []
    for label in prefix_chain(last_leaf_label):
        if label not in label_to_pair:
            return actions, states
        action, state = label_to_pair[label]
        if committed_actions and committed_actions[-1] == action:
            committed_states[-1] = state
        else:
            committed_actions.append(action)
            committed_states.append(state)
    return committed_actions, committed_states


def fact_filter(predicates=None):
    """Build a filter that keeps only facts whose predicate is in the allowlist."""
    if predicates is None:
        predicates = DYNAMIC_PREDICATES
    predicates = set(predicates)
    return lambda facts: {fact for fact in facts if fact and fact[0] in predicates}


def evaluate_goal(node, fact_set, objects_by_type, env):
    """Evaluate a parsed PDDL goal against a concrete fact set."""
    if not isinstance(node, list) or not node:
        return False

    head = node[0]
    if head == "and":
        return all(evaluate_goal(child, fact_set, objects_by_type, env) for child in node[1:])
    if head == "or":
        return any(evaluate_goal(child, fact_set, objects_by_type, env) for child in node[1:])
    if head == "not":
        return not evaluate_goal(node[1], fact_set, objects_by_type, env)
    if head == "exists":
        return evaluate_exists(node[1], node[2], fact_set, objects_by_type, env)

    args = []
    for arg in node[1:]:
        if isinstance(arg, list) or (arg.startswith("?") and arg not in env):
            return False
        args.append(env[arg] if arg.startswith("?") else arg)
    return tuple([head] + args) in fact_set


def evaluate_exists(var_decl, body, fact_set, objects_by_type, env):
    """Evaluate an existential PDDL clause by trying all typed object bindings."""
    typed_vars = []
    i = 0
    while i < len(var_decl):
        if var_decl[i].startswith("?") and i + 2 < len(var_decl) and var_decl[i + 1] == "-":
            typed_vars.append((var_decl[i], var_decl[i + 2]))
            i += 3
        else:
            i += 1

    def backtrack(pos):
        if pos == len(typed_vars):
            return evaluate_goal(body, fact_set, objects_by_type, env)
        var, typ = typed_vars[pos]
        for candidate in objects_by_type.get(typ, []):
            env[var] = candidate
            if backtrack(pos + 1):
                return True
            del env[var]
        return False

    return backtrack(0)


def episode_diagnostics(actions, model_state_texts, initial_state_text, dynamic_predicates=None):
    """Compare model-reported states against canonical replayed states.

    For each action, this function applies the action to the true state, parses the
    model's reported next state, and records omission, hallucination, propagation,
    divergence, and accumulation diagnostics. `dynamic_predicates` controls which
    predicates count for delta-based scoring; passing an empty list disables delta
    scoring but still checks overall dynamic-state divergence.
    """
    if dynamic_predicates is None:
        divergence_filter = delta_filter = fact_filter()
    elif len(dynamic_predicates) == 0:
        divergence_filter = fact_filter()
        delta_filter = lambda _facts: set()
    else:
        divergence_filter = delta_filter = fact_filter(dynamic_predicates)

    step_count = min(len(actions), len(model_state_texts))
    if step_count == 0:
        return [], []

    true_text = initial_state_text
    true_prev = ALFWorldDomain.parse_state_description(initial_state_text)
    model_prev = ALFWorldDomain.parse_state_description(initial_state_text)
    prev_divergence = set()
    per_step = []
    divergence_history = []

    for i, action in enumerate(actions[:step_count]):
        model_curr = ALFWorldDomain.parse_state_description(model_state_texts[i])
        true_text = ALFWorldState.apply_action(true_text, action)
        true_curr = ALFWorldDomain.parse_state_description(true_text)

        model_prev_facts = state_to_facts(model_prev)
        model_curr_facts = state_to_facts(model_curr)
        true_prev_facts = state_to_facts(true_prev)
        true_curr_facts = state_to_facts(true_curr)

        delta_true = diff_facts(delta_filter(true_prev_facts), delta_filter(true_curr_facts))
        delta_model = diff_facts(delta_filter(model_prev_facts), delta_filter(model_curr_facts))

        expected_changes = set(delta_true["added"] + delta_true["removed"])
        model_changes = set(delta_model["added"] + delta_model["removed"])
        omission = sorted(expected_changes - model_changes)
        hallucination = sorted(model_changes - expected_changes)

        true_dyn = divergence_filter(true_curr_facts)
        model_dyn = divergence_filter(model_curr_facts)
        divergence = (true_dyn - model_dyn) | (model_dyn - true_dyn)
        divergence_history.append(set(divergence))

        per_step.append({
            "step": i + 1,
            "action": action,
            "true_state": str(true_curr),
            "model_state": str(model_curr),
            "delta_true": delta_true,
            "delta_model": delta_model,
            "propagation_error": not prev_divergence and bool(divergence),
            "omission_error": bool(omission),
            "hallucination_error": bool(hallucination),
            "omission_facts": omission,
            "hallucination_facts": hallucination,
            "divergence_facts": sorted(divergence),
        })

        prev_divergence = divergence
        true_prev = true_curr
        model_prev = model_curr

    accumulation = mark_accumulation_errors(per_step, divergence_history)
    return per_step, accumulation


def mark_accumulation_errors(per_step, divergence_history):
    """Mark accumulation when the same divergence persists after its first step."""
    for step in per_step:
        step["accumulation_error"] = False

    first_divergent = next((i for i, divergence in enumerate(divergence_history) if divergence), None)
    if first_divergent is None:
        return []

    tail = divergence_history[first_divergent:]
    if len(tail) < 2:
        return []

    common = set(tail[0])
    for divergence in tail[1:]:
        common &= divergence
    if not common:
        return []

    for i in range(first_divergent + 1, len(per_step)):
        per_step[i]["accumulation_error"] = True
    return sorted(common)


def evaluate_dataset(args):
    """Run evaluation over sampled ALFWorld problem files and write outputs."""
    files = glob.glob(os.path.join(args.dataset_root, "**", "game.tw-pddl"), recursive=True)
    random.Random(args.seed).shuffle(files)
    if args.max_examples > 0:
        files = files[:args.max_examples]

    llm_client = get_llm_client(args.provider)
    prompt_builder = AoTPlusPromptBuilder(
        ALFWorldDomain.get_domain_description(),
        use_json_state=False,
        state_transition_fn=None,
    )
    prompt_builder.add_example(*ALFWorldDomain.generate_example_problem())

    os.makedirs(args.output_dir, exist_ok=True)
    jsonl_path = os.path.join(args.output_dir, "alfworld_error_eval.jsonl")
    summary_path = os.path.join(args.output_dir, "alfworld_error_summary.json")
    totals = empty_totals()

    with open(jsonl_path, "w") as out:
        for path in files:
            record = evaluate_file(path, args, llm_client, prompt_builder, totals)
            out.write(json.dumps(record) + "\n")

    with open(summary_path, "w") as f:
        json.dump(totals, f, indent=2)

    return jsonl_path, summary_path, totals


def empty_totals():
    """Create the aggregate metrics dictionary used across the run."""
    return {
        "episodes": 0,
        "steps": 0,
        "propagation_errors": 0,
        "accumulation_errors": 0,
        "omission_errors": 0,
        "hallucination_errors": 0,
    }


def evaluate_file(path, args, llm_client, prompt_builder, totals):
    """Evaluate one ALFWorld problem file and return its JSONL record."""
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as exc:
        return {"file": path, "error": "json_parse_failed", "detail": str(exc)}

    objects_block, init_block, goal_block = extract_problem_sections(data.get("pddl_problem", ""))
    if not objects_block or not init_block or not goal_block:
        return {"file": path, "error": "pddl_sections_missing"}

    objects_by_type = parse_typed_objects(objects_block)
    initial_state_nl = init_atoms_to_nl(parse_init_atoms(init_block), args.max_init_facts)
    goal_node = parse_sexpr(goal_block)
    goal_state_nl = goal_to_text(goal_node)

    prompt = prompt_builder.build_prompt(initial_state_nl, goal_state_nl)
    llm_output = llm_client.generate(prompt, args.temperature, args.max_tokens)
    
    print("\n========== RAW MODEL OUTPUT ==========")
    print(llm_output)
    print("========== END RAW MODEL OUTPUT ==========\n")

    actions, states = extract_actions_and_states(llm_output)
    per_step, accumulation_facts = episode_diagnostics(actions, states, initial_state_nl)
    update_totals(totals, per_step)

    true_text = initial_state_nl
    for action in actions[: len(per_step)]:
        true_text = ALFWorldState.apply_action(true_text, action)
    final_true_state = ALFWorldDomain.parse_state_description(true_text)
    goal_true = evaluate_goal(goal_node, state_to_facts(final_true_state), objects_by_type, {})

    final_model_goal = False
    if states:
        final_model_state = ALFWorldDomain.parse_state_description(states[min(len(states), len(per_step)) - 1])
        final_model_goal = evaluate_goal(goal_node, state_to_facts(final_model_state), objects_by_type, {})

    return {
        "file": path,
        "initial_state_nl": initial_state_nl,
        "goal_state_nl": goal_state_nl,
        "goal_formula": goal_block.strip(),
        "actions_count": len(actions),
        "states_count": len(states),
        "aligned_steps": len(per_step),
        "goal_satisfied_true": final_true_state is not None and goal_true,
        "goal_satisfied_model": final_model_goal,
        "accumulation_facts": accumulation_facts,
        "steps": per_step,
        "raw_output": llm_output,
    }


def update_totals(totals, per_step):
    """Add one episode's step-level diagnostics to aggregate totals."""
    totals["episodes"] += 1
    totals["steps"] += len(per_step)
    totals["propagation_errors"] += sum(step["propagation_error"] for step in per_step)
    totals["accumulation_errors"] += sum(step["accumulation_error"] for step in per_step)
    totals["omission_errors"] += sum(step["omission_error"] for step in per_step)
    totals["hallucination_errors"] += sum(step["hallucination_error"] for step in per_step)


def parse_args():
    """Parse command-line arguments for the evaluator."""
    parser = argparse.ArgumentParser(description="Evaluate ALFWorld AoT+ state-tracking errors.")
    parser.add_argument("--dataset-root", default="/Users/evan/cos-484/data/json_2.1.1")
    parser.add_argument("--output-dir", default="/Users/evan/cos-484/COS-484-final-project/results")
    parser.add_argument("--provider", default=None, help="LLM provider for get_llm_client().")
    parser.add_argument("--max-examples", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=3000)
    parser.add_argument("--max-init-facts", type=int, default=180)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    """CLI entry point."""
    args = parse_args()
    jsonl_path, summary_path, totals = evaluate_dataset(args)
    print("Wrote episode results to:", jsonl_path)
    print("Wrote summary to:", summary_path)
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
