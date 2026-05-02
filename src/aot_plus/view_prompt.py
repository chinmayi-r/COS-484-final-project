from src.domains.alfworld import ALFWorldDomain, ALFWorldState
from src.aot_plus.core import AoTPlusPromptBuilder

def main():
    domain_description = ALFWorldDomain.get_domain_description()

    builder = AoTPlusPromptBuilder(
        domain_description,
        state_transition_fn=ALFWorldState.apply_action,
    )

    initial_state, goal_state, solution, random_trajectories = (
        ALFWorldDomain.generate_example_problem()
    )

    builder.add_example(
        initial_state=initial_state,
        goal_state=goal_state,
        solution_trajectory=solution,
        random_trajectories=random_trajectories,
    )

    # Use the same example as the "problem to solve" just so you can view
    # the complete one-shot prompt structure.
    prompt = builder.build_prompt(
        initial_state=initial_state,
        goal_state=goal_state,
    )

    print(prompt)


if __name__ == "__main__":
    main()
