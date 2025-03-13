from policy import Policy, RandomPolicy, EarliestDeadlinePolicy, LowestComplexityPolicy  # Add more policies as needed

POLICY_REGISTRY = {
    "random": RandomPolicy,
    "earliest_deadline": EarliestDeadlinePolicy,
    "lowest_complexity": LowestComplexityPolicy,
}

def get_policy(policy_name: str, env) -> Policy:
    try:
        policy_class = POLICY_REGISTRY[policy_name]
    except KeyError:
        raise ValueError(f"Unknown policy '{policy_name}'. Available: {list(POLICY_REGISTRY.keys())}")
    return policy_class(env)