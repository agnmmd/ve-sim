import importlib

CLASS_REGISTRY = {
    "RandomPolicy": "policy.RandomPolicy",
    "EarliestDeadline": "policy.EarliestDeadlinePolicy",
    "LowestComplexity": "policy.LowestComplexityPolicy",
    "DQNPolicy": "policy.DQNPolicy",
    "TaskSchedulingEnv": "rl_other.TaskSchedulingEnv",
    "DQNAgent": "rl_other.DQNAgent"
}

def get_class_by_full_path(full_class_name: str):
    try:
        module_path, class_name = full_class_name.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError, ValueError) as e:
        raise ValueError(f"Could not import '{full_class_name}': {e}")

def load_class(short_class_name: str, *args, **kwargs):
    if not short_class_name or short_class_name == "None":
        return None
    try:
        full_class_path = CLASS_REGISTRY[short_class_name]
        cls = get_class_by_full_path(full_class_path)
        return cls(*args, **kwargs)
    except KeyError:
        raise ValueError(f"Unknown class key '{short_class_name}'. Available: {list(CLASS_REGISTRY.keys())}")
