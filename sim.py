import random

class Sim:
    def __init__(self):
        self._task_id_counter = -1
        self._car_id_counter = -1

    run = -1
    reperition = -1
    random.seed(42)

    policy_name = None
    policy_function = None

    def set_task_id(self):
        self._task_id_counter += 1
        return self._task_id_counter

    def set_car_id(self):
        self._car_id_counter += 1
        return self._car_id_counter

    @classmethod
    def update_sim_variables(cls, run, repetition, policy_function):
        cls.run = run
        cls.repetition = repetition
        random.seed(cls.repetition)
        cls.policy_function = policy_function
        cls.policy_name = cls.extract_method_name(policy_function)

    @staticmethod
    def extract_method_name(method):
        return method.__name__[2:]