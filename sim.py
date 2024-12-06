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
