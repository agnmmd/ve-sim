import simpy
import random

class Sim:
    def __init__(self):
        self._task_id_counter = 0
        self._car_id_counter = 0

    _random_seed_value = 42
    random.seed(_random_seed_value)

    def set_task_id(self):
        self._task_id_counter += 1
        return self._task_id_counter
    
    def set_car_id(self):
        self._car_id_counter += 1
        return self._car_id_counter
