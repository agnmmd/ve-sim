import simpy
import random

class Sim:
    env = simpy.Environment()

    _random_seed_value = 42
    random.seed(_random_seed_value)

    _task_id_counter = 0
    _car_id_counter = 0

    @classmethod
    def set_task_id(cls):
        cls._task_id_counter += 1
        return cls._task_id_counter
    
    @classmethod
    def set_car_id(cls):
        cls._car_id_counter += 1
        return cls._car_id_counter
