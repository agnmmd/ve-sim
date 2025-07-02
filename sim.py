import random
from input_manager import InputManager
import numpy as np
import simpy

class Sim:
    _sim_instance = None
    _environment = simpy.Environment()
    _task_id_counter = -1
    _car_id_counter = -1
    sim_parameters = InputManager.get_scenario_args()

    def __new__(cls):
        if cls._sim_instance is None:
            cls._sim_instance = super().__new__(cls)
            # Set random seed ONCE here
            cls.seed = cls.get_parameter("seed")
            np.random.seed(cls.seed)
            random.seed(cls.seed)
        return cls._sim_instance

    @classmethod
    def reset(cls, episode):
        cls._sim_instance = None
        cls._environment = simpy.Environment()
        cls._task_id_counter = -1
        cls._car_id_counter = -1
        cls.sim_parameters = InputManager.get_scenario_args(episode)
        # reseed the random at each episode.
        cls.seed = cls.get_parameter("seed")
        np.random.seed(cls.seed)
        random.seed(cls.seed)

    @classmethod
    def set_task_id(cls):
        cls._task_id_counter += 1
        return cls._task_id_counter

    @classmethod
    def set_car_id(cls):
        cls._car_id_counter += 1
        return cls._car_id_counter
    
    @classmethod
    def get_env(cls):
        """Get the simulation environment"""
        return cls._environment
    
    @classmethod
    def sim_clock(cls, time_interval):
        """Just a timer that progresses time until the simulation ends"""
        while True:
            print("SimPy time: ", cls.env.now)
            yield cls.env.timeout(time_interval)

    @classmethod
    def get_parameter(cls, parameter):
        # Ensure parameters exist before accessing
        if parameter not in cls.sim_parameters:
            raise KeyError(f"Parameter '{parameter}' not found in simulation parameters")
        if callable(cls.sim_parameters[parameter]):
            return cls.sim_parameters[parameter]()
        return cls.sim_parameters[parameter]


_sim_init = Sim()