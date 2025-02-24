import random
from input_manager import InputManager
import numpy as np

class Sim:
    # Class-level initialization of scenario arguments
    sim_parameters = InputManager.get_scenario_args()

    def __init__(self):
        # Initialize ID counters
        self._task_id_counter = -1
        self._car_id_counter = -1

        # Use class method to get run and repetition
        self.run = self.get_parameter('run')
        self.repetition = self.get_parameter('repetition')
        
        # Set random seed for reproducibility
        np.random.seed(self.repetition)
        random.seed(self.repetition)

    def set_task_id(self):
        self._task_id_counter += 1
        return self._task_id_counter

    def set_car_id(self):
        self._car_id_counter += 1
        return self._car_id_counter

    @classmethod
    def get_parameter(cls, parameter):
        # Ensure parameters exist before accessing
        if parameter not in cls.sim_parameters:
            raise KeyError(f"Parameter '{parameter}' not found in simulation parameters")
        return cls.sim_parameters[parameter]
    
    def get_im_parameter(self, parameter):
        return self.get_parameter(parameter)