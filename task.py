import random

from input_manager import InputManager
class Task:
    def __init__(self, env, sim, source_car):
        self.env = env
        self.sim = sim

        self.id = "t" + str(self.sim.set_task_id())
        self.source_car = source_car
        self.time_of_arrival = self.env.now

        self.deadline = InputManager.scenario_args['task_deadline']()
        self.priority = InputManager.scenario_args['task_priority']()
        self.complexity = InputManager.scenario_args['task_complexity']()
        self.processing_start = -1
        self.processing_end = -1

        # Statistic
        # A task can have status
        # 0 = generated / never assigned
        # 1 = assigned
        # 2 = processed
        # 3 = deadline expired
        # 4 = interrupted
        # 5 = requeue
        self.status = 0