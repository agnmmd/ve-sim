import numpy as np
import random

class Task:
    def __init__(self, env, sim, source_car):
        self.env = env
        self.sim = sim

        self.id = "t" + str(self.sim.set_task_id())
        self.source_car = source_car
        self.time_of_arrival = self.env.now

        self.deadline = self.sim.get_im_parameter('task_deadline')()
        self.priority = self.sim.get_im_parameter('task_priority')()
        self.complexity = self.sim.get_im_parameter('task_complexity')()
        self.processing_start = -1
        self.processing_end = -1

        # Statistic
        # A task can have status
        # 0 = generated (i.e., never assigned)
        # 1 = assigned
        # 2 = processed (before deadline)
        # 3 = deadline expired
        # 4 = interrupted
        # 5 = unassignable (i.e., no resource that can process it before deadline)
        # 6 = processed (after deadline)
        self.status = 0

    @classmethod
    def to_dict(cls, task):
        if task is None: 
            return {'deadline': 0, 'complexity': 0}
        else:
            return{
            'deadline': task.deadline,
            'complexity': task.complexity,
            }