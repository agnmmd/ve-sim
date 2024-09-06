import random

class Task:
    def __init__(self, env, sim, source_car):
        self.env = env
        self.sim = sim

        self.id = "t" + str(self.sim.set_task_id())
        self.source_car = source_car
        self.time_of_arrival = self.env.now
        self.deadline = 100 #random.randint(1, 10)
        self.priority = random.randint(0, 3)
        self.complexity = 2 #random.randint(1,6)
        self.processing_start = -1
        self.processing_end = -1

        # Statistic
        # A task can have status
        # 0 = generated
        # 1 = assigned
        # 2 = processed
        self.status = 0