from sim import Sim
import random

class Task:
    def __init__(self, source_car):
        self.id = "t" + str(Sim.set_task_id())
        self.source_car = source_car
        self.time_of_arrival = Sim.env.now
        self.deadline = 2 #random.randint(1, 10)
        self.priority = random.randint(0, 3)
        self.complexity = 2 #random.randint(1,6)
        self.processing_start = None
        self.processing_end = None