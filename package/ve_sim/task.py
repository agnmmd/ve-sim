from ve_sim.sim import Sim
class Task:
    def __init__(self, source_car):
        self.id = "t" + str(Sim.set_task_id())
        self.source_car = source_car
        self.time_of_arrival = Sim.get_env().now

        self.deadline = Sim.get_parameter('task_deadline')
        self.priority = Sim.get_parameter('task_priority')
        self.complexity = Sim.get_parameter('task_complexity')
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