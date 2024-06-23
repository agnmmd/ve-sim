from stats import Statistics
from task import Task
import simpy
import random

class Car:
    def __init__(self, env, sim, speed = None, position = None):
        self.env = env
        self.sim = sim

        self.id = "c" + str(self.sim.set_car_id())
        self.generated_tasks = []
        self.processing_power = 2
        self.idle = True
        self.dwell_time = 10
        self.assigned_tasks = []
        self.processor = simpy.Resource(self.env, capacity=1)
        self.current_task = None

        # Statistics
        self.successful_tasks = 0
        self.total_processing_time = 0
        self.processed_tasks_count = 0
        self.time_of_arrival = self.env.now


    def generate_tasks(self):
        while True:
            yield self.env.timeout(random.expovariate(1.0/5))
            task = Task(self.env, self.sim, self)
            self.generated_tasks.append(task)
            print(f"Car {self.id} generated a Task: {task.__dict__}")

    def generate_tasks_static(self, num_tasks):
        """
        Tasks generated with this method will have the same time of arrival (TOA)
        """
        self.generated_tasks = [Task(self.env, self.sim, self) for _ in range(num_tasks)]
        for task in self.generated_tasks:
            print(f"Car {self.id} generated Task {task.id}: {task.__dict__}")

    def process_task(self, selected_task):
        with self.processor.request() as req:
            yield req

            # Housekeeping
            assert(selected_task == self.assigned_tasks[0])
            self.current_task = self.assigned_tasks.pop(0)
            self.current_task.processing_start = self.env.now
            
            processing_time = self.calculate_processing_time(selected_task)
            # Start processing
            yield self.env.timeout(processing_time)
            # Finished processing

            # Update metrics
            self.total_processing_time += processing_time
            self.processed_tasks_count += 1
            if self.env.now - self.current_task.time_of_arrival <= self.current_task.deadline:
                self.successful_tasks += 1

            print(f"@t={self.env.now}, Car {self.id} finished computing Task: {selected_task.id}!")
            self.current_task.processing_end = self.env.now
            Statistics.save_task_stats(self.current_task, self.id)
            self.current_task = None
        self.idle = True

    def calculate_waiting_time(self):
        return sum(task.complexity / self.processing_power for task in self.assigned_tasks)

    def calculate_processing_time(self, task):
        return task.complexity / self.processing_power
    
    def get_remaining_time(self):
        if self.current_task is None:
            return 0
        remaining_time = (self.current_task.complexity / self.processing_power) - (self.env.now - self.current_task.processing_start)
        return remaining_time
    def finish(self):
        Statistics.save_car_stats(self, self.env.now)