from stats import Statistics
from task import Task
import simpy
import random
from input_manager import InputManager

class Car:
    def __init__(self, env, sim, speed = None, position = None):
        self.env = env
        self.sim = sim

        # Parameters
        self.id = "c" + str(self.sim.set_car_id())
        self.processing_power = InputManager.scenario_args['car_processing_power']()
        self.num_tasks = InputManager.scenario_args['task_generation']()
        self.dwell_time = 10
        self.processor = simpy.Resource(self.env, capacity=1)

        self.idle = True
        self.current_task = None
        self.generated_tasks = []
        self.assigned_tasks = []
        self.active_processes = []

        # Mobility
        self.speed = speed
        self.position = position

        # Statistics
        self.generated_tasks_count = 0
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
            self.generated_tasks_count += 1

    def generate_tasks_static(self):
        """
        Tasks generated with this method will have the same time of arrival (TOA)
        """
        self.generated_tasks = [Task(self.env, self.sim, self) for _ in range(self.num_tasks)]
        for task in self.generated_tasks:
            print(f"Car {self.id} generated Task {task.id}: {task.__dict__}")
            self.generated_tasks_count += 1

    def process_task(self, selected_task):
        try:
            with self.processor.request() as req:
                yield req   # Wait to acquire the requested resource

                # Housekeeping
                assert(selected_task == self.assigned_tasks[0])
                self.current_task = self.assigned_tasks.pop(0)
                self.current_task.processing_start = self.env.now
                self.current_task.status = 2

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
        except simpy.Interrupt:
            print(f"Process for car {self.id} interrupted!")

            # Record the statistics for the task who's process was interrupted
            if(self.current_task):
                self.current_task.status = 4
                Statistics.save_task_stats(self.current_task, self.id)
                self.current_task = None
        finally:
            self.idle = True
            if self.env.active_process in self.active_processes:
                self.active_processes.remove(self.env.active_process)


    def calculate_waiting_time(self):
        return sum(task.complexity / self.processing_power for task in self.assigned_tasks)

    def calculate_processing_time(self, task):
        return task.complexity / self.processing_power
    
    def get_remaining_time(self):
        if self.current_task is None:
            return 0
        remaining_time = (self.current_task.complexity / self.processing_power) - (self.env.now - self.current_task.processing_start)
        return remaining_time

    def update(self, speed, position):
        self.speed = speed
        self.position = position

    def finish(self):
        """
        This function is called in the end of the lifetime of a vehicle.
        For static vehicles: When the dwell time expires
        For dynamic vehicles: When they leave the scenario (traci)
        """
        Statistics.save_car_stats(self, self.env.now)

        for task in self.assigned_tasks:
            Statistics.save_task_stats(task, "NA")
        self.assigned_tasks.clear()

        for task in self.generated_tasks:
            Statistics.save_task_stats(task, "NA")
        self.generated_tasks.clear()

        # Interrupt the processes associated with this Car
        for process in list(self.active_processes):  # Copy the list to avoid modifying it during iteration
            if process.is_alive:  # Check if the process is still running
                process.interrupt()  # Interrupt the process
        self.active_processes.clear()  # Clear the list of processes