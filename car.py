from stats import Statistics
from task import Task
import simpy
import random
from sim import Sim
class Car:
    def __init__(self,speed = None, position: tuple[float, float] = None):
        
        # Parameters
        self.id = "c" + str(Sim.set_car_id())
        self.processing_power = Sim.get_parameter('car_processing_power')
        self.num_tasks = Sim.get_parameter('task_generation')
      #  self.lambda_exp = Sim.get_parameter('lambda_exp')
        self.dwell_time = 10
        self.processor = simpy.Resource(Sim.get_env(), capacity=1)

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
        self.time_of_arrival = Sim.get_env().now

    def generate_tasks(self):
        try:
            while True:
                # yield Sim.get_env().timeout(random.expovariate(self.lambda_exp))
                yield Sim.get_env().timeout(self.num_tasks)
                task = Task(self)
                self.generated_tasks.append(task)
                print(f"Car {self.id} generated a Task: {task.__dict__}")
                self.generated_tasks_count += 1
        except simpy.Interrupt:
            print(f"Process generate_tasks() for car {self.id} interrupted!")

    def generate_tasks_static(self):
        """
        Tasks generated with this method will have the same time of arrival (TOA)
        """
        self.generated_tasks = [Task(self) for _ in range(self.num_tasks)]
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
                self.current_task.processing_start = Sim.get_env().now

                processing_time = self.calculate_processing_time(selected_task)
                # Start processing
                yield Sim.get_env().timeout(processing_time)
                # Finished processing

                # Update metrics
                self.total_processing_time += processing_time
                self.processed_tasks_count += 1
                if Sim.get_env().now - self.current_task.time_of_arrival <= self.current_task.deadline:
                    self.successful_tasks += 1
                    self.current_task.status = 2
                else:
                    self.current_task.status = 6

                print(f"@t={Sim.get_env().now}, Car {self.id} finished computing Task: {selected_task.id}!")
                self.current_task.processing_end = Sim.get_env().now
                Statistics.save_task_stats(self.current_task, self.id)
                self.current_task = None

        except simpy.Interrupt:
            print(f"Process process_task() for car {self.id} interrupted!")

            # Record the statistics for the task who's process was interrupted
            if(self.current_task):
                self.current_task.status = 4
                Statistics.save_task_stats(self.current_task, self.id)
                self.current_task = None
        finally:
            self.idle = True
            if Sim.get_env().active_process in self.active_processes:
                self.active_processes.remove(Sim.get_env().active_process)

    def calculate_waiting_time(self):
        return sum(task.complexity / self.processing_power for task in self.assigned_tasks)

    def calculate_processing_time(self, task):
        return task.complexity / self.processing_power
    
    def get_remaining_time(self):
        if self.current_task is None:
            return 0
        remaining_time = (self.current_task.complexity / self.processing_power) - (Sim.get_env().now - self.current_task.processing_start)
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
        Statistics.save_car_stats(self, Sim.get_env().now)

        for task in self.assigned_tasks:
            Statistics.save_task_stats(task, "NA")
        self.assigned_tasks.clear()

        for task in self.generated_tasks:
            Statistics.save_task_stats(task, "NA")
        self.generated_tasks.clear()

        # Interrupt the processes associated with this Car
        for process in list(self.active_processes):  # Copy the list to avoid modifying it during iteration
            if process.is_alive:
                process.interrupt()
        self.active_processes.clear()

        # NOTE: Make reporting of tasks statistics a method of Tasks class