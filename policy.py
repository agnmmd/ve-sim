import random
import numpy as np

from abc import ABC, abstractmethod

class Policy(ABC):
    def __init__(self, env):
        self.env = env

    @abstractmethod
    def match_task_and_car(self, tasks, cars):
        """Select a task and a car based on the policy."""
        pass

    def select_car(self, task, cars):
        selected_car = None
        best_completion_time = float('inf')

        # # First try to compute locally if the source car is idle
        # if task.source_car in self.get_idle_cars():
        #     selected_car = task.source_car
        #     best_completion_time = self.calculate_completion_time(task.source_car, task)

        # NOTE: The iteration goes through all cars, not only idle cars. But schedule_task() only executes if there are idle cars
        # for car in self.cars:
        for car in cars:
            completion_time = self.calculate_completion_time(car, task)

            if ((self.env.now + completion_time) <= (task.time_of_arrival + task.deadline)) and (completion_time < best_completion_time):
                selected_car = car
                best_completion_time = completion_time
                print(f"  -> Best car updated to Car {car.id} with Completion Time {completion_time}")
            else:
                print(f"  -> Car {car.id} not suitable for Task {task.id}, because it can either not meet the deadline or doesn't provide a better completion time.")

        return selected_car

    def calculate_completion_time(self, car, task):
        waiting_time = car.get_remaining_time() + car.calculate_waiting_time()
        processing_time = car.calculate_processing_time(task)
        completion_time = waiting_time + processing_time

        print(f"Evaluating Car {car.id} for Task {task.id}:")
        print(f"  Current Time: {self.env.now}")
        print(f"  Waiting Time: {waiting_time}")
        print(f"  Processing Time: {processing_time}")
        print(f"  Relative Completion Time: {completion_time}")
        print(f"  Task Time of Arrival: {task.time_of_arrival}")
        print(f"  Task Deadline: {task.deadline}")
        print(f"  Estimated Task Completion Time: {self.env.now + completion_time}")

        return completion_time

class RandomPolicy(Policy):
    def match_task_and_car(self, tasks, cars):
        if tasks and cars:
            task = np.random.choice(tasks)
            car = np.random.choice(cars)
            return task, car
        return None, None

class EarliestDeadlinePolicy(Policy):
    def match_task_and_car(self, tasks, cars):
        if tasks and cars:
            task = min(tasks, key=lambda t: t.time_of_arrival + t.deadline)
            car = self.select_car(task, cars)
            return task, car
        return None, None

class LowestComplexityPolicy(Policy):
    def match_task_and_car(self, tasks, cars):
        if tasks and cars:
            task = min(tasks, key=lambda t: t.complexity)
            car = self.select_car(task, cars)
            return task, car
        return None, None
