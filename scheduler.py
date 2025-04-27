from utils import print_color
import random
import numpy as np
from stats import Statistics
from sim import Sim

class Scheduler:
    def __init__(self, traci, policy) -> None:
        self.traci = traci
        self.policy = policy
        self.cars = list()
        self.schedule = dict()
        self.static_cars = []
        self.previous_cars = set()
        self.previous_busy_cars = set()

    def generated_tasks_exist(self):
        return any(car.generated_tasks for car in self.cars)

    def get_generated_tasks(self):
        return [task for car in self.cars for task in car.generated_tasks]

    def get_assigned_tasks(self):
        return [task for car in self.cars for task in car.assigned_tasks]

    def get_idle_cars(self):
        return [car for car in self.cars if car.idle]

    def idle_cars_exist(self):
        return any(car.idle for car in self.cars)

    def has_new_cars(self):
        current_cars = set(self.cars)
        new_cars = current_cars - self.previous_cars
        self.previous_cars = current_cars
        return bool(new_cars)

    def car_was_busy_now_is_idle(self):
        currently_idle_cars = {car for car in self.cars if car.idle}
        new_idle_cars = self.previous_busy_cars & currently_idle_cars
        self.previous_busy_cars = {car for car in self.cars if not car.idle}
        return bool(new_idle_cars)
    
    def schedule_tasks(self):
        while True:
            self.cars = self.static_cars + self.traci.get_subscribed_vehicles_list()
            print_color(f"\n================== [Log] time: {Sim.get_env().now} ==================","93")
            self.print_schedule("Current State")

            # If we don't do the filetering the scheduler will keep assigning the tasks
            # Filter out the tasks whose deadline has expired
            for task in self.get_generated_tasks():
                if Sim.get_env().now >= (task.time_of_arrival + task.deadline):
                    print(f"The deadline of Task {task.id} is in the past; Removing it!")
                    # task.status = 3
                    Statistics.save_task_stats(task, "NA")
                    task.source_car.generated_tasks.remove(task) # Remove the task from the system

            while self.generated_tasks_exist() and self.idle_cars_exist():
                # By default filter out the tasks with status 5, however if a new car has joined consider status 5 tasks as well
                tasks = [t for t in self.get_generated_tasks() if t.status != 5]
                if self.has_new_cars() or self.car_was_busy_now_is_idle():
                    # print("True")
                    tasks = self.get_generated_tasks()

                selected_task, selected_car = self.policy.match_task_and_car(tasks, self.get_idle_cars())

                # NOTE: This case probably never occurs.
                if selected_task is None:
                    break

                if selected_car is None:
                    selected_task.status = 5
                    # noped_tasks.append(selected_task)
                    break

                print(f"Car {selected_car.id} <-- Task {selected_task.id}")

                # Housekeeping
                selected_car.assigned_tasks.append(selected_task) # Assign task to the selected car
                selected_task.source_car.generated_tasks.remove(selected_task) # Remove task from the list of generated tasks
                selected_task.status = 1
                selected_car.idle = False

                # Spawn processes for processing the tasks
                process = Sim.get_env().process(selected_car.process_task(selected_task))
                selected_car.active_processes.append(process)

            # Print state after assignments are finished
            print_color("----------------------------------------------------","95")
            self.print_schedule("After Scheduling")
            offset = 0.00001 * np.random.random() * np.random.choice([-1, 1])
            yield Sim.get_env().timeout(0.5 + offset)  # Check for tasks every unit of time

    def print_schedule(self, string):
        idle_cars_ids = [car.id for car in self.get_idle_cars()]
        idle_tasks_ids = [task.id for task in self.get_generated_tasks()]
        assigned_tasks_ids = [task.id for task in self.get_assigned_tasks()]

        print(f"\n[{string}:]")
        print("Idle Cars:\t", idle_cars_ids)
        print("Idle Tasks:\t", idle_tasks_ids)
        print("Assigned Tasks:\t", assigned_tasks_ids)

        # if not self.schedule:
        #     print("\n[The schedule is empty!]\n")
        # else:
        #     print("\n---------------------------")
        #     print("car_id | task_id | deadline")
        #     print("---------------------------")
        #     for car_id, task in self.schedule.items():
        #         if task:
        #             print(f"{car_id:6} | {task.id:7} | {'X' * int(task.deadline)}")
        #         else:
        #             print(f"{car_id:6} | {task.id:7} | Task Not Found")
        #     print("---------------------------\n")

    def filter_tasks(self):
        """
        If there is no suitable car for the task, the task will not be picked unless a new car enters the Region of Interest (ROI).
        """
        tasks = self.get_generated_tasks()
        return [task for task in tasks if (self.has_new_cars() and task.status == 5) or task.status != 5]

    # Static car methods
    def register_static_car(self, cars_list, remove_after_dwell_time=False):
        for car in cars_list:
            self.static_cars.append(car)

            if(remove_after_dwell_time):
                Sim.get_env().process(self.remove_after_dwell_time(car))

    def unregister_static_car(self, car):
        if car in self.static_cars:
            self.static_cars.remove(car)
            print(f"Car {car.id} removed from the system at time {Sim.get_env().now}")

    def remove_static_car_after_dwell_time(self, car):
        yield Sim.get_env().timeout(car.dwell_time)
        self.unregister_static_car(car)
        car.finish()

    def get_reordered_tasks(self, noped_tasks):
        generated_tasks = self.get_generated_tasks()
        unevaluated_tasks = [task for task in generated_tasks if task not in noped_tasks]
        return unevaluated_tasks + [task for task in generated_tasks if task in noped_tasks]