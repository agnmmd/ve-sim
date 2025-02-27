from utils import print_color
import random
import numpy as np
from stats import Statistics

class Scheduler:
    def __init__(self, env, traci) -> None:
        self.env = env
        self.traci = traci
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

    # def update_previous_busy_cars(self):
    #     self.previous_busy_cars = {car for car in self.cars if not car.idle}

    # def car_was_busy_now_is_idle(self):
    #     currently_idle_cars = {car for car in self.cars if car.idle}
    #     cars_now_idle = self.previous_busy_cars & currently_idle_cars

    #     if cars_now_idle:
    #         return True
    #     return False

    def car_was_busy_now_is_idle_2(self):
        currently_idle_cars = {car for car in self.cars if car.idle}
        new_idle_cars = self.previous_busy_cars & currently_idle_cars
        self.previous_busy_cars = {car for car in self.cars if not car.idle}
        return bool(new_idle_cars)

    def schedule_tasks(self, policy):
        while True:
            self.cars = self.static_cars + self.traci.get_subscribed_vehicles_list()
            print_color(f"\n================== [Log] time: {self.env.now} ==================","93")
            self.print_schedule("Current State")

            # If we don't do the filetering the scheduler will keep assigning the tasks
            # Filter out the tasks whose deadline has expired
            for task in self.get_generated_tasks():
                if self.env.now >= (task.time_of_arrival + task.deadline):
                    print(f"The deadline of Task {task.id} is in the past; Removing it!")
                    # task.status = 3
                    Statistics.save_task_stats(task, "NA")
                    task.source_car.generated_tasks.remove(task) # Remove the task from the system

            if self.generated_tasks_exist() and self.idle_cars_exist():
                for _ in self.get_idle_cars():
                    # If all tasks have been exhausted, break (e.g., a car that had tasks left the scenario in the meanwhile)
                    if not self.generated_tasks_exist():
                        break

                    # By default filter out the tasks with status 5, however if a new car has joined consider status 5 tasks as well
                    tasks = [t for t in self.get_generated_tasks() if t.status != 5]
                    if self.has_new_cars() or self.car_was_busy_now_is_idle_2():
                        print("True")
                        tasks = self.get_generated_tasks()

                    selected_task = policy(tasks)
                    # assert selected_task != None, "The policy returned None!"
                    if selected_task == None:
                        break

                    selected_car = self.select_car(selected_task) #np.random.choice(self.get_idle_cars())
                    # assert selected_car != None, "Car selection returned None!"
                    if selected_car == None:
                        selected_task.status = 5
                        break

                    print(f"Car {selected_car.id} <-- Task {selected_task.id}")

                    # Housekeeping
                    selected_car.assigned_tasks.append(selected_task) # Assign task to the selected car
                    selected_task.source_car.generated_tasks.remove(selected_task) # Remove task from the list of generated tasks
                    selected_task.status = 1
                    selected_car.idle = False

                    # Spawn processes for processing the tasks
                    process = self.env.process(selected_car.process_task(selected_task))
                    selected_car.active_processes.append(process)

            # Print state after assignments are finished
            print_color("----------------------------------------------------","95")
            self.print_schedule("After Scheduling")
            offset = 0.00001 * np.random.random() * np.random.choice([-1, 1])
            yield self.env.timeout(0.5 + offset)  # Check for tasks every unit of time

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

    def select_car(self, task):
        selected_car = None
        best_completion_time = float('inf')

        # # First try to compute locally if the source car is idle
        # if task.source_car in self.get_idle_cars():
        #     selected_car = task.source_car
        #     best_completion_time = self.calculate_completion_time(task.source_car, task)

        # NOTE: The iteration goes through all cars, not only idle cars. But schedule_task() only executes if there are idle cars
        # for car in self.cars:
        for car in self.get_idle_cars():
            completion_time = self.calculate_completion_time(car, task)

            if ((self.env.now + completion_time) <= (task.time_of_arrival + task.deadline)) and (completion_time < best_completion_time):
                selected_car = car
                best_completion_time = completion_time
                print(f"  -> Best car updated to Car {car.id} with Completion Time {completion_time}")
            else:
                print(f"  -> Car {car.id} not suitable for Task {task.id}, because it can either not meet the deadline or doesn't provide a better completion time.")

        return selected_car

    # Static car methods
    def register_static_car(self, cars_list, remove_after_dwell_time=False):
        for car in cars_list:
            self.static_cars.append(car)

            if(remove_after_dwell_time):
                self.env.process(self.remove_after_dwell_time(car))

    def unregister_static_car(self, car):
        if car in self.static_cars:
            self.static_cars.remove(car)
            print(f"Car {car.id} removed from the system at time {self.env.now}")

    def remove_static_car_after_dwell_time(self, car):
        yield self.env.timeout(car.dwell_time)
        self.unregister_static_car(car)
        car.finish()