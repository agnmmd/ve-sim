from sim import Sim
from utils import print_color
import random

class Scheduler:
    cars = list()
    schedule = dict()

    @classmethod
    def remove_from_schedule(cls, car_id):
        del cls.schedule[car_id]

    @classmethod
    def register_car(cls, car):
        cls.cars.append(car)

    @classmethod
    def unregister_car(cls, car):
        if car in cls.cars:
            cls.cars.remove(car)
            print(f"Car {car.id} removed from the system at time {Sim.env.now}")

    def generated_tasks_exist(self):
        return any(car.generated_tasks for car in self.cars)

    def get_generated_tasks(self):
        return [task for car in self.cars for task in car.generated_tasks]
    
    def assigned_tasks_exist(self):
        return any(task for car in self.cars for task in car.assigned_tasks)
    
    def get_assigned_tasks(self):
        return [task for car in self.cars for task in car.assigned_tasks]
    
    def get_idle_cars(self):
        return [car for car in self.cars if car.idle]
    
    def idle_cars_exist(self):
        return any(car.idle for car in self.cars)

    def schedule_tasks(self, policy):
        while True:
            print_color(f"\n================== [Log] time: {Sim.env.now} ==================","93")
            self.print_schedule("Current State")
            if self.generated_tasks_exist() and self.idle_cars_exist():
                for idle_car in self.get_idle_cars():
                    # Check if idle tasks exist. If not, break
                    if not self.generated_tasks_exist(): break

                    selected_task = policy(self.get_generated_tasks())
                    print(f"Car {idle_car.id} <-- Task {selected_task.id}")

                    # Housekeeping
                    idle_car.assigned_tasks.append(selected_task)
                    selected_task.source_car.generated_tasks.remove(selected_task)
                    idle_car.idle = False

                    # Add to schedule
                    self.schedule[idle_car.id] = selected_task

                    # Processing the task
                    Sim.env.process(idle_car.process_task(selected_task))

            # Print state after assignments are finished
            print_color("----------------------------------------------------","95")
            self.print_schedule("After Scheduling")
            # print_color(f"\n================== [End] time: {Sim.env.now} ==================","93")
            offset = 0.0001 * random.random()
            yield Sim.env.timeout(1 + offset)  # Check for tasks every unit of time

    def schedule_tasks_exhaust(self, policy):
        while True:
            print_color(f"\n================== [Log] time: {Sim.env.now} ==================","93")
            self.print_schedule("Current State")
            if self.generated_tasks_exist():
                for _ in range(len(self.get_generated_tasks())):

                    selected_task = policy(self.get_generated_tasks())
                    # selected_car = self.cars[0]
                    selected_car = self.select_car(selected_task)

                    if selected_car:
                        print(f"Car {selected_car.id} <-- Task {selected_task.id}")

                        # Housekeeping
                        selected_car.assigned_tasks.append(selected_task) # Assign task to the selected car
                        selected_task.source_car.generated_tasks.remove(selected_task) # Remove task from the list of generated tasks
                        selected_car.idle = False

                        # # Add to schedule
                        # self.schedule[idle_car.id] = selected_task

                        # Spawn processes for processing the tasks
                        Sim.env.process(selected_car.process_task(selected_task))
                    else:
                        print(f"Task {selected_task.id} couldn't be assigned; No resources to process it before deadline!")
                        if Sim.env.now >= (selected_task.time_of_arrival + selected_task.deadline):
                            print(f"The deadline of Task {selected_task.id} is in the past; Removing it!")
                            selected_task.source_car.generated_tasks.remove(selected_task) # Remove task from the list of generated tasks

            # Print state after assignments are finished
            print_color("----------------------------------------------------","95")
            self.print_schedule("After Scheduling")
            offset = 0.00001 * random.random()
            yield Sim.env.timeout(1 + offset)  # Check for tasks every unit of time

    def print_schedule(self, string):
        idle_cars_ids = [car.id for car in self.get_idle_cars()]
        idle_tasks_ids = [task.id for task in self.get_generated_tasks()]
        assigned_tasks_ids = [task.id for task in self.get_assigned_tasks()]

        print(f"\n[{string}:]")
        print("Idle Cars:\t", idle_cars_ids)
        print("Idle Tasks:\t", idle_tasks_ids)
        print("Assigned Tasks:\t", assigned_tasks_ids)

        if not self.__class__.schedule:
            print("\n[The schedule is empty!]\n")
        else:
            print("\n---------------------------")
            print("car_id | task_id | deadline")
            print("---------------------------")
            for car_id, task in self.__class__.schedule.items():
                if task:
                    print(f"{car_id:6} | {task.id:7} | {'X' * int(task.deadline)}")
                else:
                    print(f"{car_id:6} | {task.id:7} | Task Not Found")
            print("---------------------------\n")

    def select_car(self, task):
        selected_car = None
        best_completion_time = float('inf')

        for car in self.cars:
            waiting_time = car.get_remaining_time() + car.calculate_waiting_time()
            processing_time = car.calculate_processing_time(task)
            completion_time = waiting_time + processing_time

            print(f"Evaluating Car {car.id} for Task {task.id}:")
            print(f"  Current Time: {Sim.env.now}")
            print(f"  Waiting Time: {waiting_time}")
            print(f"  Processing Time: {processing_time}")
            print(f"  Completion Time: {completion_time}")
            print(f"  Task Deadline: {task.deadline}")

            if (Sim.env.now + completion_time) <= (task.time_of_arrival + task.deadline) and completion_time < best_completion_time:
                selected_car = car
                best_completion_time = completion_time
                print(f"  -> Best car updated to Car {car.id} with Completion Time {completion_time}")
        else:
                print(f"  -> Car {car.id} not suitable")

        return selected_car