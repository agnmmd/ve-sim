import simpy
import random

def print_color(string, color_code, argument=""):
        # Black (Gray): 90m
        # Red: 91m
        # Green: 92m
        # Yellow: 93m
        # Blue: 94m
        # Magenta: 95m
        # Cyan: 96m
        # White: 97m
        print(f"\033[{color_code}m{string} {argument}\033[0m")

class Sim:
    env = simpy.Environment()

    _random_seed_value = 42
    random.seed(_random_seed_value)

    _task_id_counter = 0
    _car_id_counter = 0

    @classmethod
    def set_task_id(cls):
        cls._task_id_counter += 1
        return cls._task_id_counter
    
    @classmethod
    def set_car_id(cls):
        cls._car_id_counter += 1
        return cls._car_id_counter

class Task:
    def __init__(self, source_car):
        self.id = "t" + str(Sim.set_task_id())
        self.source_car = source_car
        self.time_of_arrival = Sim.env.now
        self.deadline = random.randint(1, 10)
        self.priority = random.randint(0, 3)
        self.complexity = random.randint(1,6)

class Car:
    def __init__(self):
        self.id = "c" + str(Sim.set_car_id())
        self.pending_tasks = []
        self.computing_power = 2
        self.idle = True
        self.dwell_time = 10

        # Statistics
        self.successful_tasks = 0
        self.total_processing_time = 0
        self.processed_tasks_count = 0
        self.time_of_arrival = Sim.env.now

        # Initilization
        Scheduler.register_car(self)
        Sim.env.process(self.remove_after_dwell_time())

    def generate_task(self):
        while True:
            yield Sim.env.timeout(random.expovariate(1.0/5))
            task = Task(self)
            self.pending_tasks.append(task)
            print(f"Car {self.id} generated a Task: {task.__dict__}")

    def generate_tasks_static(self, num_tasks):
        self.pending_tasks = [Task(self) for _ in range(num_tasks)]
        for task in self.pending_tasks:
            print(f"Car {self.id} generated Task {task.id}: {task.__dict__}")

    def process_task(self, assigned_task):
        processing_time = assigned_task.deadline  # Ensure tasks finish before the deadline
        yield Sim.env.timeout(processing_time)

        # Update metrics
        self.total_processing_time += processing_time
        self.processed_tasks_count += 1
        if Sim.env.now - assigned_task.time_of_arrival <= assigned_task.deadline:
            self.successful_tasks += 1

        print(f"@t={Sim.env.now}, Car {self.id} finished computing Task: {assigned_task.id}!")
        self.idle = True
        Scheduler.remove_from_schedule(self.id)

    def remove_after_dwell_time(self):
        yield Sim.env.timeout(self.dwell_time)
        Scheduler.unregister_car(self)

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

    def pending_tasks_exist(self):
        return any(car.pending_tasks for car in self.cars)

    def get_pending_tasks(self):
        return [task for car in self.cars for task in car.pending_tasks]
    
    def get_idle_cars(self):
        return [car for car in self.cars if car.idle]
    
    def idle_cars_exist(self):
        return any(car.idle for car in self.cars)

    def schedule_tasks(self, policy):
        while True:
            print_color(f"\n================== [Log] time: {Sim.env.now} ==================","93")
            self.print_schedule("Current State")
            if self.pending_tasks_exist() and self.idle_cars_exist():
                for idle_car in self.get_idle_cars():
                    # Check if idle tasks exist. If not, break
                    if not self.pending_tasks_exist(): break

                    selected_task = policy(self.get_pending_tasks())
                    print(f"Car {idle_car.id} <-- Task {selected_task.id}")

                    # Housekeeping
                    idle_car.idle = False
                    selected_task.source_car.pending_tasks.remove(selected_task)

                    # Add to schedule
                    self.schedule[idle_car.id] = selected_task

                    # Processing the task
                    Sim.env.process(idle_car.process_task(selected_task))

            # Print state after assignments are finished
            print_color("----------------------------------------------------","95")
            self.print_schedule("After Scheduling")
            # print_color(f"\n================== [End] time: {Sim.env.now} ==================","93")
            yield Sim.env.timeout(1)  # Check for tasks every unit of time

    def print_schedule(self, string):
        idle_cars_ids = [car.id for car in self.get_idle_cars()]
        idle_tasks_ids = [task.id for task in self.get_pending_tasks()]

        print(f"\n[{string}:]")
        print("Idle Cars:", idle_cars_ids)
        print("Idle Tasks:", idle_tasks_ids)

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

class Policy:
    @staticmethod
    def random_policy(tasks):
        if tasks:
            return random.choice(tasks)
        else:
            return None

    @staticmethod
    def earliest_deadline(tasks):
        if tasks:
            # Select the task with the earliest deadline
            return min(tasks, key=lambda task: task.deadline)
        else:
            return None

    @staticmethod
    def highest_priority(tasks):
        if tasks:
            # Select the task with the highest priority
            return min(tasks, key=lambda task: task.priority)
        else:
            return None

def generate_cars(lambda_rate):
    while True:
        yield Sim.env.timeout(1)
        # yield Sim.env.timeout(random.expovariate(lambda_rate))
        new_car = Car()
        print(f"New Car {new_car.id} added at time {Sim.env.now}")
        new_car.generate_tasks_static(2)

def main():
    # env = simpy.Environment()
    scheduler = Scheduler()

    # NOTE: Static car insertion
    car1 = Car()
    car2 = Car()
    car3 = Car()
    # env.process(car1.generate_task())
    # env.process(car2.generate_task())
    car1.generate_tasks_static(1)
    car2.generate_tasks_static(1)

    # NOTE: Dynamic car insertion
    # Sim.env.process(generate_cars(lambda_rate=0.5))

    Sim.env.process(scheduler.schedule_tasks(Policy.earliest_deadline))
    
    Sim.env.run(until=20)  # Run the simulation for 20 time units

    # Print metrics
    for car in Scheduler.cars:
        print(f"Car {car.id} - Processed Tasks: {car.processed_tasks_count}; "
              f"Successful Tasks: {car.successful_tasks}; "
              f"Total Processing Time: {car.total_processing_time}; "
              f"Lifetime: {Sim.env.now - car.time_of_arrival}")

if __name__ == "__main__":
    main()

# Waiting time, migration time, dwell time

# TODO: Distribution for inserting and removing cars, and a distribution for generating tasks
# TODO: Link with sumo
# TODO: Modify the recording of the statistics. Perhaps introduce Stats class? Create structured statistics.
# TODO: Maybe add a finish() method to store all the statistics in the end in a structured way

# TODO: Optional: In the Scheduler add a list (self.task_queue) that holds all the tasks; Also, the tasks can be subscribed automatically to it
# TODO: Optional: Use the schedule as a log for the schedule mapping (car, time) to task.

# NOTE: Due to the order of the iteration there is an order of tasks. Car 0 might be favored.
# NOTE: A task should be at one entity, there should not be multiple copies of a task; Do not use the task queue for assigning tasks. Because a task is both present in Scheduler's queue and in a Car's queue. This is not singularity. Multiple copies of the same entity is not good...