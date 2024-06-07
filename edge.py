from policy import Policy
from statistics import Statistics
import random
import simpy
import simpy.util
import traci
import traci.constants as tc
import os

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
        self.deadline = 2 #random.randint(1, 10)
        self.priority = random.randint(0, 3)
        self.complexity = 2 #random.randint(1,6)
        self.processing_start = None
        self.processing_end = None

class Car:
    def __init__(self):
        self.id = "c" + str(Sim.set_car_id())
        self.generated_tasks = []
        self.processing_power = 2
        self.idle = True
        self.dwell_time = 10
        self.assigned_tasks = []
        self.processor = simpy.Resource(Sim.env, capacity=1)
        self.current_task = None

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
            self.generated_tasks.append(task)
            print(f"Car {self.id} generated a Task: {task.__dict__}")

    def generate_tasks_static(self, num_tasks):
        self.generated_tasks = [Task(self) for _ in range(num_tasks)]
        for task in self.generated_tasks:
            print(f"Car {self.id} generated Task {task.id}: {task.__dict__}")

    def process_task(self, selected_task):
        with self.processor.request() as req:
            yield req

            # Housekeeping
            assert(selected_task == self.assigned_tasks[0])
            self.current_task = self.assigned_tasks.pop(0)
            self.current_task.processing_start = Sim.env.now
            
            processing_time = self.calculate_processing_time(selected_task)
            # Start processing
            yield Sim.env.timeout(processing_time)
            # Finished processing

            # Update metrics
            self.total_processing_time += processing_time
            self.processed_tasks_count += 1
            if Sim.env.now - self.current_task.time_of_arrival <= self.current_task.deadline:
                self.successful_tasks += 1

            print(f"@t={Sim.env.now}, Car {self.id} finished computing Task: {selected_task.id}!")
            self.current_task.processing_end = Sim.env.now
            Statistics.save_task_stats(self.current_task)
            self.current_task = None
        self.idle = True
        # Scheduler.remove_from_schedule(self.id)

    def remove_after_dwell_time(self):
        yield Sim.env.timeout(self.dwell_time)
        Scheduler.unregister_car(self)

    def calculate_waiting_time(self):
        return sum(task.complexity / self.processing_power for task in self.assigned_tasks)

    def calculate_processing_time(self, task):
        return task.complexity / self.processing_power
    
    def get_remaining_time(self):
        if self.current_task is None:
            return 0
        remaining_time = (self.current_task.complexity / self.processing_power) - (Sim.env.now - self.current_task.processing_start)
        return remaining_time

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

def generate_cars(lambda_rate):
    while True:
        new_car = Car()
        print(f"New Car {new_car.id} added at time {Sim.env.now}")
        new_car.generate_tasks_static(2)
        yield Sim.env.timeout(1)
        # yield Sim.env.timeout(random.expovariate(lambda_rate))

def generate_cars_by_traces(traces, scheduler, region_of_interest):
    xmin, ymin, xmax, ymax = region_of_interest
    for car_id in traces[Sim.env.now].items():
        if car_id not in [car.id for car in scheduler.cars]:
            new_car = Car()
            print(f"New Car {new_car.id} added at time {Sim.env.now}")
            new_car.generate_tasks_static(2)

    yield Sim.env.timeout(1)
    # TODO: only register car in scheduler, if is ROI and unregister if not in ROI

class SumoManager:
    sumoBinary = ""
    sumo_cfg = ""
    traces = {}

    def set_sumo_binary(cls, path):
        cls.sumoBinary = path
    
    def set_sumo_config_path(cls, path):
        cls.sumo_cfg = path

    def run_sumo_simulation(cls, until):

        sumoCmd = [cls.sumoBinary, "-c", cls.sumo_cfg]
        traci.start(sumoCmd)
        for step in range(until):
            traci.simulationStep()
            
            for v_id in traci.vehicle.getIDList():
                v_pos = traci.vehicle.getPosition(v_id)
                if step not in cls.traces:
                    cls.traces[step] = {}
                cls.traces[step][v_id] = v_pos

        traci.close()
    
    def print_traces(cls):
        for step in cls.traces:
            print_color(f"\n============ Step {step} ============","93")
            for v_id, pos in cls.traces[step].items():
                print(f"  Vehicle {v_id}: x: {pos[0]}, y: {pos[1]}")

class TraciManager:
    def __init__(self):
        sumoBinary = "/usr/bin/sumo-gui"
        sumo_cfg = os.path.join(os.path.dirname(__file__), 'SUMO', 'street.sumocfg')
        sumoCmd = [sumoBinary, "-c", sumo_cfg, "--quit-on-end"]#, "--start"]
        traci.start(sumoCmd)
    
    def execute_one_time_step(self):
        rois = [[-50, -10, 50, 10]]
        subscribed_vehicles = {}
        SPEED_THRESHOLD = 13.0

        previous_vehicle_ids = set()

        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            yield Sim.env.timeout(1)
            print("Time in SimPy:", Sim.env.now)
            simulation_time = traci.simulation.getTime()
            print("Time in SUMO:", simulation_time)

            if Sim.env.now < 4:
                pass
            else:
                driving_vehicles = set(traci.vehicle.getIDList())
                
                for vehicle_id in driving_vehicles:
                    traci.vehicle.subscribe(vehicle_id, (tc.VAR_SPEED, tc.VAR_POSITION))
                    subscription_results = traci.vehicle.getSubscriptionResults(vehicle_id)
                    speed = subscription_results[tc.VAR_SPEED]
                    position = subscription_results[tc.VAR_POSITION]
                    print(f"vehicle: {vehicle_id}: speed={speed}, position={position}")

                    if self.inROI(position, rois):
                        print(f"Vehicle: {vehicle_id} is in ROI!")
                        subscribed_vehicles[vehicle_id] = subscription_results
                    else:
                        if vehicle_id in subscribed_vehicles.keys():
                            del subscribed_vehicles[vehicle_id]

                # Find vehicles that have left SUMO
                vehicles_left = previous_vehicle_ids - driving_vehicles
                if vehicles_left:
                    print(f"Vehicles that left the simulation at {Sim.env.now}: {vehicles_left}")

                previous_vehicle_ids = driving_vehicles

                for vehicle_id in vehicles_left:
                    if vehicle_id in subscribed_vehicles.keys():
                            del subscribed_vehicles[vehicle_id]

                ##################
                print("Vehicles that we care about, subscribed vehicles:", subscribed_vehicles.keys())
                print("")

    def inROI(self, point, boxes):
        # Unpack the point
        x, y = point
        
        # Iterate through each box
        for box in boxes:
            min_x, min_y, max_x, max_y = box
            # Check if the point is within the bounds of the current box
            if min_x <= x <= max_x and min_y <= y <= max_y:
                return True
        
        # If the point is not in any of the boxes
        return False

def just_a_timer():
    """Just a timer that progresses time until the simulation ends"""
    while True:
        print("Timer: ", Sim.env.now)
        yield Sim.env.timeout(1)

def main():
    # env = simpy.Environment()
    scheduler = Scheduler()

    # # NOTE: Static car insertion
    # car1 = Car()
    # car2 = Car()
    # car3 = Car()
    # # env.process(car1.generate_task())
    # # env.process(car2.generate_task())
    # car1.generate_tasks_static(2)
    # car2.generate_tasks_static(1)

    c_g = Car()
    t1 = Task(c_g)
    t2 = Task(c_g)
    c_g.generated_tasks = [t1, t2]
    
    t1.complexity = 4
    t1.deadline = 10

    t2.complexity = 4
    t2.deadline = 10

    # NOTE: Dynamic car insertion
    # Sim.env.process(generate_cars(lambda_rate=0.5))

    # Sim.env.process(scheduler.schedule_tasks_exhaust(Policy.random))
    Sim.env.process(just_a_timer())
    traciMgr = TraciManager()
    Sim.env.process(traciMgr.execute_one_time_step())
    
    Sim.env.run(until=30)  # Run the simulation for 20 time units

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