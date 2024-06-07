from sim import Sim
from car import Car
from task import Task
from scheduler import Scheduler
from traci_manager import TraciManager
from policy import Policy

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

def just_a_timer():
    """Just a timer that progresses time until the simulation ends"""
    while True:
        print("Timer: ", Sim.env.now)
        yield Sim.env.timeout(1)

def main():
    # env = simpy.Environment()
    scheduler = Scheduler()

    ##################################################
    # # NOTE: Static car insertion
    # car1 = Car()
    # car2 = Car()
    # car3 = Car()
    # # env.process(car1.generate_task())
    # # env.process(car2.generate_task())
    # car1.generate_tasks_static(2)
    # car2.generate_tasks_static(1)
    # Sim.env.process(scheduler.schedule_tasks(Policy.random))
    ##################################################

    ##################################################
    # # NOTE: Dynamic car insertion
    # Sim.env.process(generate_cars(lambda_rate=0.5))
    ##################################################

    ##################################################
    # c_g = Car()
    # t1 = Task(c_g)
    # t2 = Task(c_g)
    # c_g.generated_tasks = [t1, t2]
    
    # t1.complexity = 4
    # t1.deadline = 10

    # t2.complexity = 4
    # t2.deadline = 10

    # Sim.env.process(scheduler.schedule_tasks_exhaust(Policy.random))
    ##################################################
    
    ##################################################
    # Sim.env.process(just_a_timer())
    # traciMgr = TraciManager()
    # Sim.env.process(traciMgr.execute_one_time_step())
    ##################################################
    
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

# TODO: Add a finish() method to store all the statistics in the end in a structured way

# TODO: Optional: In the Scheduler add a list (self.task_queue) that holds all the tasks; Also, the tasks can be subscribed automatically to it
# TODO: Optional: Use the schedule as a log for the schedule mapping (car, time) to task.

# NOTE: Due to the order of the iteration there is an order of tasks. Car 0 might be favored.
# NOTE: A task should be at one entity, there should not be multiple copies of a task; 
# Do not use the task queue for assigning tasks. Because a task is both present in Scheduler's queue and in a Car's queue. This is not singularity. Multiple copies of the same entity is not good...
