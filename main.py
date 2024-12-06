from sim import Sim
from car import Car
from task import Task
from scheduler import Scheduler
from traci_manager import TraciManager
from policy import Policy
import traci
import os
import simpy
from traci_annotation import TraciAnnotation

# def generate_cars_by_traces(traces, scheduler, region_of_interest):
#     xmin, ymin, xmax, ymax = region_of_interest
#     for car_id in traces[Sim.env.now].items():
#         if car_id not in [car.id for car in scheduler.cars]:
#             new_car = Car()
#             print(f"New Car {new_car.id} added at time {Sim.env.now}")
#             new_car.generate_tasks_static(2)

#     yield Sim.env.timeout(1)

# def generate_cars(env, sim, scheduler):
#     while True:
#         new_car = Car(env, sim)
#         scheduler.register_static_car([new_car])
#         print(f"New Car {new_car.id} added at time {env.now}")
#         new_car.generate_tasks_static(2)
#         yield env.timeout(1)
#         # yield Sim.env.timeout(random.expovariate(lambda_rate))

def just_a_timer(env):
    """Just a timer that progresses time until the simulation ends"""
    while True:
        print("Timer: ", env.now)
        yield env.timeout(1)

def run_sim():

    env = simpy.Environment()
    sim = Sim()

    start = sim.get_im_parameter('start')
    end = sim.get_im_parameter('end')
    traci_mgr = TraciManager(env, sim, start, end) # New variable to store the simulation end time)
    # traci_mgr = TraciManager(env, sim, 18)
    # traci_mgr.set_rois([(-50, -10, 50, 10)])
    sumo_binary = "/usr/bin/sumo-gui"
    sumo_cfg = os.path.join(os.path.dirname(__file__), "SUMO", "street.sumocfg")
    sumo_cmd = [sumo_binary, "-c", sumo_cfg, "--quit-on-end", "--step-length", "0.01"]#, "--start"])
    # --start # Start the simulation immediately after loading (no need to press the start button)
    # --quit-on-end # Quit the simulation gui in the end automatically once the simulation is finished
    # --step-length TIME # Defines the step duration in seconds
    scheduler = Scheduler(env, traci_mgr)

    traci.start(sumo_cmd)

    ##################################################
    # NOTE: Scenario 1: Static car insertion
    # car1 = Car(env, sim)
    # car2 = Car(env, sim)
    # car3 = Car(env, sim)
    # # env.process(car1.generate_tasks())
    # # env.process(car2.generate_tasks())
    # car1.generate_tasks_static(2)
    # car2.generate_tasks_static(1)

    # scheduler.register_static_car([car1,car2])
    ##################################################

    ##################################################
    # NOTE: Scenario 2: Dynamic car insertion
    # env.process(generate_cars(env, sim, scheduler))
    ##################################################

    ##################################################
    # NOTE: Scenario 3: Static cars and tasks
    # car = Car(env, sim)
    # scheduler.register_static_car([car])
    # t1 = Task(env, sim, car)
    # t2 = Task(env, sim, car)
    # car.generated_tasks = [t1, t2]
    
    # t1.complexity = 4
    # t1.deadline = 10

    # t2.complexity = 4
    # t2.deadline = 10
    ##################################################
    
    ##################################################
    # NOTE: Scenario 4: Static cars and TraCI cars
    env.process(just_a_timer(env))

    car1 = Car(env, sim)
    scheduler.register_static_car([car1])
    

    ##################################################
    # drawer = TraciAnnotation()

    # # Add a rectangle using bottom-left and top-right coordinates
    # bottom_left = (100, 100)
    # top_right = (200, 200)
    # drawer.add_rectangle("rectangle1", bottom_left, top_right)

    # # Add a circle
    # drawer.add_circle("circle1", center=(150, 150), radius=30)

    # # Draw all shapes in the SUMO simulation
    # drawer.draw_shapes()

    # env.process(traci_mgr.execute_one_time_step())
    ##################################################
    
    # Start Scheduling
    env.process(scheduler.schedule_tasks_exhaust(policy_func))

    env.run(until=30)

    # Print statistics for static cars that haven't been removed by dwell time
    for car in scheduler.static_cars:
        car.finish()

if __name__ == "__main__":
    # # Executing multiple simulation for each different policy
    # run = 0
    # repeat = 1
    # for repetition in range(repeat):
    #     # Inner loop
    #     for policy_func in Policy.get_policies():
    #         print(run, repetition, policy_func)
    #         run_sim(policy_func, run, repetition)
    #         run += 1

    # Executing single scenario
    # run_sim(policy_func=Policy.p_random)

    run_sim()


# TODO: Optional: In the Scheduler add a list (self.task_queue) that holds all the tasks; Also, the tasks can be subscribed automatically to it
# TODO: Optional: Use the schedule as a log for the schedule mapping (car, time) to task.

# NOTE: Due to the order of the iteration there is an order of tasks. Car 0 might be favored.
# NOTE: A task should be at one entity, there should not be multiple copies of a task; 
# NOTE: Initiation of the tasks currently is happening in the Traci; Alternative: move to __init__ of Car?
# NOTE: A record of the the car that processes a task is not stored in the Task object for Statistic purposes. Maybe integrate.
# NOTE: Currently the Stats module exists separately, maybe create the Stats module inside Sim in the constructor...
# NOTE: Sim configuration is updated with e staticmethod. Maybe generate a Sim() object with all necessary config parameters in the constuctor.
