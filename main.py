from sim import Sim
from car import Car
from task import Task
from scheduler import Scheduler
from traci_manager import TraciManager
from policy import *
import traci
import os
import simpy
from traci_annotation import TraciAnnotation
from policy_factory import get_policy

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

def sim_clock(env, time_interval):
    """Just a timer that progresses time until the simulation ends"""
    while True:
        print("SimPy time: ", env.now)
        yield env.timeout(time_interval)

def run_sim():

    env = simpy.Environment()
    sim = Sim()

    start = sim.get_im_parameter('start')
    end = sim.get_im_parameter('end')
    traci_mgr = TraciManager(env, sim, start, end) # New variable to store the simulation end time)
    # traci_mgr = TraciManager(env, sim, 18)
    # traci_mgr.set_rois([(6430,7180,6562,7257)])
    # # 6430,7180-6562,7257
    sumo_binary = sim.get_im_parameter('sumo_binary')
    sumo_cfg = sim.get_im_parameter('sumo_cfg')
    sumo_step_length = sim.get_im_parameter('sumo_step_length')

    sumo_cmd = [sumo_binary, "-c", sumo_cfg, "--quit-on-end", "--step-length", sumo_step_length]#, "--start"])
    # --start # Start the simulation immediately after loading (no need to press the start button)
    # --quit-on-end # Quit the simulation gui in the end automatically once the simulation is finished
    # --step-length TIME # Defines the step duration in seconds

    # Set up the Scheduler
    policy = get_policy(Sim.get_parameter('policy_name'), env)


    scheduler = Scheduler(env, traci_mgr, policy)

    traci.start(sumo_cmd)
    env.process(traci_mgr.execute_one_time_step())

    if not traci.isLoaded():
        env.process(sim_clock(env, 0.1))

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

    car1 = Car(env, sim, speed=-1, position=(-1,-1))
    # car1.generate_tasks_static()
    car1.generate_tasks()
    scheduler.register_static_car([car1])

    ##################################################
    # drawer = TraciAnnotation()

    # # Add a rectangle using bottom-left and top-right coordinates
    # # 6430,7180-6562,7257
    # bottom_left = (6430,7180)
    # top_right = (6562,7257)
    # drawer.add_rectangle('rectangle1', bottom_left, top_right)

    # # # Add a circle
    # drawer.add_circle('circle1', center=(6430,7180), radius=1500)

    # # # Draw all shapes in the SUMO simulation
    # drawer.draw_shapes()
    ##################################################
    
    # Start Scheduling
    env.process(scheduler.schedule_tasks_2(rl_env))

    env.run(until=end+1)

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
