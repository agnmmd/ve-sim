from sim import Sim
from car import Car
from task import Task
from scheduler import Scheduler
from traci_manager import TraciManager
from policy import Policy
import traci
import os
import simpy

def generate_cars(env, sim, scheduler):
    while True:
        new_car = Car(env, sim, scheduler)
        print(f"New Car {new_car.id} added at time {env.now}")
        new_car.generate_tasks_static(2)
        yield env.timeout(1)
        # yield Sim.env.timeout(random.expovariate(lambda_rate))

# def generate_cars_by_traces(traces, scheduler, region_of_interest):
#     xmin, ymin, xmax, ymax = region_of_interest
#     for car_id in traces[Sim.env.now].items():
#         if car_id not in [car.id for car in scheduler.cars]:
#             new_car = Car()
#             print(f"New Car {new_car.id} added at time {Sim.env.now}")
#             new_car.generate_tasks_static(2)

#     yield Sim.env.timeout(1)
#     # TODO: only register car in scheduler, if is ROI and unregister if not in ROI

def just_a_timer(env):
    """Just a timer that progresses time until the simulation ends"""
    while True:
        print("Timer: ", env.now)
        yield env.timeout(1)

def run_sim(policy_func, run=-1, repetition=-1):
    env = simpy.Environment()
    sim = Sim()
    Sim.update_sim_variables(run=run, repetition=repetition, policy_function=policy_func)
    scheduler = Scheduler(env)
    traci_mgr = TraciManager(env, sim)
    # traci_mgr.set_rois([(-50, -10, 50, 10)])
    sumo_binary = "/usr/bin/sumo"
    sumo_cfg = os.path.join(os.path.dirname(__file__), 'SUMO', 'street.sumocfg')
    sumo_cmd = [sumo_binary, "-c", sumo_cfg] #, "--quit-on-end"]#, "--start"])

    ##################################################
    # NOTE: Static car insertion
    car1 = Car(env, sim, scheduler)
    car2 = Car(env, sim, scheduler)
    car3 = Car(env, sim, scheduler)
    # env.process(car1.generate_task())
    # env.process(car2.generate_task())
    car1.generate_tasks_static(2)
    car2.generate_tasks_static(1)
    env.process(scheduler.schedule_tasks(Policy.random))
    ##################################################

    ##################################################
    # # NOTE: Dynamic car insertion
    # env.process(generate_cars(env, sim, scheduler))
    ##################################################

    ##################################################
    # car = Car(env, sim, scheduler)
    # t1 = Task(env, sim, car)
    # t2 = Task(env, sim, car)
    # car.generated_tasks = [t1, t2]
    
    # t1.complexity = 4
    # t1.deadline = 10

    # t2.complexity = 4
    # t2.deadline = 10

    # env.process(scheduler.schedule_tasks_exhaust(Policy.random))
    ##################################################
    
    ##################################################
    # env.process(just_a_timer(env))
    
    # traci_mgr = TraciManager(env)
    # sumo_binary = "/usr/bin/sumo-gui"
    # sumo_cfg = os.path.join(os.path.dirname(__file__), 'SUMO', 'street.sumocfg')
    # sumo_cmd = [sumo_binary, "-c", sumo_cfg] #, "--quit-on-end"]#, "--start"])
    # traci.start(sumo_cmd)
    
    # env.process(traci_mgr.execute_one_time_step())
    ##################################################
    
    env.run(until=30)  

    # Print metrics
    for car in scheduler.cars:
        print(f"Car {car.id} - Processed Tasks: {car.processed_tasks_count}; "
              f"Successful Tasks: {car.successful_tasks}; "
              f"Total Processing Time: {car.total_processing_time}; "
              f"Lifetime: {env.now - car.time_of_arrival}")    
    # Print statistics for static cars that haven't been removed by dwell time
    for car in scheduler.static_cars:
        car.finish()
        
if __name__ == "__main__":
    for x in Policy.get_policies():
        print(x)
            run_sim(policy_func, run, repetition)

    # Executing single scenario
    # run_sim(policy_func=Policy.p_random)


# TODO: Optional: In the Scheduler add a list (self.task_queue) that holds all the tasks; Also, the tasks can be subscribed automatically to it
# TODO: Optional: Use the schedule as a log for the schedule mapping (car, time) to task.

# NOTE: Due to the order of the iteration there is an order of tasks. Car 0 might be favored.
# NOTE: A task should be at one entity, there should not be multiple copies of a task; 
# NOTE: Initiation of the tasks currently is happening in the Traci; Alternative: move to __init__ of Car?
# NOTE: A record of the the car that processes a task is not stored in the Task object for Statistic purposes. Maybe integrate.
# NOTE: Currently the Stats module exists separately, maybe create the Stats module inside Sim in the constructor...
# NOTE: Sim configuration is updated with e staticmethod. Maybe generate a Sim() object with all necessary config parameters in the constuctor.
