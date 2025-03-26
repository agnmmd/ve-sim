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
# from policy_factory import get_policy
from rlrl import *
from rl_other import *
from class_factory import load_class

def sim_clock(env, time_interval):
    """Just a timer that progresses time until the simulation ends"""
    while True:
        print("SimPy time: ", env.now)
        yield env.timeout(time_interval)

def setup_traci(env, sim):
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

    traci.start(sumo_cmd)
    env.process(traci_mgr.execute_one_time_step())

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
    
    return traci_mgr

def run_sim():

    sim = Sim()
    rl_env = load_class(Sim.get_parameter("rl_environment"), sim=sim)
    agent = load_class(Sim.get_parameter("rl_agent"), rl_env=rl_env, sim=sim)

    env = simpy.Environment()
    
    policy_name = Sim.get_parameter("policy")
    if  policy_name == "DQNPolicy":
        policy = load_class(Sim.get_parameter("policy"), simenv=env, gymenv=rl_env, agent=agent)
    else:
        policy = load_class(Sim.get_parameter("policy"), env=env)
    
    traci_mgr = setup_traci(env, sim)
    scheduler = Scheduler(env, traci_mgr, policy)

    # Load static car
    car1 = Car(env, sim, speed=-1, position=(-1,-1))
    # car1.generate_tasks_static()
    car1.generate_tasks()
    scheduler.register_static_car([car1])

    # Start Scheduling
    env.process(scheduler.schedule_tasks_2())
    end = sim.get_im_parameter('end')
    env.run(until=end+1)

    # Print statistics for static cars that haven't been removed by dwell time
    for car in scheduler.static_cars:
        car.finish()

if __name__ == "__main__":
    run_sim()
