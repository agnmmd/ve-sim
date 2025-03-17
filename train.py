from sim import Sim
from scheduler import Scheduler
from policy import *
import traci
import os
import simpy
from traci_annotation import TraciAnnotation
from policy_factory import get_policy
from rlrl import *
from rl_other import *
from main import setup_traci

def run_sim():
    pass

if __name__ == "__main__":

    sim = Sim()
    rl_env = TaskSchedulingEnv(sim)
    agent = DQNAgentOther(rl_env, sim)
    n_episodes = Sim.get_parameter("n_episodes")

    for episode in range(n_episodes):
        env = simpy.Environment()
        traci_mgr = setup_traci(env, sim)
        policy = DQLTrainingPolicyOther(env, rl_env, agent)
        scheduler = Scheduler(env, traci_mgr, policy)
        # car1 = Car(env, sim, speed=-1, position=(-1,-1))
        # # car1.generate_tasks_static()
        # car1.generate_tasks()
        # scheduler.register_static_car([car1])
        env.process(scheduler.schedule_tasks_2(rl_env))
        end = sim.get_im_parameter('end')
        env.run(until=end+1)

        # Print statistics for static cars that haven't been removed by dwell time
        for car in scheduler.static_cars:
            car.finish()

        agent.decay_epsilon(episode)
        agent.save_model()