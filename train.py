from sim import Sim
from scheduler import Scheduler
from policy import *
import simpy
from rl import *
from main import setup_traci
from stats import Statistics

import random
import numpy as np

if __name__ == "__main__":

    sim = Sim()
    rl_env = TaskSchedulingEnv(sim)
    agent = DQNAgent(rl_env, sim)
    n_episodes = Sim.get_parameter("n_episodes")

    for episode in range(n_episodes):
        # Randomness between episodes
        sim.run = episode
        sim.repetition = episode
        np.random.seed(episode)
        random.seed(episode)

        env = simpy.Environment()
        policy = DQNPolicy(env, rl_env, agent, episode=episode)
        traci_mgr = setup_traci(env, sim)
        scheduler = Scheduler(env, traci_mgr, policy)

        rl_env.reset()

        # Start Scheduling
        env.process(scheduler.schedule_tasks())
        end = sim.get_im_parameter('end')
        env.run(until=end+1)

        # End of simulation episode start here

        # Print statistics for static cars that haven't been removed by dwell time
        for car in scheduler.static_cars:
            car.finish()

        print(f"Episode {episode}: Total Reward: {policy.get_episode_reward()}")
        Statistics.save_episode_stats(episode, policy.get_episode_reward(), policy.get_episode_best_selection_ratio(), policy.get_episode_action_count())

        # Decay epsilon after each episode
        agent.decay_epsilon_exp(episode)

        # Periodically update the target network
        if episode % agent.target_update_freq == 0:
            agent.target_network.load_state_dict(agent.q_network.state_dict())

    agent.save_model("./rl/training-dqn.pth")