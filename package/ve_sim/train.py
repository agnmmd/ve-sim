from ve_sim.sim import Sim
from ve_sim.scheduler import Scheduler
from ve_sim.policy import *
from ve_sim.rl import *
from ve_sim.main import setup_traci
from ve_sim.stats import Statistics
from ve_sim.sim import Sim
import random
import numpy as np
from ve_sim.input_manager import InputManager

if __name__ == "__main__":

    rl_env = TaskSchedulingEnv()
    agent = DQNAgent(rl_env)
    n_episodes = InputManager.command_line_args.episodes

    for episode in range(n_episodes):
        # Randomness between episodes
        Sim.reset()
        Sim.run = episode
        Sim.repetition = episode
        np.random.seed(episode)
        random.seed(episode)

        policy = DQNPolicy(rl_env, agent, episode=episode)
        traci_mgr = setup_traci()
        scheduler = Scheduler(traci_mgr, policy)

        rl_env.reset()

        # Start Scheduling
        Sim.get_env().process(scheduler.schedule_tasks())
        end = Sim.get_parameter('end')
        Sim.get_env().run(until=end+1)

        # NOTE: Episode ends here

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

    agent.save_model("./training-dqn.pth")