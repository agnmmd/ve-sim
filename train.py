from sim import Sim
from scheduler import Scheduler
from policy import *
import simpy
from rl_other import *
from main import setup_traci

import random
import numpy as np

if __name__ == "__main__":

    sim = Sim()
    rl_env = TaskSchedulingEnv(sim)
    agent = DQNAgent(rl_env, sim)
    n_episodes = Sim.get_parameter("n_episodes")
    n_episodes = 5

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

        # Load static car
        # car1 = Car(env, sim, speed=-1, position=(-1,-1))
        # # car1.generate_tasks_static()
        # car1.generate_tasks()
        # scheduler.register_static_car([car1])

        # # Scenario 5: Fixed tasks
        # self.tasks = [{'complexity': 2,
        #                'deadline': 1}
        #               for _ in range(np.random.randint(5, self.max_tasks))]  # Random task count
        
        # self.resources = [{'cpu_capacity': np.random.uniform(1.0, 3.0)}
        #                   for _ in range(np.random.randint(2, self.max_resources))]  # Random resource count

        # Start Scheduling
        env.process(scheduler.schedule_tasks_2())
        end = sim.get_im_parameter('end')
        env.run(until=end+1)

        # End of simulation episode start here

        # Print statistics for static cars that haven't been removed by dwell time
        for car in scheduler.static_cars:
            car.finish()

        print(f"Episode {episode}: Total Reward: {policy.get_episode_reward()}")

        # Decay epsilon after each episode
        agent.decay_epsilon(episode)

        # Periodically update the target network
        if episode % agent.target_update_freq == 0:
            agent.target_network.load_state_dict(agent.q_network.state_dict())

    agent.save_model("./rl/training-dqn.pth")