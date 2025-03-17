import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from collections import deque
import utils as utl

# class Task2:
#     def __init__(self, complexity, deadline):
#         self.complexity = complexity
#         self.deadline = deadline

# class Car2:
#     def __init__(self, processing_power):
#         self.processing_power = processing_power

class TaskSchedulingEnv(gym.Env):
    def __init__(self, sim):
        super(TaskSchedulingEnv, self).__init__()
        
        # FIXME: fix the hard-coded parameters here
        self.max_tasks = 20
        self.max_resources = 5
        self.tasks = []
        self.resources = []
        self.current_task = None
        self.done = False
        self.best_resource = None
        self.current_time = None
        self.sim = sim
        self.duration = self.sim.get_im_parameter('duration')

        # Define Gymnasium action and observation space
        self.action_space = spaces.Discrete(self.max_resources)

        # Observation: [task_complexity, task_deadline] + resource_cpu_capacity_padded
        low_obs = np.array([0.0, 0.0] + [0.0] * self.max_resources, dtype=np.float32)
        high_obs = np.array([np.finfo(np.float32).max] * (2 + self.max_resources), dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.tasks = []
        self.resources = []
        self.done = False
        
        return self._get_state(), {}  # Gymnasium expects (obs, info)

    def _get_state(self):
        task = self.current_task
        task_vector = [task.complexity, task.deadline] if task else [0, 0]
        resources_vector = [res.processing_power for res in self.resources]
        resources_vector += [0] * (self.max_resources - len(resources_vector))
        
        state = task_vector + resources_vector
        return np.array(state, dtype=np.float32)

    def set_values(self, tasks, idle_cars, time):
        self.tasks = tasks
        self.current_task = tasks[0]
        self.resources = idle_cars
        self.current_time = time

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode has ended. Please call reset().")

        if action >= len(self.resources):
            reward = -2.0
        else:
            resource = self.resources[action]
            self.resources.remove(resource)
            completion_time = utl.calculate_completion_time(self.current_time, resource, self.current_task)
            reward = 1.0 if utl.before_deadline(self.current_time, self.current_task, completion_time) else -1.0

        # Update state related parameters
        info = self.current_task
        self.tasks.remove(self.current_task)
        self.current_task = None

        self.done = self.current_time == self.duration

        obs = self._get_state() if not self.done else np.zeros(self.observation_space.shape, dtype=np.float32)

        # Gymnasium expects: obs, reward, terminated, truncated, info = {}
        return obs, reward, self.done, False, info


# DQN Neural Network (unchanged)
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x, valid_mask=None):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_values = self.fc3(x)
        if valid_mask is not None:
            q_values = q_values.masked_fill(valid_mask == 0, float('-inf'))
        return q_values

class ReplayBuffer:
    def __init__(self, capacity, state_size):
        self.buffer = deque(maxlen=capacity)
        self.state_size = state_size

    def push(self, state, action, reward, next_state, done):
        if next_state is None:
            next_state = np.zeros(self.state_size)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.FloatTensor(np.array(states)),
                torch.LongTensor(actions),
                torch.FloatTensor(rewards),
                torch.FloatTensor(np.array(next_states)),
                torch.FloatTensor(dones))

    def __len__(self):
        return len(self.buffer)

class DQNAgentOther:
    def __init__(self, rl_env, sim):
        
        self.rl_env = rl_env
        self.sim = sim

        self.state_size = rl_env.observation_space.shape[0]
        self.action_size = rl_env.action_space.n
        self.max_resources = rl_env.max_resources

        self.lr = self.sim.get_im_parameter('learning_rate')
        self.replay_buffer_capacity = self.sim.get_im_parameter('replay_buffer_capacity')
        self.gamma = self.sim.get_im_parameter('gamma')
        self.batch_size = self.sim.get_im_parameter('batch_size')
        self.target_update_freq = self.sim.get_im_parameter('target_update_freq')
        self.epsilon_max = self.sim.get_im_parameter('epsilon_max')
        self.epsilon = self.epsilon_max
        self.epsilon_min = self.sim.get_im_parameter('epsilon_min')
        self.epsilon_decay = self.sim.get_im_parameter('epsilon_decay_rate')

        self.q_network = DQN(self.state_size, self.action_size)
        self.target_network = DQN(self.state_size, self.action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.replay_buffer = ReplayBuffer(self.replay_buffer_capacity, self.state_size)
        self.train_step_count = 0

    def take_action(self, state, env):
        valid_mask = torch.zeros(self.max_resources)
        valid_mask[:len(env.resources)] = 1

        if random.random() < self.epsilon:
            valid_actions = [i for i in range(len(env.resources))]
            return random.choice(valid_actions)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor, valid_mask)
            return torch.argmax(q_values).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            max_next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_step_count += 1
        if self.train_step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self, episode):
        self.epsilon = max(self.epsilon_min, self.epsilon_max - episode / self.epsilon_decay)

    def load_model(self, path):
        self.current_model.load_state_dict(torch.load(path, weights_only=True))

    def save_model(self, path):
        torch.save(self.current_model.state_dict(), path)