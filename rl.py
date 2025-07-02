import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sim import Sim
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import torch.nn.functional as F

from policy import Policy

seed = Sim.get_parameter("seed")
local_random_generator = random.Random(seed)

class TaskSchedulingEnv(gym.Env):
    def __init__(self):
        super(TaskSchedulingEnv, self).__init__()

        self.max_resources = Sim.get_parameter('max_cars')
        self.tasks = []
        self.resources = []
        self.current_task = None
        self.done = False
        self.best_resource = None
        self.current_time = None
        self.duration = Sim.get_parameter('duration')

        # Statistics
        self.stat_best_resource_index = -1

        # Define Gymnasium action and observation space
        self.action_space = spaces.Discrete(self.max_resources)

        # Observation: [resource_count, task_complexity, task_deadline] + resource_cpu_capacity_padded
        low_obs = np.array([0.0, 0.0, 0.0] + [0.0] * self.max_resources, dtype=np.float32)
        high_obs = np.array([self.max_resources, np.finfo(np.float32).max, np.finfo(np.float32).max] + [np.finfo(np.float32).max] * self.max_resources, dtype=np.float32)
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
        # Normalize task features
        task_complexity_norm = task.complexity / 2 if task else 0
        task_deadline_norm = task.deadline / 1 if task else 0

        # Min-max scale processing power to [0, 1]
        processing_powers_norm = [(res.processing_power - 1) / (3 - 1) for res in self.resources]
        # Pad to max_resources
        processing_powers_norm += [0] * (self.max_resources - len(processing_powers_norm))

        # Normalize num available cars
        num_available_norm = len(self.resources) / self.max_resources

        return np.array(
            [num_available_norm, task_complexity_norm, task_deadline_norm] +
            processing_powers_norm,
            dtype=np.float32
        )

    def set_values(self, tasks, idle_cars, time):
        self.tasks = tasks
        self.current_task = tasks[0]
        self.resources = idle_cars
        self.current_time = time

        # Stat
        self.stat_best_resource_index = self.get_best_resource_index()  # NOTE: This information can also be obtained at match_task_and_car() from 'cars' list

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode has ended. Please call reset().")

        if action >= len(self.resources):
            reward = -10.0
        else:
            resource = self.resources[action]
            completion_time = Policy.calculate_completion_time(self.current_time, resource, self.current_task)

            resource = self.resources[action]
            completion_time = Policy.calculate_completion_time(self.current_time, resource, self.current_task)
            best_resource_index = self.get_best_resource_index()

            if action == best_resource_index:
                if Policy.is_before_deadline(self.current_time, self.current_task, completion_time):
                    reward = 2.0  # Selected best resource and met deadline
                else:
                    reward = 0.0  # Selected best resource but couldn't meet deadline (no penalty)
            else:
                reward = -10.0

        # NOTE: Here the selected task is returned as info
        info = self.current_task

        # Housekeeping: Update states and statistics
        self.tasks.remove(self.current_task)
        self.current_task = None
        self.resources.remove(resource)

        self.done = self.current_time == self.duration
        obs = self._get_state() if not self.done else np.zeros(self.observation_space.shape, dtype=np.float32)

        # Gymnasium expects: obs, reward, terminated, truncated, info = {}
        return obs, reward, self.done, False, info

    def get_best_resource_index(self):
        return max(range(len(self.resources)), key=lambda i: self.resources[i].processing_power)

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x, valid_mask=None):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)

        # Apply action mask: Invalid actions get -inf Q-values
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
        batch = local_random_generator.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.FloatTensor(np.array(states)),
                torch.LongTensor(actions),
                torch.FloatTensor(rewards),
                torch.FloatTensor(np.array(next_states)),
                torch.FloatTensor(dones))

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, rl_env):
        
        self.rl_env = rl_env

        self.state_size = rl_env.observation_space.shape[0]
        self.action_size = rl_env.action_space.n
        self.max_resources = rl_env.max_resources

        self.lr = Sim.get_parameter('learning_rate')
        self.replay_buffer_capacity = Sim.get_parameter('replay_buffer_capacity')
        self.gamma = Sim.get_parameter('gamma')
        self.batch_size = Sim.get_parameter('batch_size')
        self.target_update_freq = Sim.get_parameter('target_update_freq')
        self.epsilon_max = Sim.get_parameter('epsilon_max')
        self.epsilon = self.epsilon_max
        self.epsilon_min = Sim.get_parameter('epsilon_min')
        # self.epsilon_decay = Sim.get_parameter('epsilon_decay')
        self.epsilon_decay_rate = Sim.get_parameter('epsilon_decay_rate')

        self.q_network = DQN(self.state_size, self.action_size)
        self.target_network = DQN(self.state_size, self.action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.replay_buffer = ReplayBuffer(self.replay_buffer_capacity, self.state_size)
        self.train_step_count = 0

        # self.explore = Sim.get_parameter('explore')

    def take_action(self, state, env):
        valid_mask = torch.zeros(self.max_resources)
        valid_mask[:len(env.resources)] = 1  # Mark available resources as valid

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor, valid_mask)

        if local_random_generator.random() < self.epsilon:  # Exploration
            valid_actions = torch.where(valid_mask == 1)[0].tolist()
            return local_random_generator.choice(valid_actions) if valid_actions else 0  # Prevent errors
        else:  # Exploitation
            return torch.argmax(q_values).item()

    def update(self):
        if len(self.replay_buffer) < self.batch_size: #OK
            return

        # This is a batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Compute predicted Q-values for current states
        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # Get Q-values for chosen actions

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            max_next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)

        # Compute loss
        loss = nn.MSELoss()(q_values, target_q_values)

        # Update the network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self, episode):
        self.epsilon = max(self.epsilon_min, self.epsilon_max - episode / self.epsilon_decay)

    def decay_epsilon_exp(self, episode):
        self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * np.exp(-self.epsilon_decay_rate * episode)

    def load_model(self, path):
        self.current_model.load_state_dict(torch.load(path, weights_only=True))

    def save_model(self, path):
        torch.save(self.q_network.state_dict(), path)