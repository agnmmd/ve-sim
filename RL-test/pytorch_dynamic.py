# import torch
# import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class TaskSchedulingEnv:
    def __init__(self, max_tasks=20, max_resources=5):
        self.max_tasks = max_tasks
        self.max_resources = max_resources
        self.tasks = []
        self.resources = []
        self.current_task_idx = 0
        self.done = False
        self.best_resource = None
        self.best_resource_prev_episode = None
        self.reset()
    
    def reset(self):
        """Resets the environment with a new dynamic set of tasks and resources."""
        
        self.best_resource_prev_episode = self.best_resource

        self.tasks = [{'complexity': np.random.uniform(1.0, 2.0),
                       'deadline': np.random.uniform(0.5, 2.0)}
                      for _ in range(np.random.randint(5, self.max_tasks))]  # Random task count
        
        # Resources have fixed processing power (random CPU capacity between 0.5 and 2.0)
        self.resources = [{'cpu_capacity': np.random.uniform(0.5, 2.0)}
                          for _ in range(np.random.randint(2, self.max_resources))]  # Random resource count
        
        self.best_resource = max(range(len(self.resources)), key=lambda i: self.resources[i]['cpu_capacity'])
        
        self.current_task_idx = 0
        self.done = False
        return self._get_state()

    def _get_state(self):
        """
        Convert variable-length tasks and resources into a fixed-size representation.
        """
        task = self.tasks[self.current_task_idx] if self.current_task_idx < len(self.tasks) else None
        task_vector = [task['complexity'], task['deadline']] if task else [0, 0]

        resources_vector = [res['cpu_capacity'] for res in self.resources]
        
        # Pad the resources vector to a fixed size (max_resources)
        resources_vector += [0] * (self.max_resources - len(resources_vector))

        state = task_vector + resources_vector
        return np.array(state, dtype=np.float32)

    def step(self, action):
        if self.done:
            raise Exception("Episode has ended. Reset the environment.")
        
        task = self.tasks[self.current_task_idx]
        
        # Ensure action is within available resources
        if action >= len(self.resources):
            reward = -2.0  # Penalty for invalid action
        else:
            resource = self.resources[action]
            processing_time = task['complexity'] / resource['cpu_capacity']
            reward = 1.0 if processing_time <= task['deadline'] else -1.0

        self.current_task_idx += 1
        if self.current_task_idx >= len(self.tasks):
            self.done = True

        next_state = self._get_state() if not self.done else None
        return next_state, reward, self.done
    
    def is_best_resource_changed(self):
        if self.best_resource_prev_episode is None:
            return False  # No previous episode to compare with
        return self.best_resource != self.best_resource_prev_episode

# DQN Neural Network (unchanged)
class DQN(nn.Module):
    def __init__(self, max_resources):
        super(DQN, self).__init__()
        input_dim = 2 + max_resources  # Task features + padded resource features
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, max_resources)  # Output Q-values for all possible actions
    
    def forward(self, x, valid_mask=None):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_values = self.fc3(x)

        # Apply masking: Set invalid resource actions to a low value
        if valid_mask is not None:
            q_values = q_values.masked_fill(valid_mask == 0, float('-inf'))
        
        return q_values
    
class ReplayBuffer:
    def __init__(self, capacity, state_size):
        self.buffer = deque(maxlen=capacity)
        self.state_size = state_size
    
    def push(self, state, action, reward, next_state, done):
        if next_state is None:  # When the episode ends
            next_state = np.zeros(self.state_size)  # Replace None with zero-filled state
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
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

# Epsilon-greedy action selection (unchanged)
def epsilon_greedy_action(q_network, state, epsilon, env):
    valid_mask = torch.zeros(env.max_resources)
    valid_mask[:len(env.resources)] = 1  # Only first `n` resources are valid
    
    if random.random() < epsilon:
        valid_actions = [i for i in range(len(env.resources))]
        return random.choice(valid_actions)  # Choose only from valid actions
    else:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
        q_values = q_network(state_tensor, valid_mask)
        return torch.argmax(q_values).item()  # Pick best valid action

# Q-learning update without replay buffer
def update_q_network(q_network, target_network, optimizer, state, action, reward, next_state, done, gamma):
    # Convert to PyTorch tensors
    state = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
    next_state = torch.FloatTensor(next_state).unsqueeze(0) if next_state is not None else None
    
    # Compute predicted Q-value for the current state-action pair
    q_values = q_network(state)
    q_value = q_values[0, action]  # Select Q-value for the taken action
    
    # Compute target Q-value
    with torch.no_grad():
        if done:
            target_q_value = torch.FloatTensor([reward])  # Convert reward to a tensor
        else:
            next_q_values = target_network(next_state)
            max_next_q_value = torch.max(next_q_values)
            target_q_value = torch.FloatTensor([reward + gamma * max_next_q_value.item()])  # Convert to a tensor

    # Compute loss (MSE)
    loss = nn.MSELoss()(q_value, target_q_value)
    
    # Update the network
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def update_q_network_batch(q_network, target_network, optimizer, batch, gamma):
    states, actions, rewards, next_states, dones = batch
    
    # Compute predicted Q-values for current states
    q_values = q_network(states)
    q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # Get Q-values for chosen actions
    
    # Compute target Q-values
    with torch.no_grad():
        next_q_values = target_network(next_states)
        max_next_q_values = next_q_values.max(1)[0]
        target_q_values = rewards + gamma * max_next_q_values * (1 - dones)

    # Compute loss
    loss = nn.MSELoss()(q_values, target_q_values)
    
    # Update the network
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Training loop without replay buffer
def train_dqn(n_episodes=500, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=500, gamma=0.99, lr=0.001, target_update_freq=10, batch_size=64, replay_buffer_capacity=10000):
    
    max_resources = 5  # Define the maximum resources the network can handle
    q_network = DQN(max_resources)
    target_network = DQN(max_resources)
    target_network.load_state_dict(q_network.state_dict())
    
    optimizer = optim.Adam(q_network.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(replay_buffer_capacity, state_size=2 + max_resources)

    epsilon = epsilon_start

    for episode in range(n_episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        actions_for_episode = []
        
        while not done:
            # Select action using epsilon-greedy policy
            action = epsilon_greedy_action(q_network, state, epsilon, env)
            actions_for_episode.append(action)
            
            # Take action and observe reward and next state
            next_state, reward, done = env.step(action)
            episode_reward += reward
            
            # Store the transition in replay buffer
            replay_buffer.push(state, action, reward, next_state if not done else None, done)
            state = next_state

            # Train the Q-network if we have enough samples in the buffer
            if len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                update_q_network_batch(q_network, target_network, optimizer, batch, gamma)
        
        # Decay epsilon after each episode
        epsilon = max(epsilon_end, epsilon_start - episode / epsilon_decay)
        
        # Periodically update the target network
        if episode % target_update_freq == 0:
            target_network.load_state_dict(q_network.state_dict())
        
        print(f"Episode {episode}: Total Reward: {episode_reward}")
        print(env.tasks)
        print(env.resources)
        print(actions_for_episode)
        count = actions_for_episode.count(env.best_resource)
        print("Number_of_Tasks", len(actions_for_episode))
        print("Count_Best", count)
        print("Count_Other", len(actions_for_episode) - count)
        print("Ratio_Best", count/len(actions_for_episode))

        if env.is_best_resource_changed():
            print("Best_resource_changed", "1")
        else:
            print("Best_resource_changed", "0")

if __name__ == "__main__":
    # Initialize the environment
    env = TaskSchedulingEnv()
    
    # Hyperparameters for training
    n_episodes = 2000
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 500
    gamma = 0.99
    lr = 0.001
    target_update_freq = 10
    
    # Start the DQN training
    train_dqn(n_episodes=n_episodes,
              epsilon_start=epsilon_start,
              epsilon_end=epsilon_end,
              epsilon_decay=epsilon_decay,
              gamma=gamma,
              lr=lr,
              target_update_freq=target_update_freq)
    
# # Example usage

# # Initiate the environment, which has the static tasks, static resources, and other flags necessary for the program
# env = TaskSchedulingEnv()

# # Reset the state of the environment for new episode
# state = env.reset()

# # We interact with the environment 10 times because there are 10 tasks
# for _ in range(N_TASKS):
#     action = random.choice([0, 1, 2])  # Randomly select a resource (0, 1, or 2)
#     next_state, reward, done = env.step(action)
#     print(f"Action: {action}, Reward: {reward}, Done: {done}")
#     if done:
#         break

