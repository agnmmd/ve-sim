# import torch
# import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class TaskSchedulingEnv:
    def __init__(self, n_tasks=10, n_resources=3):
        self.n_tasks = n_tasks
        self.n_resources = n_resources

        # Tasks have complexity and deadlines
        self.tasks = [{'complexity': np.random.uniform(1.0, 2.0),  # Random complexity between 0.1 and 1.0
                       'deadline': np.random.uniform(0.5, 2)}      # Random deadline between 10 and 50 seconds
                      for _ in range(n_tasks)]
        
        # Resources have fixed processing power (random CPU capacity between 0.5 and 2.0)
        self.resources = [{'cpu_capacity': np.random.uniform(0.5, 2.0)}
                          for _ in range(n_resources)]
        
        self.current_task_idx = 0
        self.done = False
    
    def reset(self):
        self.current_task_idx = 0
        self.done = False
        return self._get_state()

    def _get_state(self):
        """
        Flatten the task and resource information into a numerical vector.
        Example: (task complexity, task deadline) + (resource capacities)
        """
        task = self.tasks[self.current_task_idx]
        
        # Flatten task information
        task_vector = [task['complexity'], task['deadline']]
        
        # Flatten resource information (CPU capacities of all resources)
        resources_vector = [res['cpu_capacity'] for res in self.resources]
        
        # Combine task and resource information into a single state vector
        state = task_vector + resources_vector
        return np.array(state)

    def step(self, action):
        if self.done:
            raise Exception("Episode has already ended. Please reset the environment.")
        
        task = self.tasks[self.current_task_idx]
        resource = self.resources[action]
        
        processing_time = task['complexity'] / resource['cpu_capacity']
        reward = 1.0 if processing_time <= task['deadline'] else -1.0
        
        # Check the next task on the task list
        # If reached all the tasks in the list activate the done flag
        self.current_task_idx += 1
        if self.current_task_idx >= self.n_tasks:
            self.done = True
        
        next_state = self._get_state() if not self.done else None
        return next_state, reward, self.done

# DQN Neural Network (unchanged)
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
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
def epsilon_greedy_action(q_network, state, epsilon, n_actions):
    if random.random() < epsilon:
        return random.randint(0, n_actions - 1)  # Random action
    else:
        state = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
        q_values = q_network(state)
        return torch.argmax(q_values).item()  # Greedy action

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
def train_dqn(n_episodes=1000, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=500, gamma=0.99, lr=0.001, target_update_freq=10, batch_size=64, replay_buffer_capacity=10000):
    
    input_dim = 2 + env.n_resources  # Task complexity + deadline + resource CPU
    output_dim = env.n_resources     # Actions (resource choices)
    state_size = input_dim           # This is the size of the state vector
    
    q_network = DQN(input_dim, output_dim)
    target_network = DQN(input_dim, output_dim)
    target_network.load_state_dict(q_network.state_dict())  # Synchronize networks
    
    optimizer = optim.Adam(q_network.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(replay_buffer_capacity, state_size)  # Pass state_size here

    epsilon = epsilon_start

    for episode in range(n_episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        actions_for_episode = []
        
        while not done:
            # Select action using epsilon-greedy policy
            action = epsilon_greedy_action(q_network, state, epsilon, env.n_resources)
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

if __name__ == "__main__":
    # Initialize the environment
    env = TaskSchedulingEnv(n_tasks=10, n_resources=3)
    
    # Hyperparameters for training
    n_episodes = 500
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

