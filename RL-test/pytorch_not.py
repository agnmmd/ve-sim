import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

# Parameters for the DQN
GAMMA = 0.99  # Discount factor
LEARNING_RATE = 0.001  # Learning rate
EPSILON = 1.0  # Initial exploration rate
EPSILON_DECAY = 0.995  # Decay rate of epsilon
MIN_EPSILON = 0.01  # Minimum epsilon for exploration
BATCH_SIZE = 32  # Batch size for experience replay
TARGET_UPDATE = 10  # Target network update frequency
MEMORY_SIZE = 10000  # Experience replay buffer size

# Environment parameters
NUM_DEVICES = 3  # Number of mobile devices
MAX_QUEUE_LENGTH = 5  # Max tasks in queue

# DQN Model Definition
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# DQN Agent
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = EPSILON
        self.memory = ReplayBuffer(MEMORY_SIZE)

        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.MSELoss()

        # Sync target network
        self.update_target_network()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def select_action(self, state, mask):
        if random.random() < self.epsilon:
            # Exploration
            print("Exploring...")
            # Random action with masking
            valid_actions = np.flatnonzero(mask)
            print("Valid Actions: ", valid_actions)
            return random.choice(valid_actions)
        else:
            # Exploitation
            print("Exploiting...")
            with torch.no_grad():
                q_values = self.policy_net(torch.tensor(state, dtype=torch.float32))
                print(q_values)
                q_values = q_values * torch.tensor(mask)  # Apply mask
                print(q_values)
                return torch.argmax(q_values).item()

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return

        # Sample from experience replay
        batch = self.memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Compute Q values for current states
        q_values = self.policy_net(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q values
        next_q_values = self.target_net(next_states).max(1)[0]
        target_q_values = rewards + GAMMA * next_q_values * (1 - dones)

        # Compute loss
        loss = self.loss_fn(q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(MIN_EPSILON, self.epsilon * EPSILON_DECAY)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

# Example Environment Class
class OffloadingEnv:
    def __init__(self):
        self.num_devices = NUM_DEVICES
        self.reset()

    def reset(self):
        # Randomly initialize CPU capacities, task queues, and task info
        self.devices = [{'cpu': random.random(), 'queue': random.randint(0, MAX_QUEUE_LENGTH), 'busy': False} for _ in range(self.num_devices)]
        self.task = {'complexity': random.random(), 'deadline': random.uniform(0.5, 1.0)}

        print("Devices:", self.devices)
        print("Task", self.task)
        return self.get_state()

    def get_state(self):
        state = []
        for device in self.devices:
            state.append(device['cpu'])
            state.append(device['queue'])
        state.append(self.task['complexity'])
        state.append(self.task['deadline'])
        return state

    def get_action_mask(self):
        mask = []
        for device in self.devices:
            if not device['busy']:
                mask.append(1)
            else:
                mask.append(0)
        return mask

    def step(self, action):
        if action == 0:
            # Process locally
            reward = self.process_task(self.devices[0])
        else:
            # Offload to another device
            target_device = self.devices[action]
            reward = self.process_task(target_device)
        
        next_state = self.reset()
        done = False
        return next_state, reward, done

    def process_task(self, device):
        if device['queue'] < MAX_QUEUE_LENGTH:
            device['queue'] += 1
            device['busy'] = random.random() < 0.5  # Random chance to become busy
            return 1  # Successful task processing
        else:
            return -1  # Failed due to full queue

# Main Training Loop
def train_dqn():
    env = OffloadingEnv()
    state_dim = len(env.get_state())
    action_dim = env.num_devices  # One action per device
    agent = DQNAgent(state_dim, action_dim)

    # num_episodes = 1000
    num_episodes = 1
    for episode in range(num_episodes):
        state = env.reset()
        mask = env.get_action_mask()

        done = False
        episode_reward = 0  # Track total reward for the episode
        step = 0  # Track the number of steps

        print("")
        print("State:", state)
        print("Mask:", mask)

    # while not done:
        step += 1
        action = agent.select_action(state, mask)
        print("Action:", action)
        next_state, reward, done = env.step(action)
        next_mask = env.get_action_mask()
        agent.store_transition(state, action, reward, next_state, done)
        agent.train()

        state = next_state
        mask = next_mask
        episode_reward += reward

    if episode % TARGET_UPDATE == 0:
        agent.update_target_network()
    
    # Print the episode progress
    print(f"Episode {episode}/{num_episodes}, Total Reward: {episode_reward}, Steps: {step}, Epsilon: {agent.epsilon:.3f}")

    if episode % 100 == 0:  # Print every 100 episodes
        print(f"Checkpoint at Episode {episode}: Epsilon: {agent.epsilon}, Total Reward: {episode_reward}")


if __name__ == "__main__":
    train_dqn()
