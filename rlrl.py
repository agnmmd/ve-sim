import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# from rl_buffer import ExperienceReplayBuffer
from utils import print_color

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from task import Task
from car import Car

class CaseCounter:

    instances = []

    def __init__(self, name):
        self.name = name
        self.count = 0
        CaseCounter.instances.append(self)

    def increment(self):
        self.count += 1

    def get_count(self):
        return self.count
    
    @classmethod
    def print_instances(cls):
        for counter in cls.instances:
            print(f"{counter.name}: {counter.count}")

    @classmethod
    def get_case_counter_by_name(cls, name):
        for counter in cls.instances:
            if counter.name == name:
                return counter
        return None
     
noop_success_counter = CaseCounter("Noop Success")
noop_failure_counter = CaseCounter("Noop Failure")
task_success_counter = CaseCounter("Task Success")
task_acceptable_counter = CaseCounter("Task Acceptable")
task_failure_counter = CaseCounter("Task Failure")

class DQLEnvironment(gym.Env):
    def __init__(self, sim):
        super(DQLEnvironment, self).__init__()
        self.sim = sim
        self.tasks_per_car = sim.get_im_parameter('max_tasks')
        self.max_cars = sim.get_im_parameter('max_cars')
        self.max_tasks = self.max_cars * self.tasks_per_car
        self.time = 0
        self.pending_tasks = [None] * self.max_tasks
        self.idle_cars = [None] * self.max_cars
        self.current_task = None                                 
        self.action_space = spaces.Discrete(self.max_cars + 1) # noop
        #Discrete(4) means [0,3]

        task_dict = spaces.Dict({
        'deadline': spaces.Discrete(sim.get_im_parameter('max_deadline')+1),
        'complexity': spaces.Discrete(sim.get_im_parameter('max_complexity')+1),
        })

        car_dict = spaces.Dict({ 
            'time_of_arrival': spaces.Box(low = 0.0, high = sim.get_im_parameter('end')+1, shape=(), dtype = np.float32),
            'processing_power': spaces.Discrete(sim.get_im_parameter('max_processing_power')+1),
            'speed': spaces.Box(low = 0.0, high = sim.get_im_parameter('max_speed')+1, shape=(), dtype = np.float32), #my guess
            'position': spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        })

        self.observation_space = spaces.Dict({
            'tasks': spaces.Tuple([task_dict for _ in range(self.max_tasks)]),
            'cars': spaces.Tuple([car_dict for _ in range(self.max_cars)]),
            'time': spaces.Box(low= 0.0, high= sim.get_im_parameter('end')+1, shape=(), dtype=np.float32)  
        })

        self.state = self._get_state()

    def _get_state(self):
        tasks_array = [Task.to_dict(task) for task in self.pending_tasks]
        cars_array = [Car.to_dict(car) for car in self.idle_cars]
        time =  np.array(self.time, dtype=np.float32)
        return {'tasks': tuple(tasks_array), 'cars': tuple(cars_array), 'time': time}
    
    def make_mask(self, state):
        mask = []
        for car in state['cars']:
            if (
                car['time_of_arrival'] == 0 and 
                car['processing_power'] == 0 and 
                car['speed'] == 0 and 
                np.all(car['position'] == 0)
                ):
                mask.append(0)
            else:
                mask.append(1)

        return mask
    
    def reset(self, seed = None, options = None):
        self.pending_tasks = [None] * self.max_tasks
        self.idle_cars = [None] * self.max_cars
        self.time = 0
        self.state = self._get_state()
        return self.state, {} #info dict

    def calculate_reward(self, action): 
        task = self.current_task
        migration_time = 0  #task.size / rl_env.data_rate
        task_sim_deadline = task.time_of_arrival + task.deadline
        # noop decision
        if action == self.action_space.n -1:
            if(
                any(
                    self.time + idle_car.calculate_processing_time(task) + migration_time
                    <= task_sim_deadline
                    for idle_car in self.idle_cars if idle_car is not None 
                )
            ):
                noop_failure_counter.increment()
                return -10
            else:
                noop_success_counter.increment()
                return 0
        # car decision
        else:
            selected_car = self.idle_cars[action]
            processing_time = selected_car.calculate_processing_time(task)
            completion_time = self.time + processing_time + migration_time 
            if(
                completion_time <= task_sim_deadline  
                and
                not any(
                    idle_car.calculate_processing_time(task) < processing_time
                    for idle_car in self.idle_cars if idle_car is not None
                )
            ):
                task_success_counter.increment()
                return 10
            elif (completion_time <= task_sim_deadline):
                task_acceptable_counter.increment()
                return 5
            else:
                task_failure_counter.increment()
                return -5

    def step(self, action):
        index = action
        reward = self.calculate_reward(action)
        # if action is not noop (normal action)
        if action < self.action_space.n - 1:
            self.idle_cars.pop(index)
            self.idle_cars.append(None)
            self.pending_tasks.pop(0)
            self.pending_tasks.append(None)
        else:
            task = self.pending_tasks.pop(0)
            if None in self.pending_tasks:
                index = self.pending_tasks.index(None)
                self.pending_tasks[index] = task 
                self.pending_tasks.append(None)
            else:
                self.pending_tasks.append(task)

        self.state = self._get_state()
        #observation, reward, done, truncated, info
        return self.state, reward, False, False, {}

    def normalize(self, value, min_value, max_value):
        # [-1,1] - for pos. & neg. ranges (position)
        if value == 0:
            result = 0
        else: 
            result = (-1 + ((value - min_value) * (1 - (-1)))/ (max_value - min_value))
        return result

    def flatten_state(self):
        flattened = []
        for task in self.state['tasks']:
            flattened.extend([
                self.normalize(task['deadline'], 1, self.sim.get_im_parameter('max_deadline')),
                self.normalize(task['complexity'], 1, self.sim.get_im_parameter('max_complexity')),
                ])
        for car in self.state['cars']:
            flattened.extend([
                self.normalize(float(car['time_of_arrival']), 0, 86400),
                self.normalize(float(car['processing_power']), 1, self.sim.get_im_parameter('max_processing_power')),
                self.normalize(float(car['speed']), 0, self.sim.get_im_parameter('max_speed'))
                ])
            flattened.extend([
                self.normalize(pos, min_pos, max_pos)
                for pos, min_pos, max_pos in zip(
                    list(car['position']),
                    [self.sim.get_im_parameter('roi_min_x'), self.sim.get_im_parameter('roi_min_y')],
                    [self.sim.get_im_parameter('roi_max_x'), self.sim.get_im_parameter('roi_max_y')]
                )
                ])
        flattened.append(
                self.normalize(float(self.state['time']), 0 , 86400)
                )
        assert len(flattened) == 1 + (2 * len(self.state['tasks'])) + (5 * len(self.state['cars'])) 
        return flattened
    
    def set_values(self, selected_task, idle_cars, time, tasks):
        self.current_task = selected_task
        self.idle_cars = idle_cars + [None] * (len(self.idle_cars) - len(idle_cars))
        self.time = time
        self.pending_tasks = tasks + [None] * (len(self.pending_tasks) - len(tasks))

class DQLAgent:
    def __init__(self, rl_env, sim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rl_env = rl_env
        self.sim = sim
        self.state_space = len(rl_env.flatten_state())
        self.action_space = rl_env.action_space.n
        self.memory = ExperienceReplayBuffer(sim.get_im_parameter('replay_buffer_capacity'))
        self.current_model = self.build_model().to(self.device)
        self.target_model = self.build_model().to(self.device)
        self.copy_current_q_weights()
        self.optimizer = optim.AdamW(self.current_model.parameters(), lr= sim.get_im_parameter('learning_rate'), amsgrad=True)

        self.episode = 1
        self.step_counter = 0

        self.batch_size = sim.get_im_parameter('batch_size')
        self.gamma = sim.get_im_parameter('gamma')
        self.epsilon = 1
        self.epsilon_decay_rate = sim.get_im_parameter('epsilon_decay_rate')
        self.epsilon_min = sim.get_im_parameter('epsilon_min')
        
    def build_model(self):
        return nn.Sequential(
            nn.Linear(self.state_space, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_space)
        )

    def copy_current_q_weights(self):
        self.target_model.load_state_dict(self.current_model.state_dict())

    def get_valid_actions(self, mask):
        return [a for a in range(self.action_space) if a == self.action_space - 1 or mask[a] == 1]

    def get_argmax_valid_q_value(self, env_state, mask):
        state_tensor = torch.FloatTensor(env_state).unsqueeze(0).to(self.device)
        valid_actions = self.get_valid_actions(mask)
        valid_q_values = self.current_model(state_tensor)[0, valid_actions].cpu().detach().numpy()
        action_index = valid_q_values.argmax()
        best_action = torch.tensor([[valid_actions[action_index]]], device = self.device, dtype=torch.long)
        best_q_value = self.current_model(state_tensor)[0, best_action].item()
        return best_action, best_q_value

    def get_argmax_target_q_values(self, next_state_batch, next_state_mask_batch):
        # get all best q values  the  next_state_batch , after using next_state_mask_batch 
        batch_size = next_state_batch.shape[0]
        max_valid_q_values = torch.zeros(batch_size, device=self.device)
        
        with torch.no_grad():
            # q values for each state from batch
            all_q_values = self.target_model(next_state_batch)
            
            for i in range(batch_size):
                valid_actions = torch.nonzero(next_state_mask_batch[i], as_tuple=True)[0]
                valid_q_values = all_q_values[i, valid_actions]
                # If no valid actions from mask -> only option is noop
                if valid_actions.numel() == 0:
                    max_valid_q_values[i] = all_q_values[i, -1]
                else:
                    max_valid_q_values[i] = valid_q_values.max()
        
        return max_valid_q_values

    def take_action(self, state, mask):
        if np.random.rand() <= self.epsilon:
            action = torch.tensor([[random.choice(self.get_valid_actions(mask))]], device=self.device, dtype=torch.long)
            action_q_value = self.current_model(torch.FloatTensor(state).unsqueeze(0).to(self.device))[0, action.item()].item()
            print_color(f"EXPLORING - Action: {action}",96)
            return action, action_q_value
        else:
            with torch.no_grad():   #saves memory and speeds up computation - no gradient calculation
                action, max_q_value = self.get_argmax_valid_q_value(state, mask)
                print_color(f"EXPLOITING - Action: {action}",95)
                return action, max_q_value
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return -1
        
        batch = self.memory.sample(self.batch_size)
        state_batch = torch.tensor(np.array([experience.state for experience in batch]), dtype= torch.float32, device = self.device)
        action_batch = torch.cat([experience.action.to(self.device) for experience in batch])
        reward_batch = torch.cat([torch.tensor([experience.reward], dtype=torch.float32, device=self.device) for experience in batch])
        next_state_batch = torch.tensor(np.array([experience.next_state for experience in batch]), dtype=torch.float32, device=self.device)
        next_state_mask_batch = torch.tensor(np.array([experience.next_state_mask for experience in batch]), dtype=torch.bool, device=self.device)

        current_q_values = self.current_model(state_batch).gather(1, action_batch)

        target_q_values = torch.zeros(self.batch_size, device= self.device)
        with torch.no_grad():
            target_q_values = self.get_argmax_target_q_values(next_state_batch, next_state_mask_batch)

        target = (reward_batch + (self.gamma * target_q_values)).unsqueeze(-1)

        loss = nn.MSELoss()(current_q_values, target)
        #td_error = target_q_values - current_q_values
        #clipped_td_error = torch.clamp(td_error, -1, 1)
        #loss = (clipped_td_error ** 2).mean()

        # stochastic gradient descent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_counter += 1
        if self.step_counter == 100:
            self.copy_current_q_weights()
            self.step_counter = 0

        return loss.item()

    def update_epsilon(self):
        self.epsilon = self.epsilon_min + (1-self.epsilon_min) * (math.e ** (- self.epsilon_decay_rate * self.episode))

    def load_model(self, path):
        self.current_model.load_state_dict(torch.load(path, weights_only=True))

    def save_model(self, path):
        torch.save(self.current_model.state_dict(), path)

import random
from collections import namedtuple, deque
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'next_state_mask'])

class ExperienceReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, state, action, reward, next_state, next_state_mask):
        self.memory.append(Experience(state, action, reward, next_state, next_state_mask))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)