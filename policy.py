import random
import numpy as np

from abc import ABC, abstractmethod

class Policy(ABC):
    def __init__(self, env):
        self.env = env

    @abstractmethod
    def match_task_and_car(self, tasks, cars):
        """Select a task and a car based on the policy."""
        pass

    def select_car(self, task, cars):
        selected_car = None
        best_completion_time = float('inf')

        # # First try to compute locally if the source car is idle
        # if task.source_car in self.get_idle_cars():
        #     selected_car = task.source_car
        #     best_completion_time = self.calculate_completion_time(task.source_car, task)

        # NOTE: The iteration goes through all cars, not only idle cars. But schedule_task() only executes if there are idle cars
        for car in cars:
            completion_time = self.calculate_completion_time(self.env.now, car, task)

            if self.before_deadline(self.env.now, task, completion_time) and (completion_time < best_completion_time):
                selected_car = car
                best_completion_time = completion_time
                print(f"  -> Best car updated to Car {car.id} with Completion Time {completion_time}")
            else:
                print(f"  -> Car {car.id} not suitable for Task {task.id}, because it can either not meet the deadline or doesn't provide a better completion time.")

        return selected_car

    @staticmethod
    def calculate_completion_time(current_time, car, task):
        waiting_time = car.get_remaining_time() + car.calculate_waiting_time()
        processing_time = car.calculate_processing_time(task)
        completion_time = waiting_time + processing_time

        print(f"Evaluating Car {car.id} for Task {task.id}:")
        print(f"  Current Time: {current_time}")
        print(f"  Waiting Time: {waiting_time}")
        print(f"  Processing Time: {processing_time}")
        print(f"  Relative Completion Time: {completion_time}")
        print(f"  Task Time of Arrival: {task.time_of_arrival}")
        print(f"  Task Deadline: {task.deadline}")
        print(f"  Estimated Task Completion Time: {current_time + completion_time}")

        return completion_time

    @staticmethod
    def before_deadline(current_time, task, completion_time):
        return (current_time + completion_time) <= (task.time_of_arrival + task.deadline)

class RandomPolicy(Policy):
    def match_task_and_car(self, tasks, cars):
        if tasks and cars:
            task = np.random.choice(tasks)
            car = np.random.choice(cars)
            return task, car
        return None, None

class EarliestDeadline(Policy):
    def match_task_and_car(self, tasks, cars):
        if tasks and cars:
            task = min(tasks, key=lambda t: t.time_of_arrival + t.deadline)
            car = self.select_car(task, cars)
            return task, car
        return None, None

class LowestComplexity(Policy):
    def match_task_and_car(self, tasks, cars):
        if tasks and cars:
            task = min(tasks, key=lambda t: t.complexity)
            car = self.select_car(task, cars)
            return task, car
        return None, None

class DQLTraining(Policy):
    def __init__(self, simenv, gymenv, agent):
        super().__init__(simenv)
        self.agent = agent
        self.gymenv = gymenv

    def match_task_and_car(self, tasks, cars):
        if tasks and cars:
            selected_task = tasks[0]
            self.gymenv.set_values(selected_task, cars, self.env.now, tasks)
            # rl_env.set_values(selected_task, self.get_idle_cars(), self.env.now, self.get_reordered_tasks(noped_tasks))

            state_flattened = np.array(self.gymenv.flatten_state())
            action, q_value = self.agent.take_action(state_flattened, self.gymenv.make_mask(self.gymenv.state))

            if action == self.gymenv.action_space.n-1:
                selected_car = None
            else:
                selected_car = self.gymenv.idle_cars[action]

            next_state, reward, _, _, _ = self.gymenv.step(action)
            next_state_flattened = self.gymenv.flatten_state()
            self.agent.memory.push(state_flattened, action, reward, next_state_flattened, self.gymenv.make_mask(next_state))
            loss = self.agent.replay()

            return selected_task, selected_car
        return None, None

class DQLPolicy(Policy):
    def __init__(self, simenv, gymenv, agent):
        super().__init__(simenv)
        self.agent = agent
        self.gymenv = gymenv

    def match_task_and_car(self, tasks, cars):
        if tasks and cars:
            selected_task = tasks[0]
            self.gymenv.set_values(selected_task, cars, self.env.now, tasks)
            # rl_env.set_values(selected_task, self.get_idle_cars(), self.env.now, self.get_reordered_tasks(noped_tasks))

            state_flattened = np.array(self.gymenv.flatten_state())
            action, q_value = self.agent.take_action(state_flattened, self.gymenv.make_mask(self.gymenv.state))

            if action == self.gymenv.action_space.n-1:
                selected_car = None
            else:
                selected_car = self.gymenv.idle_cars[action]

            self.gymenv.step(action)

            return selected_task, selected_car
        return None, None
    
class DQNPolicy(Policy):
    def __init__(self, simenv, gymenv, agent, episode=-1):
        super().__init__(simenv)
        self.agent = agent
        self.gymenv = gymenv
        
        self.episode = episode
        self.episode_reward = 0
        self.episode_actions = []
        self.episode_best_selection = []

    def match_task_and_car(self, tasks, cars):
        if tasks and cars:
            stat_resource_count = len(cars) # This is before the decision was made and the resource is removed from the state

            self.gymenv.set_values(tasks, cars, self.env.now)
            print("State:", self.gymenv._get_state())

            # Execute car selection action
            action = self.agent.take_action(self.gymenv._get_state(), self.gymenv)

            # If the agent returned noop set it to 'None', else return the selected car
            if action == self.gymenv.action_space.n-1:
                selected_car = None
            else:
                selected_car = self.gymenv.resources[action]
            
            # Take action and observe reward and next state
            next_state, reward, terminated, _, selected_task = self.gymenv.step(action)
            print("State after processing the action:", self.gymenv._get_state())
            
            # Stats
            print("===============================================")
            print("Reward:", reward)
            self.episode_reward += reward
            self.episode_actions.append(action)
            best_selected = action == self.gymenv.stat_best_resource_index
            print("Episode actions: \t", self.episode_actions)
            print("Best resource index: \t", self.gymenv.stat_best_resource_index)
            print("Best selected: \t", best_selected)
            self.episode_best_selection.append(best_selected)

            user_input = input("Enter something (or press Enter to continue): ")
            print("===============================================")

            from stats import Statistics
            Statistics.save_action_stats(self.env.now, self.episode, action, reward, best_selected, stat_resource_count)
            # Statistics.save_training_stats(self.agent.episode, reward, q_value, loss, self.agent.epsilon, self.agent.sim.is_training, self.agent.sim.is_fixed)


            # Store the transition in replay buffer
            # FIXME: set the 'done' flag here. When do we consider this to be done? When time expires?
            done = False # done = terminated or truncated
            self.agent.replay_buffer.push(self.gymenv._get_state(), action, reward, next_state, done)
            
            # Train the Q-network if we have enough samples in the buffer
            self.agent.update()

            return selected_task, selected_car
        return None, None
    
    def get_episode_reward(self):
        return self.episode_reward