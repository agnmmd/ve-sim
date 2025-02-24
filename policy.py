import random
import numpy as np

class Policy:
    @staticmethod
    def p_random(tasks):
        if tasks:
            return np.random.choice(tasks)
        else:
            return None

    @staticmethod
    def p_earliest_deadline(tasks):
        if tasks:
            # Select the task with the soonest deadline
            return min(tasks, key=lambda task: task.time_of_arrival + task.deadline)
        else:
            return None
        
    @staticmethod
    def p_lowest_complexity(tasks):
        if tasks:
            # Select the task with the lowest complexity
            return min(tasks, key=lambda task: task.complexity)
        else:
            return None
        
    # @staticmethod
    # def p_shortest_deadline(tasks):
    #     if tasks:
    #         # Select the task with the shortest deadline
    #         return min(tasks, key=lambda task: task.deadline)
    #     else:
    #         return None

    # @staticmethod
    # def p_highest_priority(tasks):
    #     if tasks:
    #         # Select the task with the highest priority
    #         return min(tasks, key=lambda task: task.priority)
    #     else:
    #         return None

    @classmethod
    def get_policies(cls):
        policies = {
            policy_name[2:]: getattr(cls, policy_name) 
            for policy_name in dir(cls) 
            if callable(getattr(cls, policy_name)) and policy_name.startswith("p_")
        }
        return policies