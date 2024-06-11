import random

class Policy:
    @staticmethod
    def random(tasks):
        if tasks:
            return random.choice(tasks)
        else:
            return None

    @staticmethod
    def shortest_deadline(tasks):
        if tasks:
            # Select the task with the shortest deadline
            return min(tasks, key=lambda task: task.deadline)
        else:
            return None

    @staticmethod
    def highest_priority(tasks):
        if tasks:
            # Select the task with the highest priority
            return min(tasks, key=lambda task: task.priority)
        else:
            return None
        
    @staticmethod
    def earliest_deadline(tasks):
        if tasks:
            # Select the task with the soonest deadline
            return min(tasks, key=lambda task: task.time_of_arrival + task.deadline)
        else:
            return None
        
    @staticmethod
    def lowest_complexity(tasks):
        if tasks:
            # Select the task with the lowest complexity
            return min(tasks, key=lambda task: task.complexity)
        else:
            return None
        
    @classmethod
    def get_policies(cls):
        # Get all static methods from the Policy class
        policies = [method for method in dir(cls) if callable(getattr(cls, method)) and not method.startswith("__")]
        return [getattr(cls, method) for method in policies]