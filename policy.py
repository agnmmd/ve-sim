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