import simpy
import random

class Sim:
    global env
    env = simpy.Environment()

    _task_id_counter = 0
    _car_id_counter = 0

    @classmethod
    def set_task_id(cls):
        cls._task_id_counter += 1
        return cls._task_id_counter
    
    @classmethod
    def set_car_id(cls):
        cls._car_id_counter += 1
        return cls._car_id_counter

class Task:
    def __init__(self, source_car):
        self.id = Sim.set_task_id()
        self.source_car = source_car
        self.duration = random.randint(1, 10)

class Car:
    def __init__(self):
        self.id = Sim.set_car_id()
        self.pending_tasks = []
        self.computing_power = 2
        self.idle = True

    def generate_task(self):
        while True:
            yield env.timeout(random.expovariate(1.0/5))
            task = Task(self)
            self.pending_tasks.append(task)
            print(f"Car {self.id} generated a Task: {task.__dict__}")
            # Rescheduling logic, if any, can be implemented outside of this method

    def generate_tasks_static(self):
        num_tasks = 2  # Number of tasks is limited to two
        
        self.pending_tasks = [Task(self) for _ in range(num_tasks)]
        for task in self.pending_tasks:
            print(f"Car {self.id} generated Task {task.id}: {task.__dict__}")

    def process_task(self, assigned_task):
        yield env.timeout(assigned_task.duration)
        print(f"At: t={env.now}, Car {self.id} computed task: {assigned_task.__dict__}")
        self.idle = True

class Scheduler:
    def __init__(self, cars):
        self.cars = cars

    def queued_tasks_exist(self):
        return any(car.pending_tasks for car in self.cars)

    def idle_cars_exist(self):
        return any(car.idle for car in self.cars)

    def get_idle_cars(self):
        return [car for car in self.cars if car.idle]

    def schedule_tasks(self):
        while True:
            print("time:",env.now)
            if self.queued_tasks_exist() and self.idle_cars_exist():
                for idle_car in self.get_idle_cars():
                    random_car = random.choice(self.cars)
                    random_task = random.choice(random_car.pending_tasks)
                    print(f"Task {random_task.id} --> Car {idle_car.id}")

                    # Housekeeping
                    idle_car.idle = False
                    random_car.pending_tasks.remove(random_task)
                    # Processing the task
                    env.process(idle_car.process_task(selected_task))
            yield env.timeout(1)  # Check for tasks every unit of time

def main():
    # env = simpy.Environment()

    car1 = Car()
    car2 = Car()
    cars = [car1, car2]
    scheduler = Scheduler(cars)

    # env.process(car1.generate_task())
    # env.process(car2.generate_task())
    car1.generate_tasks_static()
    car2.generate_tasks_static()
    env.process(scheduler.schedule_tasks())
    
    env.run(until=20)  # Run the simulation for 20 time units

if __name__ == "__main__":
    main()

# A task should be at one entity, there should not be multiple copies of a task
# TODO: In the Scheduler add a list (self.task_queue) that holds all the tasks; Also, the tasks can be subscribed automatically to it
# TODO: Do I need a separate list of the tasks?
# TODO: Due to the order of the iteration there is an order of tasks. Car 0 might be favored.
# Do not use the task queue for assigning tasks. Because a task is both present in Scheduler's queue and in a Car's queue. This is not singularity. Multiple copies of the same entity is not good...
# TODO:
# Print of the queue, cars and the assignment, before and after