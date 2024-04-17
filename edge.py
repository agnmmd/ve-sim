import simpy
import random

class Sim:
    # def __init__(self):
    #     print("Initiating:", self.__class__.__name__)
    
    # warm_up = 10
    # sim_duration = 20

    _task_id_counter = 0  # Static variable to keep track of the unique ID
    _car_id_counter = 0

    @classmethod
    def getTaskId(cls):
        cls._task_id_counter += 1
        return cls._task_id_counter
    
    @classmethod
    def getCarId(cls):
        cls._car_id_counter += 1
        return cls._car_id_counter

class Task:
    def __init__(self, source_id, duration):
        self.id = Sim.getTaskId()
        # TODO: replace with an object instead of an id?
        self.source_id = source_id
        self.duration = duration

class Car:
    def __init__(self, env):
        self.env = env
        self.id = Sim.getCarId()
        self.tasks = []
        # TODO: model computing power
        self.computing_power = 2
        self.idle = True

    def generate_task(self):
        while True:
            yield self.env.timeout(random.expovariate(1.0/5))  # Generate tasks at an average rate of 5 per time unit
            # destination = random.choice(["car1", "car2"])
            duration = random.uniform(1, 3)
            task = Task(self.id, duration)
            self.tasks.append(task)
            print(f"Car {self.id} generated a Task: {task.__dict__}")
            # TODO: Do rescheduling here?

    def generate_tasks_static(self):
        # TODO: Number of tasks is limited to two
        # TODO: Currently using fixed duration tasks; Previously: uniform(1, 3)
        self.tasks = [Task(self.id, 10) for _ in range(2)]
        
        for t in self.tasks:
            print(f"Car {self.id} generated Task {t.id}: {t.__dict__}")

    # def compute_tasks(self):
    #     while True:
    #         if self.tasks:
    #             task = self.tasks.pop(0)
    #             yield self.env.timeout(task.duration)
    #             print(f"{self.id} computed task: {task.__dict__}")
    #         else:
    #             yield self.env.timeout(1)  # Check for tasks every unit of time

class Scheduler:
    def __init__(self, env, cars):
        self.env = env
        self.cars = cars

        # TODO: Add a class varible that holds all the tasks; Also, the tasks can be subscribed automatically to it
        # self.task_queue

    def tasks_exist(self):
        task_queue = [task for car in self.cars for task in car.tasks]
        if len(task_queue) == 0:
            return False
        else:
            return True
        
    def idle_cars_exist(self):
        for car in self.cars:
            if car.idle == True:
                return True
        return False
    
    def get_idle_cars(self):
        idle_cars = []
        for car in self.cars:
            if car.idle == True:
                idle_cars.append(car)
        return idle_cars

    def schedule_tasks(self):
        while True:
            # TODO: Do I need a separate list of the tasks?
            # TODO: Due to the order of the iteration there is an order of tasks. Car 0 might be favored.
            # Do not use the task queue for assigning tasks. Because a task is both present in Scheduler's queue and in a Car's queue. This is not singularity. Multiple copies of the same entity is not good...
            
            if self.tasks_exist() == False:
                print("The queue is empty. Nothing to schedule.")
                continue
            else:
                if self.idle_cars_exist() == True:
                    for idle_car in self.get_idle_cars():
                        random_car = random.choice(self.cars)
                        random_task = random.choice(random_car.tasks)
                        print(f"Task {random_task.id} assigned to Car {idle_car.id}")
                        idle_car.idle = False
                    print()
                else:
                    print("There are no idle resources...")



            # for car in self.cars:
            #     if car.tasks:  # If the car has tasks
            #         task = car.tasks[0]
            #         if task.destination == car.name:  # Task is for the car itself
            #             # Compute the task locally
            #             yield self.env.process(car.compute_tasks())
            #         else:
            #             # Assign task to the appropriate car
            #             destination_car = next((c for c in self.cars if c.name == task.destination), None)
            #             if destination_car:
            #                 destination_car.tasks.append(task)
            #                 print(f"Task assigned from {car.name} to {destination_car.name}: {task.__dict__}")
            #                 car.tasks.pop(0)  # Remove task from the current car's task list
            # # TODO: How often do we schedule the tasks. Probably we need some signaling mechanism there.
            yield self.env.timeout(1)  # Check for tasks every unit of time

def main():
    env = simpy.Environment()

    car1 = Car(env)
    car2 = Car(env)
    cars = [car1, car2]
    scheduler = Scheduler(env, cars)

    # env.process(car1.generate_task())
    # env.process(car2.generate_task())
    car1.generate_tasks_static()
    car2.generate_tasks_static()
    env.process(scheduler.schedule_tasks())
    
    env.run(until=20)  # Run the simulation for 20 time units

if __name__ == "__main__":
    main()


# A task should be at one entity, there should not be multiple copies of a task