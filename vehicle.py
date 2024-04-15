import simpy
import numpy as np

class VehicleExample:
    """
    Vehicle object that can drive() for a given time
    depending on its fuel consumption and fuel capacity.

    Attributes:
        env (simpy.Environment): the simpy environment to run with.
        min_speed (float, optional): min speed expressed in km/hs. Defaults to 80.
        max_speed (float, optional): max speed expressed in km/hs. Defaults to 100.
        fuel_consumption (float, optional): fuel consumption expressed in km/lt. Defaults to 10.
        fuel_capacity (float, optional): fuel capacity expressed in lt. Defaults to 35.
    """

    def __init__(
        self,
        env: simpy.Environment,
        min_speed: float = 80.0,
        max_speed: float = 100,
        fuel_consumption: float = 10,
        fuel_capacity: float = 35,
    ):
        self.env = env
        self.min_speed = min_speed  # km/hs
        self.max_speed = max_speed  # km/hs
        self.fuel_consumption = fuel_consumption  # km/lt
        self.fuel_capacity = fuel_capacity  # lt
        self.fuel = fuel_capacity  # lt

    def drive(self):
        """Excecutes the loop that makes the vehicle drive"""
        while True:
            print(f"Start Driving at {self.env.now:.2f}")
            travel_time = (self.fuel * self.fuel_consumption) / (
                np.random.randint(self.min_speed, self.max_speed)
            )
            yield self.env.timeout(travel_time)
            print(f"Need refueling at {self.env.now:.2f}")

            print(f"Start refueling at {self.env.now:.2f}")
            yield self.env.timeout(np.random.uniform(0.05, 0.15))
            self.fuel = np.random.randint(self.fuel_capacity - 5, self.fuel_capacity)
            print(
                f"Finished refueling at {self.env.now:.2f}: fuel now is {self.fuel:.2f}"
            )

if __name__ == "__main__":

    env = simpy.Environment()
    v = VehicleExample(env)
    env.process(v.drive())
    env.run(until=24)