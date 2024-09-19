import traci
import traci.constants as tc
from car import Car

class TraciManager:
    """
    Manages the interaction between SUMO traffic simulation and SimPy environment.
    Handles vehicle subscriptions, updates, and region of interest (ROI) filtering.
    """
    def __init__(self, env, sim, end_time):
        self.env = env
        self.sim = sim
        self.rois = []  # List of regions of interest
        self.subscribed_vehicles = {}  # Dictionary to store subscribed vehicles
        self.end_time = end_time  # New variable to store the simulation end time

    def execute_one_time_step(self):
        previous_vehicle_ids = set()

        while traci.simulation.getMinExpectedNumber() > 0 and self.env.now <= self.end_time:
            try:
                traci.simulationStep()
                yield self.env.timeout(1)
                print(f"Time in SimPy: {self.env.now}, Time in SUMO: {traci.simulation.getTime()}")

                if self.env.now >= 4:
                    self._process_vehicles(previous_vehicle_ids)

            except traci.exceptions.TraCIException as e:
                print(f"TraCI Exception: {e}")
                break

        # Graceful termination
        self._handle_simulation_end()

    def _handle_simulation_end(self):
        print(f"Simulation ending at time: {self.env.now}")
        # Perform any cleanup operations here
        for vehicle in self.subscribed_vehicles.values():
            vehicle.finish()
        self.subscribed_vehicles.clear()
        traci.close()
        print("TraCI disconnected")

    def _process_vehicles(self, previous_vehicle_ids):
        driving_vehicles = set(traci.vehicle.getIDList())

        for vehicle_id in driving_vehicles:
            self._process_single_vehicle(vehicle_id)

        self._handle_left_vehicles(previous_vehicle_ids, driving_vehicles)
        previous_vehicle_ids.update(driving_vehicles)

    def _process_single_vehicle(self, vehicle_id):
        traci.vehicle.subscribe(vehicle_id, (tc.VAR_SPEED, tc.VAR_POSITION))
        subscription_results = traci.vehicle.getSubscriptionResults(vehicle_id)
        speed = subscription_results[tc.VAR_SPEED]
        position = subscription_results[tc.VAR_POSITION]
        print(f"Vehicle: {vehicle_id}: speed={speed}, position={position}")

        if not self.rois or self.in_roi(position, self.rois):
            print(f"Vehicle: {vehicle_id} is in ROI!")
            if vehicle_id not in self.subscribed_vehicles:
                # Create new Car object for newly subscribed vehicles
                self.subscribed_vehicles[vehicle_id] = Car(self.env, self.sim, speed=speed, position=position)
                self.subscribed_vehicles[vehicle_id].generate_tasks_static(10)
                # self.env.process(self.subscribed_vehicles[vehicle_id].generate_tasks())
            else:
                # Update existing Car object
                self.subscribed_vehicles[vehicle_id].update(speed=speed, position=position)
        elif vehicle_id in self.subscribed_vehicles:
            # Remove vehicle from subscriptions if it's no longer in ROI
            del self.subscribed_vehicles[vehicle_id]

    def _handle_left_vehicles(self, previous_vehicle_ids, driving_vehicles):
        vehicles_left = previous_vehicle_ids - driving_vehicles
        if vehicles_left:
            print(f"Vehicles that left the simulation at {self.env.now}: {vehicles_left}")
            for vehicle_id in vehicles_left:
                if vehicle_id in self.subscribed_vehicles:
                    self.subscribed_vehicles[vehicle_id].finish()
                    del self.subscribed_vehicles[vehicle_id]

        print("Vehicles that we care about, subscribed vehicles:", self.subscribed_vehicles.keys())
        print("")

    @staticmethod
    def in_roi(point, boxes):
        x, y = point
        return any(min_x <= x <= max_x and min_y <= y <= max_y for min_x, min_y, max_x, max_y in boxes)

    def set_rois(self, rois):
        self.rois = rois

    def get_subscribed_vehicles_list(self):
        return list(self.subscribed_vehicles.values())