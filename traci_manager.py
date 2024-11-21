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
        while traci.simulation.getMinExpectedNumber() > 0 and self.env.now <= self.end_time:
            try:
                traci.simulationStep()
                yield self.env.timeout(1)
                print(f"Time in SimPy: {self.env.now}, Time in SUMO: {traci.simulation.getTime()}")

                if self.env.now >= 4:
                    self.update_subscriptions()
                    self.update_vehicle_data()

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

    def _is_in_roi(self, point):
        x, y = point
        return any(min_x <= x <= max_x and min_y <= y <= max_y for min_x, min_y, max_x, max_y in self.rois)

    def set_rois(self, rois):
        self.rois = rois

    def get_subscribed_vehicles_list(self):
        return list(self.subscribed_vehicles.values())

    def update_subscriptions(self):
        traci_vehicles = set(traci.vehicle.getIDList())

        # self.remove_unsubscribed_vehicles(traci_vehicles)

        if self.rois:
            self.update_subscriptions_with_roi(traci_vehicles)
        else:
            self.update_subscriptions_without_roi(traci_vehicles)

    def update_subscriptions_with_roi(self, traci_vehicles):
        for vehicle_id in traci_vehicles:
            if vehicle_id not in self.subscribed_vehicles:
                position = traci.vehicle.getPosition(vehicle_id)
                if self._is_in_roi(position):
                    self.subscribe_to_vehicle(vehicle_id)

        # Remove vehicles
        for vehicle_id in list(self.subscribed_vehicles.keys()):
            position = traci.vehicle.getPosition(vehicle_id)
            if not self._is_in_roi(position):
                self._handle_left_vehicles(vehicle_id, traci_vehicles)

    def update_subscriptions_without_roi(self, traci_vehicles):
        for vehicle_id in traci_vehicles:
            if vehicle_id not in self.subscribed_vehicles:
                self.subscribe_to_vehicle(vehicle_id)

        # Remove vehicles
        vehicles_to_remove = set(self.subscribed_vehicles.keys()) - traci_vehicles
        for vehicle_id in vehicles_to_remove:
            self._handle_left_vehicles(vehicle_id, traci_vehicles)

    def subscribe_to_vehicle(self, vehicle_id):
        traci.vehicle.subscribe(vehicle_id, [tc.VAR_POSITION, tc.VAR_SPEED])
        self.subscribed_vehicles[vehicle_id] = Car(self.env, self.sim, speed=None, position=None)
        self.subscribed_vehicles[vehicle_id].generate_tasks_static()
        # self.env.process(self.subscribed_vehicles[vehicle_id].generate_tasks())

    def remove_unsubscribed_vehicles(self, traci_vehicles):
    #     vehicles_to_remove = set(self.subscribed_vehicles.keys()) - traci_vehicles
    #     print("\tVehicles_to_remove:", vehicles_to_remove)
    #     for vehicle_id in vehicles_to_remove:
    #         self._handle_left_vehicles(vehicle_id, traci_vehicles)

    #     if self.rois:
    #         for vehicle_id in list(self.subscribed_vehicles.keys()):
    #             position = traci.vehicle.getPosition(vehicle_id)
    #             if not self._is_in_roi(position):
    #                 print("\tVehilce left ROI:", vehicle_id)
    #                 self._handle_left_vehicles(vehicle_id, traci_vehicles)
        pass

    def update_vehicle_data(self):
        for vehicle_id in self.subscribed_vehicles:
            results = traci.vehicle.getSubscriptionResults(vehicle_id)
            if results:
                position = results[tc.VAR_POSITION]
                speed = results[tc.VAR_SPEED]
                self.subscribed_vehicles[vehicle_id].update(speed=speed, position=position)
                print(f"Vehicle: {vehicle_id}, Position: {position}, Speed: {speed}")

    def _handle_left_vehicles(self, vehicle_id, traci_vehicles):
        if vehicle_id in traci_vehicles:
            traci.vehicle.unsubscribe(vehicle_id)
        self.subscribed_vehicles[vehicle_id].finish()
        del self.subscribed_vehicles[vehicle_id]