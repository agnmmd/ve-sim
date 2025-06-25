import traci
import traci.constants as tc
from car import Car
from sim import Sim
class TraciManager:
    """
    Manages the interaction between SUMO traffic simulation and SimPy environment.
    Handles vehicle subscriptions, updates, and region of interest (ROI) filtering.
    """
    def __init__(self):
        self.rois = []  # List of regions of interest
        self.subscribed_vehicles = {}  # Dictionary to store subscribed vehicles
        self.end_time = Sim.get_parameter("end")  # New variable to store the simulation end time
        self.start_time = Sim.get_parameter("start")
        self.traci_step_length = Sim.get_parameter('step_length')

    def execute_one_time_step(self):
        while traci.simulation.getMinExpectedNumber() > 0 and Sim.get_env().now <= self.end_time:
            try:
                traci.simulationStep()
                yield Sim.get_env().timeout(float(self.traci_step_length))
                traci_time = traci.simulation.getTime()
                print(f"Time in SimPy: {Sim.get_env().now}, Time in SUMO: {traci_time}")

                # Round the times to 6 decimals and make sure that they are in sync
                assert round(Sim.get_env().now, 6) == round(traci_time, 6), "SimPy time and TraCI time are not the same!"

                if Sim.get_env().now >= self.start_time:
                    self.update_subscriptions()
                    self.update_vehicle_data()

            except traci.exceptions.TraCIException as e:
                print(f"TraCI Exception: {e}")
                # https://github.com/eclipse-sumo/sumo/issues/13730
                # break

        # Graceful termination
        self._handle_simulation_end()

    def _handle_simulation_end(self):
        print(f"Simulation ending at time: {Sim.get_env().now}")
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
        if self.rois:
            self.update_subscriptions_with_roi()
        else:
            self.update_subscriptions_without_roi()

    def update_subscriptions_with_roi(self):
        for vehicle_id in set(traci.vehicle.getIDList()):
            if vehicle_id not in self.subscribed_vehicles:
                position = traci.vehicle.getPosition(vehicle_id)
                if self._is_in_roi(position):
                    print(f"Vehicle: {vehicle_id} is in ROI!")
                    self.subscribe_to_vehicle(vehicle_id)

        # Remove vehicles
        for vehicle_id in list(self.subscribed_vehicles.keys()):
            position = traci.vehicle.getPosition(vehicle_id)
            if not self._is_in_roi(position):
                self._handle_left_vehicles(vehicle_id)

    def update_subscriptions_without_roi(self):
        for vehicle_id in set(traci.vehicle.getIDList()):
            if vehicle_id not in self.subscribed_vehicles:
                self.subscribe_to_vehicle(vehicle_id)

        # Remove vehicles
        vehicles_to_remove = set(self.subscribed_vehicles.keys()) - set(traci.vehicle.getIDList())
        for vehicle_id in vehicles_to_remove:
            self._handle_left_vehicles(vehicle_id)

    def subscribe_to_vehicle(self, vehicle_id):
        traci.vehicle.subscribe(vehicle_id, [tc.VAR_POSITION, tc.VAR_SPEED])
        car = Car(speed=None, position=None)
        self.subscribed_vehicles[vehicle_id] = car
        # car.generate_tasks_static()
        process = Sim.get_env().process(car.generate_tasks())
        car.active_processes.append(process)

    def update_vehicle_data(self):
        for vehicle_id in self.subscribed_vehicles:
            results = traci.vehicle.getSubscriptionResults(vehicle_id)
            if results:
                position = results[tc.VAR_POSITION]
                speed = results[tc.VAR_SPEED]
                self.subscribed_vehicles[vehicle_id].update(speed=speed, position=position)
                # print(f"Vehicle: {vehicle_id}, Position: {position}, Speed: {speed}")

    def _handle_left_vehicles(self, vehicle_id):
        if vehicle_id in set(traci.vehicle.getIDList()):
            traci.vehicle.unsubscribe(vehicle_id)
        self.subscribed_vehicles[vehicle_id].finish()
        del self.subscribed_vehicles[vehicle_id]