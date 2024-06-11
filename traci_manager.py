from sim import Sim
import traci
import traci.constants as tc

class TraciManager:
    def __init__(self, env):
        self.env = env

    def execute_one_time_step(self):
        rois = [[-50, -10, 50, 10]]
        subscribed_vehicles = {}
        SPEED_THRESHOLD = 13.0

        previous_vehicle_ids = set()

        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            yield self.env.timeout(1)
            print("Time in SimPy:", self.env.now)
            simulation_time = traci.simulation.getTime()
            print("Time in SUMO:", simulation_time)

            if self.env.now < 4:
                pass
            else:
                driving_vehicles = set(traci.vehicle.getIDList())
                
                for vehicle_id in driving_vehicles:
                    traci.vehicle.subscribe(vehicle_id, (tc.VAR_SPEED, tc.VAR_POSITION))
                    subscription_results = traci.vehicle.getSubscriptionResults(vehicle_id)
                    speed = subscription_results[tc.VAR_SPEED]
                    position = subscription_results[tc.VAR_POSITION]
                    print(f"vehicle: {vehicle_id}: speed={speed}, position={position}")

                    if self.inROI(position, rois):
                        print(f"Vehicle: {vehicle_id} is in ROI!")
                        subscribed_vehicles[vehicle_id] = subscription_results
                    else:
                        if vehicle_id in subscribed_vehicles.keys():
                            del subscribed_vehicles[vehicle_id]

                # Find vehicles that have left SUMO
                vehicles_left = previous_vehicle_ids - driving_vehicles
                if vehicles_left:
                    print(f"Vehicles that left the simulation at {self.env.now}: {vehicles_left}")

                previous_vehicle_ids = driving_vehicles

                for vehicle_id in vehicles_left:
                    if vehicle_id in subscribed_vehicles.keys():
                        del subscribed_vehicles[vehicle_id]

                ##################
                print("Vehicles that we care about, subscribed vehicles:", subscribed_vehicles.keys())
                print("")
        traci.close()
        print("TraCI disconnected")

    def inROI(self, point, boxes):
        # Unpack the point
        x, y = point
        
        # Iterate through each box
        for box in boxes:
            min_x, min_y, max_x, max_y = box
            # Check if the point is within the bounds of the current box
            if min_x <= x <= max_x and min_y <= y <= max_y:
                return True
        
        # If the point is not in any of the boxes
        return False