import traci
import traci.constants as tc
from car import Car

class TraciManager:
    def __init__(self, env, sim):
        self.env = env
        self.sim = sim

        self.rois = []
        self.subscribed_vehicles = {}
        self.subscribed_vehicles_list = []

    def execute_one_time_step(self):
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

                    # Subscribe to all vehicles if no ROI is set; Check if the vehicle is in ROI
                    if not self.rois or self.inROI(position, self.rois):
                        print(f"Vehicle: {vehicle_id} is in ROI!")
                        if vehicle_id not in self.subscribed_vehicles:
                            self.subscribed_vehicles[vehicle_id] = Car(self.env, self.sim, speed=speed, position=position)
                            self.subscribed_vehicles[vehicle_id].generate_tasks_static(1)
                            # self.env.process(self.subscribed_vehicles[vehicle_id].generate_tasks())
                        else:
                            self.subscribed_vehicles[vehicle_id].update(speed=speed, position=position)
                    else:
                        # If the vehicle is not in ROI anymore, remove it from subscribed_vehicles
                        if vehicle_id in self.subscribed_vehicles:
                            del self.subscribed_vehicles[vehicle_id]

                # Find vehicles that have left SUMO
                vehicles_left = previous_vehicle_ids - driving_vehicles
                if vehicles_left:
                    print(f"Vehicles that left the simulation at {self.env.now}: {vehicles_left}")

                previous_vehicle_ids = driving_vehicles

                for vehicle_id in vehicles_left:
                    # Print statistics
                    self.subscribed_vehicles[vehicle_id].finish()
                    # Remove from the dictionary vehicle_id:vehicle object
                    del self.subscribed_vehicles[vehicle_id]

                self.subscribed_vehicles_list = self.subscribed_vehicles.values()
                print("Test:", self.subscribed_vehicles_list)

                print("Vehicles that we care about, subscribed vehicles:", self.subscribed_vehicles.keys())
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

    def set_rois(self, rois):
        self.rois = rois

    def get_subscribed_vehicles_list(self):
        return self.subscribed_vehicles_list