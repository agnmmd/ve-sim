from sim import Sim
from car import Car
from scheduler import Scheduler
from utils import print_color
import traci

class SumoManager:
    sumoBinary = ""
    sumo_cfg = ""
    traces = {}

    def set_sumo_binary(cls, path):
        cls.sumoBinary = path
    
    def set_sumo_config_path(cls, path):
        cls.sumo_cfg = path

    def run_sumo_simulation(cls, until):

        sumoCmd = [cls.sumoBinary, "-c", cls.sumo_cfg]
        traci.start(sumoCmd)
        for step in range(until):
            traci.simulationStep()
            
            for v_id in traci.vehicle.getIDList():
                v_pos = traci.vehicle.getPosition(v_id)
                if step not in cls.traces:
                    cls.traces[step] = {}
                cls.traces[step][v_id] = v_pos

        traci.close()
    
    def print_traces(cls):
        for step in cls.traces:
            print_color(f"\n============ Step {step} ============","93")
            for v_id, pos in cls.traces[step].items():
                print(f"  Vehicle {v_id}: x: {pos[0]}, y: {pos[1]}")