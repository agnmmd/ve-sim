import csv
import os
from sim import Sim

class Statistics:
    _initialized = False
    _files = {
        'task' :  [
            'run', 'repetition', 'config', 
            'task_id', 'source_car_id', 'time_of_arrival', 'deadline', 'priority', 
            'complexity', 'status', 'processing_car', 'processing_start', 'processing_end', 'policy', 'lambda_exp'
        ],
        'car' : [
            'run', 'repetition', 'config', 
            'car_id', 'generated_tasks', 'processed_tasks', 'successful_tasks', 'queued_tasks', 'processing_power', 
            'total_processing_time', 'arrival', 'departure', 'lifetime', 'policy', 'lambda_exp'
        ]
    }

    @staticmethod
    def _filename(stat_type):
        filename = [
        f"_r_{Sim.get_parameter('run')}",
        f"_cf_{Sim.get_parameter('configfile')}" if Sim.get_parameter('configfile') else "",
        f"_c_{Sim.get_parameter('sim_config')}" if Sim.get_parameter('sim_config') else "",
    ]
        return f"{stat_type}" + "".join(name for name in filename if name) + ".csv"

    @classmethod
    def _initialize_files(cls):
        # FIXME: The 'results/...' directory path is hardcoded here. We need a smarter handling of this -- Probably from the config file
        directory = 'results' + '/' + Sim.get_parameter('sim_config')
        os.makedirs(directory, exist_ok=True)
        for stat_type, fieldnames in cls._files.items():
            filepath = os.path.join(directory, Statistics._filename(stat_type))
            with open(filepath, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter="\t")
                writer.writeheader()
        cls._initialized = True

    @classmethod
    def _check_initialization(cls):
        if not cls._initialized:
            cls._initialize_files()

    @classmethod
    def _save_stats(cls, stat_type, data):
        cls._check_initialization()
        # FIXME: The 'results/...' directory path is hardcoded here. We need a smarter handling of this -- Probably from the config file
        filepath = os.path.join('results' + '/' + Sim.get_parameter('sim_config'), Statistics._filename(stat_type))
        fieldnames = cls._files[stat_type]
        with open(filepath, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter="\t")
            writer.writerow(data)

    @classmethod
    def save_task_stats(cls, task, processing_car_id):
        data = {
            'task_id': task.id,
            'source_car_id': task.source_car.id,
            'time_of_arrival': task.time_of_arrival,
            'deadline': task.deadline,
            'priority': task.priority,
            'complexity': task.complexity,
            'status': task.status,
            'processing_car': processing_car_id,
            'processing_start': task.processing_start,
            'processing_end': task.processing_end,
            'repetition': Sim.get_parameter('repetition'),
            'policy': Sim.get_parameter('policy_name'),
            'run': Sim.get_parameter('run'),
            'config': Sim.get_parameter('sim_config'),
            'lambda_exp': Sim.get_parameter('lambda_exp')() # FIXME: here I am executing the lambda to get the value that I need. This needs to be handled in a smarter way by Sim.get_parameter()
        }
        cls._save_stats('task', data)

    @classmethod
    def save_car_stats(cls, car, current_time):
        data = {
            'car_id': car.id,
            'generated_tasks': car.generated_tasks_count,
            'processed_tasks': car.processed_tasks_count,
            'successful_tasks': car.successful_tasks,
            'queued_tasks': len(car.assigned_tasks),
            'processing_power': car.processing_power,
            'total_processing_time': car.total_processing_time,
            'arrival': car.time_of_arrival,
            'departure': current_time,
            'lifetime': current_time - car.time_of_arrival,
            'repetition': Sim.get_parameter('repetition'),
            'policy': Sim.get_parameter('policy_name'),
            'run': Sim.get_parameter('run'),
            'config': Sim.get_parameter('sim_config'),
            'lambda_exp': Sim.get_parameter('lambda_exp')() # FIXME: here I am executing the lambda to get the value that I need. This needs to be handled in a smarter way by Sim.get_parameter()
        }
        cls._save_stats('car', data)
