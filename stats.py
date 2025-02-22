import csv
import os
from sim import Sim

class Statistics:
    _initialized = False
    _files = {
        'task' :  [
            'Run', 'Repetition', 'Config',
            'Task ID', 'Source Car ID', 'Time of Arrival', 'Deadline', 'Priority',
            'Complexity', 'Status', 'Processing Car', 'Processing Start', 'Processing End', 'Policy'
        ],
        'car' : [
            'Run', 'Repetition', 'Config',
            'Car ID', 'Generated Tasks', 'Processed Tasks', 'Successful Tasks',
            'Queued Tasks', 'Processing power' ,'Total Processing Time', 'Lifetime', 'Policy'
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
        directory = 'results'
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
        filepath = os.path.join('results', Statistics._filename(stat_type))
        fieldnames = cls._files[stat_type]
        with open(filepath, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter="\t")
            writer.writerow(data)

    @classmethod
    def save_task_stats(cls, task, processing_car_id):
        data = {
            'Task ID': task.id,
            'Source Car ID': task.source_car.id,
            'Time of Arrival': task.time_of_arrival,
            'Deadline': task.deadline,
            'Priority': task.priority,
            'Complexity': task.complexity,
            'Status': task.status,
            'Processing Car': processing_car_id,
            'Processing Start': task.processing_start,
            'Processing End': task.processing_end,
            'Repetition': Sim.get_parameter('repetition'),
            'Policy': Sim.get_parameter('policy_name'),
            'Run': Sim.get_parameter('run'),
            'Config': Sim.get_parameter('sim_config')
            'lambda_exp': Sim.get_parameter('lambda_exp')() # FIXME: here I am executing the lambda to get the value that I need. This needs to be handled in a smarter way by Sim.get_parameter()
        }
        cls._save_stats('task', data)

    @classmethod
    def save_car_stats(cls, car, current_time):
        data = {
            'Car ID': car.id,
            'Generated Tasks': car.generated_tasks_count,
            'Processed Tasks': car.processed_tasks_count,
            'Successful Tasks': car.successful_tasks,
            'Queued Tasks': len(car.assigned_tasks),
            'Processing power' : car.processing_power,
            'Total Processing Time': car.total_processing_time,
            'Lifetime': current_time - car.time_of_arrival,
            'Repetition': Sim.get_parameter('repetition'),
            'Policy': Sim.get_parameter('policy_name'),
            'Run': Sim.get_parameter('run'),
            'Config': Sim.get_parameter('sim_config')
            'lambda_exp': Sim.get_parameter('lambda_exp')() # FIXME: here I am executing the lambda to get the value that I need. This needs to be handled in a smarter way by Sim.get_parameter()
        }
        cls._save_stats('car', data)
