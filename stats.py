import csv
import os
from sim import Sim

class Statistics:
    _initialized = False

    @classmethod
    def _initialize_file(cls, filename, fieldnames):
        directory = 'results'
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, filename)
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    @classmethod
    def _initialize_files(cls):
        cls._initialize_file('task_statistics.csv', [
            'Task ID', 'Source Car ID', 'Time of Arrival', 'Deadline', 'Priority', 
            'Complexity', 'Processing Start', 'Processing End', 'Processing Car', 
            'Repetition', 'Policy', 'Run'
        ])
        cls._initialize_file('car_statistics.csv', [
            'Car ID', 'Generated Tasks', 'Processed Tasks', 'Successful Tasks', 
            'Total Processing Time', 'Lifetime', 'Repetition', 'Policy', 'Run'
        ])
        cls._initialized = True

    @classmethod
    def _check_initialization(cls):
        if not cls._initialized:
            cls._initialize_files()

    @staticmethod
    def _save_stats(filename, fieldnames, data):
        Statistics._check_initialization()
        filepath = os.path.join('results', filename)
        with open(filepath, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(data)

    @staticmethod
    def save_task_stats(task, processing_car_id):
        fieldnames = [
            'Task ID', 'Source Car ID', 'Time of Arrival', 'Deadline', 'Priority', 
            'Complexity', 'Processing Start', 'Processing End', 'Processing Car', 
            'Repetition', 'Policy', 'Run'
        ]
        data = {
            'Task ID': task.id,
            'Source Car ID': task.source_car.id,
            'Time of Arrival': task.time_of_arrival,
            'Deadline': task.deadline,
            'Priority': task.priority,
            'Complexity': task.complexity,
            'Processing Start': task.processing_start,
            'Processing End': task.processing_end,
            'Processing Car': processing_car_id,
            'Repetition': Sim.repetition,
            'Policy': Sim.policy_name,
            'Run': Sim.run
        }
        Statistics._save_stats('task_statistics.csv', fieldnames, data)

    @staticmethod
    def save_car_stats(car, current_time):
        fieldnames = [
            'Car ID', 'Generated Tasks', 'Processed Tasks', 'Successful Tasks', 
            'Total Processing Time', 'Lifetime', 'Repetition', 'Policy', 'Run'
        ]
        data = {
            'Car ID': car.id,
            'Generated Tasks': len(car.generated_tasks),
            'Processed Tasks': car.processed_tasks_count,
            'Successful Tasks': car.successful_tasks,
            'Total Processing Time': car.total_processing_time,
            'Lifetime': current_time - car.time_of_arrival,
            'Repetition': Sim.repetition,
            'Policy': Sim.policy_name,
            'Run': Sim.run
        }
        Statistics._save_stats('car_statistics.csv', fieldnames, data)


# Example of usage:
# At the beginning of the simulation, you do not need to call any initialization method explicitly.
# The first call to save_task_stats or save_car_stats will ensure the files are reset.
# Then use save_task_stats() and save_car_stats() during the simulation.
