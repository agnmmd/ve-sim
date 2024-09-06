import csv
import os
from sim import Sim

class Statistics:
    _initialized = False
    _files = {
        'task_statistics.csv': [
            'Task ID', 'Source Car ID', 'Time of Arrival', 'Deadline', 'Priority',
            'Complexity', 'Status', 'Processing Car', 'Processing Start', 'Processing End',
            'Repetition', 'Policy', 'Run'
        ],
        'car_statistics.csv': [
            'Car ID', 'Generated Tasks', 'Processed Tasks', 'Successful Tasks',
            'Queued Tasks', 'Total Processing Time', 'Lifetime', 'Repetition', 'Policy', 'Run'
        ]
    }

    @classmethod
    def _initialize_files(cls):
        directory = 'results'
        os.makedirs(directory, exist_ok=True)
        for filename, fieldnames in cls._files.items():
            filepath = os.path.join(directory, filename)
            with open(filepath, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter="\t")
                writer.writeheader()
        cls._initialized = True

    @classmethod
    def _check_initialization(cls):
        if not cls._initialized:
            cls._initialize_files()

    @classmethod
    def _save_stats(cls, filename, data):
        cls._check_initialization()
        filepath = os.path.join('results', filename)
        fieldnames = cls._files[filename]
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
            'Repetition': Sim.repetition,
            'Policy': Sim.policy_name,
            'Run': Sim.run
        }
        cls._save_stats('task_statistics.csv', data)

    @classmethod
    def save_car_stats(cls, car, current_time):
        data = {
            'Car ID': car.id,
            'Generated Tasks': len(car.generated_tasks),
            'Processed Tasks': car.processed_tasks_count,
            'Successful Tasks': car.successful_tasks,
            'Queued Tasks': len(car.assigned_tasks),
            'Total Processing Time': car.total_processing_time,
            'Lifetime': current_time - car.time_of_arrival,
            'Repetition': Sim.repetition,
            'Policy': Sim.policy_name,
            'Run': Sim.run
        }
        cls._save_stats('car_statistics.csv', data)