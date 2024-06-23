import csv
import os
from sim import Sim

class Statistics:
    _initialized = False

    @classmethod
    def _initialize_files(cls):
        directory = 'results'
        os.makedirs(directory, exist_ok=True)

        # Initialize task statistics file
        task_filename = os.path.join(directory, 'task_statistics.csv')
        with open(task_filename, 'w', newline='') as csvfile:
            fieldnames = ['Task ID', 'Source Car ID', 'Time of Arrival', 'Deadline', 'Priority', 'Complexity', 'Processing Start', 'Processing End', 'Processing Car', 'Repetition', 'Policy', 'Run']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

        # Initialize car statistics file
        car_filename = os.path.join(directory, 'car_statistics.csv')
        with open(car_filename, 'w', newline='') as csvfile:
            fieldnames = ['Car ID', 'Generated Tasks', 'Processed Tasks', 'Successful Tasks', 'Total Processing Time', 'Lifetime']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

        cls._initialized = True

    @classmethod
    def _check_initialization(cls):
        if not cls._initialized:
            cls._initialize_files()

    @staticmethod
    def save_task_stats(task, processing_car_id):
        Statistics._check_initialization()
        directory = 'results'
        filename = os.path.join(directory, 'task_statistics.csv')

        with open(filename, 'a', newline='') as csvfile:
            fieldnames = ['Task ID', 'Source Car ID', 'Time of Arrival', 'Deadline', 'Priority', 'Complexity', 'Processing Start', 'Processing End', 'Processing Car', 'Repetition', 'Policy', 'Run']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({
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
            })

    @staticmethod
    def save_car_stats(car, current_time):
        Statistics._check_initialization()
        directory = 'results'
        filename = os.path.join(directory, 'car_statistics.csv')

        with open(filename, 'a', newline='') as csvfile:
            fieldnames = ['Car ID', 'Generated Tasks', 'Processed Tasks', 'Successful Tasks', 'Total Processing Time', 'Lifetime', 'Repetition', 'Policy', 'Run']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({
                'Car ID': car.id,
                'Generated Tasks': len(car.generated_tasks),
                'Processed Tasks': car.processed_tasks_count,
                'Successful Tasks': car.successful_tasks,
                'Total Processing Time': car.total_processing_time,
                'Lifetime': current_time - car.time_of_arrival,
                'Repetition': Sim.repetition,
                'Policy': Sim.policy_name,
                'Run': Sim.run
            })

# Example of usage:
# At the beginning of the simulation, you do not need to call any initialization method explicitly.
# The first call to save_task_stats or save_car_stats will ensure the files are reset.
# Then use save_task_stats() and save_car_stats() during the simulation.
