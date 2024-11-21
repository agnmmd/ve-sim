import csv
import os
from sim import Sim
from input_manager import InputManager

class Statistics:
    _initialized = False
    _files = {
        'task' :  [
            'Task ID', 'Source Car ID', 'Time of Arrival', 'Deadline', 'Priority',
            'Complexity', 'Status', 'Processing Car', 'Processing Start', 'Processing End',
            'Repetition', 'Policy', 'Run'
        ],
        'car' : [
            'Car ID', 'Generated Tasks', 'Processed Tasks', 'Successful Tasks',
            'Queued Tasks', 'Processing power' ,'Total Processing Time', 'Lifetime', 'Repetition', 'Policy', 'Run'
        ]
    }

    @staticmethod
    def _filename(stat_type):
        filename = [
        f"_r_{InputManager.parsed_args.run}",
        f"_p_{InputManager.parsed_args.policy}" if InputManager.parsed_args.policy else "",
        f"_rep_{InputManager.parsed_args.repetition}" if InputManager.parsed_args.repetition else "",
        f"_cf_{InputManager.parsed_args.configfile}" if InputManager.parsed_args.configfile else "",
        f"_c_{InputManager.parsed_args.configurations}" if InputManager.parsed_args.configurations else "",
        f"_tg_{InputManager.parsed_args.task_generation}" if InputManager.parsed_args.task_generation else "",
        f"_s_{InputManager.parsed_args.start}" if InputManager.parsed_args.start else "",
        f"_dur_{InputManager.parsed_args.duration}" if InputManager.parsed_args.duration else "",
        f"_tc_{InputManager.parsed_args.task_complexity}" if InputManager.parsed_args.task_complexity else "",
        f"_tp_{InputManager.parsed_args.task_priority}" if InputManager.parsed_args.task_priority else "",
        f"_td_{InputManager.parsed_args.task_deadline}" if InputManager.parsed_args.task_deadline else "",
        f"_cp_{InputManager.parsed_args.car_processing_power}" if InputManager.parsed_args.car_processing_power else ""
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
            'Repetition': Sim.repetition,
            'Policy': Sim.policy_name,
            'Run': Sim.run
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
            'Repetition': Sim.repetition,
            'Policy': Sim.policy_name,
            'Run': Sim.run
        }
        cls._save_stats('car', data)