import csv
import os
from ve_sim.sim import Sim
import types
class Statistics:
    _initialized = False
    _files = {
        'task',
        'car',
        'action',
        'episode',
    }
    # Required parameters for all stats files.
    # Parameter names must exactly match those in the config file.
    set_shared_params = {
        "run",
        "sim_config",
        "repetition",
        "configfile",
        "task_generation",
        "episode",
        "policy",
        "seed",
        "lambda_exp",
    }
    _shared_params = dict()
    _results_directory = None

    @classmethod
    def get_shared_params(cls):

        for key in cls.set_shared_params:
            if key not in Sim.sim_parameters:
                raise KeyError(f"Parameter '{key}' not found")

            value = Sim.sim_parameters[key]

            if value is None:
                continue

            if isinstance(value, types.FunctionType):
                try:
                    value = value.__closure__[1].cell_contents
                except Exception:
                    value = "<lambda>"

            cls._shared_params[key] = value

        return cls._shared_params

    @classmethod
    def results_directory(cls):
        cls._results_directory = Sim.get_parameter('results_directory') + "/" + Sim.get_parameter("sim_config")
        return cls._results_directory

    @staticmethod
    def _filename(stat_type):
        filename = [
            f"_r_{Sim.get_parameter('run')}" if Sim.get_parameter("episodes") is None else f"_episodes_{Sim.get_parameter('episodes')}",
            f"_cf_{Sim.get_parameter('configfile')}",
            f"_c_{Sim.get_parameter('sim_config')}",
        ]
        return f"{stat_type}" + "".join(name for name in filename if name) + ".csv"

    @classmethod
    def _initialize_files(cls):
        cls.results_directory()
        os.makedirs(cls._results_directory, exist_ok=True)
        for stat_type in cls._files:
            filepath = os.path.join(cls._results_directory, cls._filename(stat_type))
            with open(filepath, "w", newline="") as csvfile:
                pass
        cls._initialized = True

    @classmethod
    def _check_initialization(cls):
        if not cls._initialized:
            cls._initialize_files()

    @classmethod
    def _save_stats(cls, stat_type, data):
        cls._check_initialization()
        filepath = os.path.join(cls._results_directory, cls._filename(stat_type))
        with open(filepath, "a", newline="") as csvfile:
            file_exists = os.path.isfile(filepath)
            is_empty = not file_exists or os.path.getsize(filepath) == 0
            writer = csv.DictWriter(csvfile, fieldnames=data.keys(), delimiter="\t")
            if is_empty:
                writer.writeheader()
            writer.writerow(data)

    @classmethod
    def save_task_stats(cls, task, processing_car_id):
        cls.get_shared_params()
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
            **cls._shared_params,
        }

        cls._save_stats("task", data)

    @classmethod
    def save_car_stats(cls, car, current_time):
        cls.get_shared_params()
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
            'task_generation': car.num_tasks,
            **cls._shared_params,
        }
        cls._save_stats('car', data)

    @classmethod
    def save_action_stats(cls, current_time, action, reward, is_best, resource_count):
        cls.get_shared_params()
        data = {
            'time': current_time,
            'action': action,
            'reward': reward,
            'best_action': is_best,
            'resource_count': resource_count,
            **cls._shared_params,
            
        }
        cls._save_stats('action', data)

    @classmethod
    def save_episode_stats(cls, total_reward, best_selection_ratio, num_actions):
        cls.get_shared_params()
        data = {
            'total_reward': total_reward,
            'best_selection_ratio': best_selection_ratio,
            'num_actions': num_actions,
            **cls._shared_params,
        }
        cls._save_stats('episode', data)
