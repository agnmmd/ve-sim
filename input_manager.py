import argparse
import configparser
from policy import Policy
import random
import numpy as np
import re
import itertools

def convert_time(time):
    """Convert time from hh:mm format to seconds."""
    if ':' in str(time):
        hours, minutes = map(int, time.split(':'))
        return (hours * 60 * 60) + (minutes * 60)
    return int(time)

def has_matching_parentheses(s):
    stack = []
    for char in s:
        if char == "(":
            stack.append("(")
        elif char == ")":
            if not stack:
                return False
            stack.pop()  
    return len(stack) == 0

def get_items(items):
    if items is None:
        return
    if items.startswith("(") and "," in items and not items.endswith(")"):
        raise ValueError("Input starts with '(' and contains ',' but does not end with ')'.")
    if has_matching_parentheses(items) is False:
        raise ValueError("Input has wrong number of ( ).")
    if items.startswith("(") and items.endswith(")"):
        content = items[1:-1]
        matches = re.findall(r"(\w+\(.*?\)|\d+\.\d+|\d+)", content)
        if matches:
            return matches
        else:
            return [item.strip() for item in content.split(",")]
    return [items]


class InputManager:
    distribution_map = {
        "exponential": np.random.exponential,
        "normal": random.normalvariate,
        "randint": random.randint,
        "poisson": np.random.poisson,
    } 

    # TODO: get rid of the class variables if not absolutely necessary. Use local variables whenever possible.
    scenario_args = None

    @staticmethod
    def parse_command_line_arguments():
        #Parse command-line arguments
        parser = argparse.ArgumentParser(description="")
        parser.add_argument("-cf", "--configfile", required = True, type=str, help="Specifies the name of the configuration file to load simulation settings. If a configuration file is not provided, all required simulation parameters must be specified as command-line arguments.")
        parser.add_argument("-c", "--sim_config", type=str, help="Defines the name of the specific simulation or scenario configuration to run.")
        parser.add_argument("-r", "--run", type=int, help="Specifies the unique run number for the simulation.A run number is required to execute the simulation.If a config file is provided, use --dry-run to view all available run numbers and sim_config.")
        # print the parameters in terminal
        # parser.add_argument("--dry-run", action='store_true', help="Displays the parameters for the specified simulation. If a run index is not provided, all scenarios for the given config file and configuration will be shown.")
        # Store parsed arguments in the class variable
        command_line_args = parser.parse_args()
        return command_line_args

    @staticmethod
    def parse_configfile_arguments(config, section):
        configfile_args = {
            'repeat': get_items(config.get(section, 'repeat')),
            'start': get_items(config.get(section, 'start' )), # Use fallback only if you need a specific default value beyond what DEFAULT provides 
            'duration': get_items(config.get(section, 'duration' )),
            'policy': get_items(config.get(section, 'policy')),
            'task_generation': get_items(config.get(section , 'task_generation')),
            'task_complexity': get_items(config.get(section, 'task_complexity' )),
            'task_priority': get_items(config.get(section, 'task_priority')),
            'task_deadline': get_items(config.get(section, 'task_deadline')),
            'car_processing_power': get_items(config.get(section, 'car_processing_power')),
            'sumo_binary': get_items(config.get(section, 'sumo_binary')),
            'sumo_cfg': get_items(config.get(section, 'sumo_cfg')),
            'sumo_step_length': get_items(config.get(section, 'sumo_step_length')),
            'traci_step_length': get_items(config.get(section, 'traci_step_length')),
            'lambda_exp': get_items(config.get(section, 'lambda_exp')),
            # ROI
            'roi_min_x': get_items(config.get(section, 'roi_min_x')),
            'roi_min_y': get_items(config.get(section, 'roi_min_y')),
            'roi_max_x': get_items(config.get(section, 'roi_max_x')),
            'roi_max_y': get_items(config.get(section, 'roi_max_y')),
            # RL
            'max_tasks': get_items(config.get(section , 'max_tasks')),
            'max_cars': get_items(config.get(section, 'max_cars')),
            'max_speed': get_items(config.get(section, 'max_speed')),
            'max_complexity': get_items(config.get(section, 'max_complexity')),
            'max_deadline': get_items(config.get(section, 'max_deadline')),
            'max_processing_power': get_items(config.get(section, 'max_processing_power')),
            'n_episodes': get_items(config.get(section, 'n_episodes')),
            # 'memory': get_items(config.get(section, 'memory')),
            'replay_buffer_capacity': get_items(config.get(section, 'replay_buffer_capacity')),
            'batch_size': get_items(config.get(section, 'batch_size')),
            'learning_rate': get_items(config.get(section, 'learning_rate')),
            'gamma': get_items(config.get(section, 'gamma')),
            'epsilon_decay_rate': get_items(config.get(section, 'epsilon_decay_rate')),
            'epsilon_min': get_items(config.get(section, 'epsilon_min')),
            'epsilon_max': get_items(config.get(section, 'epsilon_max')),
            'target_update_freq': get_items(config.get(section, 'target_update_freq')),
        }
        return configfile_args

    @classmethod
    def finalize_parameters(cls, command_line_args, sim_config_runs):
        run = command_line_args.run
        sim_config = command_line_args.sim_config
        run_parameters  = sim_config_runs[(run, sim_config)]

        cls.scenario_args = {
            'configfile': command_line_args.configfile,
            'run': run,
            'sim_config': sim_config,
            'repeat': int(run_parameters['repeat']),
            'repetition': run % int(run_parameters['repeat']),
            'start': convert_time(run_parameters['start']),
            'duration': convert_time(run_parameters['duration']),
            'end': convert_time(run_parameters['start']) + convert_time(run_parameters['duration']),
            'policy_name': run_parameters['policy'],
            # 'policy': Policy.get_policies().get(run_parameters['policy'], None),
            'task_generation': lambda: cls.range_int(run_parameters['task_generation']),
            'task_complexity': lambda: cls.range_int(run_parameters['task_complexity']),
            'task_priority': lambda: cls.range_int(run_parameters['task_priority']),
            'task_deadline': lambda: cls.range_int(run_parameters['task_deadline']),
            'car_processing_power': lambda: cls.range_int(run_parameters['car_processing_power']),
            'sumo_binary': str(run_parameters['sumo_binary']),
            'sumo_cfg': str(run_parameters['sumo_cfg']),
            'sumo_step_length': str(run_parameters['sumo_step_length']),
            'traci_step_length': float(run_parameters['traci_step_length']),
            'lambda_exp': lambda: cls.range_int(run_parameters['lambda_exp']),
            # ROI
            'roi_min_x': int(run_parameters['roi_min_x']),
            'roi_min_y': int(run_parameters['roi_min_y']),
            'roi_max_x': int(run_parameters['roi_max_x']),
            'roi_max_y': int(run_parameters['roi_max_y']),
            # RL
            'max_tasks': int(run_parameters['max_tasks']),
            'max_cars': int(run_parameters['max_cars']),
            'max_speed': int(run_parameters['max_speed']),
            'max_complexity': int(run_parameters['max_complexity']),
            'max_deadline': int(run_parameters['max_deadline']),
            'max_processing_power': int(run_parameters['max_processing_power']), 
            'n_episodes': int(run_parameters['n_episodes']),
            # 'memory': int(run_parameters['memory']),
            'replay_buffer_capacity': int(run_parameters['replay_buffer_capacity']),
            'batch_size': int(run_parameters['batch_size']),
            'learning_rate': float(run_parameters['learning_rate']),
            'gamma': float(run_parameters['gamma']),
            'epsilon_decay_rate': float(run_parameters['epsilon_decay_rate']),
            'epsilon_min': float(run_parameters['epsilon_min']),
            'epsilon_max': float(run_parameters['epsilon_max']),
            'target_update_freq': float(run_parameters['target_update_freq']),
        }

    @classmethod
    def compile_all_runs_from_configfile(cls, configfile):
        """
        Prepare simulation runs by processing configuration file and generating scenarios for all sections.
        """
        # Load configuration file
        config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation(), inline_comment_prefixes=('#',';'))
        config.read(configfile)

        # Reset runs dictionary
        all_runs = dict()

        # Process all sections in the configuration file
        for section in config.sections():
            run_number = 0

            configfile_args = cls.parse_configfile_arguments(config, section)

            # Use itertools to generate all parameter combinations
            parameters_cross_product = list(itertools.product(*configfile_args.values()))
            scenarios = [dict(zip(configfile_args.keys(), combination)) for combination in parameters_cross_product]

            # Assign run numbers based on repetitions and combination of parameters
            repeat = int(configfile_args.get('repeat')[0])

            # Create runs for each scenario
            for scenario in scenarios:
                for _ in range(repeat):
                    # Store run with unique index and section name
                    all_runs[(run_number, section)] = scenario
                    run_number += 1

        return all_runs

    @classmethod
    def get_scenario_args(cls):
        # Parse command-line arguments
        command_line_args = cls.parse_command_line_arguments()
        
        # Compile runs from configuration file
        all_runs = cls.compile_all_runs_from_configfile(command_line_args.configfile)

        sim_config_runs = {(run, sim_config): dict_of_parameters for (run, sim_config), dict_of_parameters in all_runs.items() if command_line_args.sim_config == sim_config}
        
        # Finalize parameters
        cls.finalize_parameters(command_line_args, sim_config_runs)
        
        # Return the scenario arguments
        return cls.scenario_args

    @classmethod
    def get_distribution_and_range(cls, input_string):
        '''Filtering the list, keeping only elements that are not fully numeric'''
        pattern = r"(\w+)\((.*?)\)"
        match = re.match(pattern , input_string)
        if match:
            distribution =  match.group(1)
            values = match.group(2)
        else:
            raise ValueError(f"the input is not correct")
        if distribution and values:
            return cls.distribution_map.get(distribution.lower(), None), values 

    @classmethod
    def range_int(cls, string):
        if '(' in str(string):
            distribution , values = cls.get_distribution_and_range(string) 
            if "," in values:
                min_val, max_val = values.split(",")
                return distribution(int(min_val), int(max_val))
            else:
                return distribution(int(values))
        else:
            return float(string)

    # FIXME: Currently the '--dry-run' option is disabled. Enable it in the ArgumentParser and make it work in this form: 'python3 main.py -cf config_file.ini --dry-run'
    # @classmethod
    # def show_parameter(cls, command_line_args, run_parameters):
    #     if command_line_args.run is not None:
    #          print(f"\nThe parameters for run index ({command_line_args.run}) and  simulation ({command_line_args.sim_config}) are: \n\n{run_parameters[(command_line_args.run , command_line_args.sim_config)]}\n")
    #     else:
    #         for (run_index,simulation), run_parameters in run_parameters.items():
    #             if simulation == command_line_args.sim_config: 
    #                 print(f"\nThe parameters for run index ({run_index}) and simulation ({simulation}) are: \n\n{run_parameters}\n")