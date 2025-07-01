import argparse
import configparser
import random
import numpy as np
import re
import itertools
import os

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
        matches = re.findall(r"[-+]?\d*\.?\d+", content)

        if matches:
            return matches
        else:
            return [item.strip() for item in content.split(",")]
    return [items]

def get_number(number):
    try:
        return int(number)
    except ValueError:
        try:
            return float(number)
        except ValueError:
            return None

def get_parameter_type(string):
    if "/" in string or "\\" in string:
        return "path"
    elif ":" in string:
        return "time"
    elif "(" in string:
        return "range"
    elif isinstance(get_number(string), (int, float)):
        return "number"
    elif string.strip().lower() in {"true", "false"}:
        return "boolean"
    else:
        return "str"
class InputManager:
    distribution_map = {
        "exponential": np.random.exponential,
        "normal": random.normalvariate,
        "randint": random.randint,
        "poisson": np.random.poisson,
        "uniform": np.random.uniform
    } 

    # TODO: get rid of the class variables if not absolutely necessary. Use local variables whenever possible.
    scenario_args = None
    all_runs = dict()
    command_line_args = None

    @classmethod
    def parse_command_line_arguments(cls):
        # Parse command-line arguments
        parser = argparse.ArgumentParser(description="")
        parser.add_argument("-cf", "--configfile", required = True, type=str, help="Specifies the name of the configuration file to load simulation settings.")
        parser.add_argument("-c", "--sim_config", type=str, help="Specifies which simulation or scenario configuration should be executed.")
        parser.add_argument("-r", "--run", type=int, help="Specifies the unique run number for the simulation. A run number is required to execute the simulation.")
        # print the parameters in terminal
        parser.add_argument("--dry-run", action='store_true', help="Displays the parameters for the specified simulation.")
        # To train a reinforcement learning (RL) model
        parser.add_argument("--train", action='store_true', help="To execute RL training" )
        parser.add_argument("--episodes" , type=int, help="Defines the upper limit for the number of training episodes, but do not specify the run number during training.")
        # Store parsed arguments in the class variable
        cls.command_line_args = parser.parse_args()
        return cls.command_line_args

    @staticmethod
    def parse_configfile_arguments(config, section):
        if section not in config:
            raise "section does not exist "

        configfile_args = {
            parameter: get_items(config.get(section, parameter))
            for parameter in config.options(section)
        }

        return configfile_args

    @classmethod
    def finalize_parameters(cls, sim_config_runs, episode = None):
        run = cls.command_line_args.run
        episodes = cls.command_line_args.episodes
        sim_config = cls.command_line_args.sim_config
        
        if cls.command_line_args.train and episode is not None:
            run_parameters = sim_config_runs[(episode, sim_config)]
        else:
            run_parameters = sim_config_runs[(run, sim_config)]

        for key, value in run_parameters.items():
            run_parameters[key] = cls.resolve_parameter(value)

        cls.scenario_args = {
            **run_parameters,
            "configfile": cls.command_line_args.configfile,
            "run": run,
            "sim_config": sim_config,
            "repetition": run % get_number(run_parameters["repeat"]) if run is not None else None,
            "end": run_parameters["start"] + run_parameters["duration"],
            "episode": episode,
            "episodes": episodes,
            "seed": run % get_number(run_parameters["repeat"]) if run is not None else episode
        }

    @classmethod
    def compile_all_runs_from_configfile(cls):
        """
        Prepare simulation runs by processing configuration file and generating scenarios for all sections.
        """
        # Load configuration file
        config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation(), inline_comment_prefixes=("#", ";"))
        config.read(cls.command_line_args.configfile)

        for section in config.sections():
            configfile_args = cls.parse_configfile_arguments(config, section)

            if cls.command_line_args.train:
                cls.all_episodes_from_configfile(configfile_args, section)
            else : 
                cls.all_runs_from_configfile(configfile_args, section)

        return cls.all_runs
    
    @classmethod
    def all_runs_from_configfile(cls, configfile_args, section):
        run_number = 0
        parameter_cross_product = list(itertools.product(*configfile_args.values()))
        scenarios = [dict(zip(configfile_args.keys(), combination)) for combination in parameter_cross_product]
        repeat = int(configfile_args.get("repeat")[0])

        for scenario in scenarios:
            for _ in range(repeat):
                cls.all_runs[(run_number, section)] = scenario
                run_number += 1

    @classmethod
    def all_episodes_from_configfile(cls, configfile_args, section):
        episode_number = 0
        parameter_cross_product = list(itertools.product(*configfile_args.values()))
        scenarios = [dict(zip(configfile_args.keys(), combination)) for combination in parameter_cross_product]
        episodes = cls.command_line_args.episodes
        
        for scenario in scenarios:
            for _ in range(episodes):
                cls.all_runs[(episode_number, section)] = scenario
                episode_number += 1

    @classmethod
    def get_scenario_args(cls, episode = None):
        # Parse command-line arguments
        cls.parse_command_line_arguments()

        # training mode
        if cls.command_line_args.train and episode is None:
            episode = 0

        # Compile runs from configuration file
        cls.compile_all_runs_from_configfile()

        sim_config_runs = {
            (index, sim_config): dict_of_parameters
            for (index, sim_config), dict_of_parameters in cls.all_runs.items()
            if cls.command_line_args.sim_config == sim_config
        }

        # Finalize parameters
        cls.finalize_parameters(sim_config_runs, episode)

        # Check for missing parameters and raise an error if any are absent
        cls.check_missing_parameters()

        # Return the scenario arguments
        return cls.scenario_args

    @classmethod
    def get_distribution_and_range(cls, input_string):
        '''Filtering the list, keeping only elements that are not fully numeric'''
        pattern = r"(\w+)\((.*?)\)"
        match = re.match(pattern, input_string)
        if match:
            distribution = match.group(1)
            values = match.group(2)
        else:
            raise ValueError(f"the input is not correct")
        if distribution and values:
            return cls.distribution_map.get(distribution.lower(), None), values
        
    @classmethod
    def get_range(cls, string):
        if '(' in str(string):
            distribution, values = cls.get_distribution_and_range(string)
            if "," in values:
                min_val, max_val = values.split(",")
                return distribution(get_number(min_val), get_number(max_val))
            else:
                return distribution(get_number(values))
        else:
            return get_number(string)

    @classmethod
    def resolve_parameter(cls, param):
        param_type = get_parameter_type(param)
        if param_type == "range":
            return lambda: cls.get_range(param)
        elif param_type == "time":
            return convert_time(param)
        elif param_type == "number":
            return get_number(param)
        elif param_type == "path" or param_type == "str":
            return str(param)
        elif param_type == "boolean":
            return param.strip().lower() == "true"
        else:
            raise ValueError(f"Unknown parameter type for '{param}'")

    @staticmethod 
    def find_parameter():
        pattern = r'get_parameter\(["\'](.*?)["\']\)'
        all_python_files = [file for file in os.listdir() if file.endswith('.py')]
        matches = []

        for file in all_python_files:
            with open(file,'r', encoding='utf-8') as f:
                lines = f.readlines()

                for line in lines:
                    stripped_line = line.split("#")[0].strip()  # Remove inline comments
                    if not stripped_line:  # Ignore fully commented lines
                        continue
                    found = re.findall(pattern, line)
                    matches.extend(found)

        return matches
    
    @classmethod
    def check_missing_parameters(cls):
        extracted_parameters = cls.find_parameter()
        config_parameters = [f for f in cls.scenario_args.keys()]
        missing_parameters = []

        for parameter in extracted_parameters:
            if parameter not in config_parameters:
                missing_parameters.append(parameter)

        if missing_parameters:
            raise ValueError(f"Missing parameters in config: {missing_parameters}")
    
        print("All required parameters are present.")

    @classmethod
    def dry_run(cls):
        all_parameters = cls.compile_all_runs_from_configfile()

        if cls.command_line_args.dry_run:
            if cls.command_line_args.run is not None:
                print(f"\nThe parameters for run index ({cls.command_line_args.run}) and  simulation ({cls.command_line_args.sim_config}) are: \n\n{all_parameters[(cls.command_line_args.run , cls.command_line_args.sim_config)]}\n")
            if cls.command_line_args.train is not None:
                print(f"\nThe parameters for episode ({cls.command_line_args.episodes}) and  simulation ({cls.command_line_args.sim_config}) are: \n\n{all_parameters[(cls.command_line_args.episodes -1 , cls.command_line_args.sim_config)]}\n")
            return True
        else:
            return False