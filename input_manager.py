import argparse
import configparser
import random
from policy import Policy
import random
import numpy as np
import re
import itertools

class InputManager:

    policy_map = {
        "highest_priority": Policy.p_highest_priority,
        "shortest_deadline": Policy.p_shortest_deadline,
        "earliest_deadline": Policy.p_earliest_deadline,
        "lowest_complexity": Policy.p_lowest_complexity,
        "random": Policy.p_random ,
    }

    Distribution_map = {
        "exponential" : np.random.exponential,
        "normal" : random.normalvariate,
        "randint" : random.randint,
        "poisson" : np.random.poisson ,
    }

    configfile = None
    parsed_args = None
    config_args = None
    merged_args =   None
    scenario_args = None
    run_index = 0
    runs = dict()

    @classmethod
    def parse_arguments(cls):
        #Parse command-line arguments
        parser = argparse.ArgumentParser(description="General : For running the simulation, the policy name must be specified. All other parameters can be provided either via command line or in a configuration file. Command-line parameters take priority over those in the config file, so if a parameter is specified in both, the command-line value will be used.")
        parser.add_argument("-r", "--run", type=int , help="Specifies the unique run number for the simulation.A run number is required to execute the simulation.If a config file is provided, use --dry-run to view all available run numbers and configurations.")
        parser.add_argument("-p", "--policy", type=str,  help="Policy selection for task prioritization in the simulation. Options include 'highest_priority', 'shortest_deadline', 'lowest_complexity', or 'random'.")
        parser.add_argument("-rep", "--repetition", type=int,   help="Defines the number of times the simulation will repeat. Each repetition uses the same parameters for repeatability. it should start from 1  , example :   -rep 1 ")
        parser.add_argument("-cf", "--configfile", type=str,  help="Specifies the name of the configuration file to load simulation settings. If a configuration file is not provided, all required simulation parameters must be specified as command-line arguments.")
        parser.add_argument("-c", "--configurations", type=str ,  help="Defines the name of the specific simulation or scenario configuration to run.")
        parser.add_argument("-s","--start", type=str, help="Specifies the start time (hh:mm) for the simulation. Use format '05:00' for hour and minute, or a single integer for seconds (e.g., '300' for 5 minutes). If not provided, defaults will be used from the configfile")
        parser.add_argument("-dur","--duration", type=str, help="Sets the duration of the simulation. Provide time in 'HH:MM' format (e.g., '00:10' for 10 minutes) or as a single integer in seconds (e.g., '600' for 10 minutes). Uses default if unspecified.")
        parser.add_argument("-tc","--task_complexity", type=str, help="Defines the task complexity level. Accepts either an integer (e.g., 2) or a random distribution expression like 'randint(0,3)' for a random complexity range.")
        parser.add_argument("-tp","--task_priority", type=str, help="Sets the priority level of tasks. Accepts an integer or random distribution format, e.g., 'randint(0,3)', for variable priority values.")
        parser.add_argument("-td","--task_deadline", type=str, help="Specifies the task deadline. Can be an integer or a random distribution like 'randint(0,3)' for randomly generated deadlines.")
        parser.add_argument("-cp","--car_processing_power", type=str, help="Defines the processing power of each car in the simulation. Provide as an integer (default 2) or a distribution, e.g., 'randint(0,3)' for random power levels.")
        parser.add_argument("-tg", "--task_generation" , type = str , help="Specifies the task generation method. Choose 'static' for a fixed number of tasks (provide an integer count) or 'dynamic' for tasks generated on the fly (provide a distribution, e.g., 'poisson(3)' for a Poisson distribution or 'randint(0, 5)' for task deadlines with random values between 0 and 5). This controls the frequency and timing of task creation.")
        # print the parameters in terminal
        parser.add_argument("--dry-run" , action='store_true' , help="Displays the parameters for the specified simulation. If a run index is not provided, all scenarios for the given config file and configuration will be shown.")
        # to combine files
        parser.add_argument("--combine", action="store_true", help="Combines all CSV files from the results directory into a single file. Useful for aggregating results from multiple runs.")  # New argument to combine files

        # Store parsed arguments in the class variable
        cls.parsed_args = parser.parse_args()
        cls.parsed_args_errors()

    @classmethod
    def configfile_arguments(cls , simulation):
        cls.config_args = {
            'policy' : cls.get_items(cls.configfile.get(simulation,'policy')),
            'repetition':  cls.get_items(cls.configfile.get(simulation,'repetition')),
            'task_generation':cls.get_items(cls.configfile.get(simulation , 'task_generation' , fallback= '1')),
            'start':  cls.get_items(cls.configfile.get(simulation , 'start' )),    #Use fallback only if you need a specific default value beyond what DEFAULT provides or to prevent
            'duration':  cls.get_items(cls.configfile.get(simulation, 'duration' )),
            'task_complexity':  cls.get_items(cls.configfile.get(simulation, 'task_complexity' )),
            'task_priority':  cls.get_items(cls.configfile.get(simulation, 'task_priority')),
            'task_deadline':  cls.get_items(cls.configfile.get(simulation, 'task_deadline')),
            'car_processing_power': cls.get_items(cls.configfile.get(simulation, 'car_processing_power')),
        }
        return cls.config_args

    @classmethod
    def parsed_args_errors(cls):
        if cls.parsed_args.repetition != None :
            if cls.parsed_args.repetition <= 0 :
                raise ValueError("\n Error: the rep can't be 0 or less than 0. for more info use -h, --help ")
        if cls.parsed_args.configurations is not None and cls.parsed_args.configfile is None:
            raise ValueError("\nError: Some required arguments are missing. Please provide configfile , -cf. for more info use -h, --help ")
        if  cls.parsed_args.configfile is None and (cls.parsed_args.run is None or cls.parsed_args.policy is None or cls.parsed_args.task_generation is None or cls.parsed_args.start is None or cls.parsed_args.duration is None or cls.parsed_args.task_complexity is None or cls.parsed_args.task_priority is None or cls.parsed_args.task_deadline is None or  cls.parsed_args.car_processing_power is None ): 
            raise ValueError("\nError: Some required arguments are missing. Please provide -configfile or simulation parameter -tc and -tp ect. for more info use -h, --help")
        if cls.parsed_args.dry_run and ( cls.parsed_args.configfile is None or cls.parsed_args.configurations is None) :
            raise ValueError("\nError : To view the parameters, specify  configuration file, and desired configurations. For additional details, use -h or --help.")

    @classmethod
    def merge_config_and_parsed_args(cls):
        if cls.configfile:
            cls.get_run_number()
            run  = cls.runs[(cls.parsed_args.run , cls.parsed_args.configurations)]

        """Merge parsed_args and config_args with fallback to configfile values."""
        cls.merged_args = {
            'run': cls.parsed_args.run,
            'policy': cls.parsed_args.policy if cls.parsed_args.policy else run['policy'],
            'repetition': cls.parsed_args.repetition if cls.parsed_args.repetition else run['repetition'],
            'configfile': cls.parsed_args.configfile,
            'configurations': cls.parsed_args.configurations,
            'task_generation': cls.parsed_args.task_generation if cls.parsed_args.task_generation else run['task_generation'],
            'start': cls.parsed_args.start if cls.parsed_args.start else run['start'],
            'duration': cls.parsed_args.duration if cls.parsed_args.duration else run['duration'],
            'task_complexity': cls.parsed_args.task_complexity if cls.parsed_args.task_complexity else run['task_complexity'],
            'task_priority': cls.parsed_args.task_priority if cls.parsed_args.task_priority else run['task_priority'],
            'task_deadline': cls.parsed_args.task_deadline if cls.parsed_args.task_deadline else run['task_deadline'],
            'car_processing_power': cls.parsed_args.car_processing_power if cls.parsed_args.car_processing_power else run['car_processing_power'],
        }
        return cls.merged_args

    @classmethod
    def scenario_arguments(cls):
        ''' Here we get the str or int from command line or configfile and prepare them for use in the code. '''
        cls.merge_config_and_parsed_args()
        
        cls.scenario_args = {
            'run': cls.merged_args['run'],
            'policy': cls.policy_map.get(cls.merged_args['policy'], None) ,
            'repetition': cls.merged_args['run'] % cls.range_int(cls.merged_args['repetition']),
            'configfile':cls.merged_args['configfile'],
            'configurations': cls.merged_args['configurations'],
            'task_generation': lambda: cls.range_int(cls.merged_args['task_generation']),
            'start': cls.time_converter(cls.merged_args['start']),
            'duration': cls.time_converter(cls.merged_args['duration']),
            'end': cls.time_converter(cls.merged_args['start']) + cls.time_converter(cls.merged_args['duration']),
            'task_complexity': lambda: cls.range_int(cls.merged_args['task_complexity']),
            'task_priority': lambda: cls.range_int(cls.merged_args['task_priority']),
            'task_deadline': lambda: cls.range_int(cls.merged_args['task_deadline']),
            'car_processing_power': lambda: cls.range_int(cls.merged_args['car_processing_power']),
        }
        return cls.scenario_args

    @classmethod
    def load_config(cls):
        file_name = cls.parsed_args.configfile
        cls.configfile = configparser.ConfigParser(interpolation = configparser.ExtendedInterpolation())
        if file_name:
            try:
                cls.configfile.read(file_name)
            except FileNotFoundError:
                print(f"Error: Configuration file '{file_name}' not found.")
                exit(1)

    @classmethod
    def get_distribution_and_range(cls,input_string):
        ''' Filtering the list, keeping only elements that are not fully numeric'''
        pattern = r"(\w+)\((.*?)\)"
        match = re.match(pattern , input_string)
        if match:
            distribution =  match.group(1)
            values = match.group(2)
        else :
            raise ValueError(f"the input is not correct")
        if distribution and values:
            return cls.Distribution_map.get(distribution.lower(), None) , values 

    @classmethod
    def range_int(cls, string):
        if '(' in string:
            distribution , values = cls.get_distribution_and_range(string) 
            if "," in values: 
                min_val , max_val = values.split(",")
                return   distribution(int(min_val) ,int(max_val))
            else: 
                return distribution(int(values))
        else : 
            return int(string)

    @classmethod
    def time_converter(cls, time):
        #Convert time from hh:mm format to seconds.
        if ':' in str(time):
            hours, minutes = map(int, time.split(':'))
            return (hours * 60 * 60) + (minutes * 60)
        return int(time)
    
    @staticmethod
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
    
    @classmethod
    def get_items(cls,items):
        if items.startswith("(") and "," in items and not items.endswith(")"):
            raise ValueError("Input starts with '(' and contains ',' but does not end with ')'.")
        elif  cls.has_matching_parentheses(items) is False:
            raise ValueError("Input has wrong number of ( ) .")
        if items.startswith("(") and items.endswith(")"):
            content = items[1:-1]
            matches = re.findall(r"(\w+\(.*?\)|\d+)", content)
            if matches :
                return matches
            else : return [item.strip() for item in content.split(",")]
        return [items]
    
    @classmethod
    def show_parameter(cls):
        if  cls.parsed_args.configurations and cls.parsed_args.run  :
            print(f"\nThe parameters for run index ({cls.parsed_args.run}) and  simulation ({cls.parsed_args.configurations}) are: \n\n{cls.runs[(cls.parsed_args.run , cls.parsed_args.configurations)]}\n")
        else : 
            for (run_index,simulation), run_parameter in cls.runs.items() :
                if simulation == cls.parsed_args.configurations and cls.parsed_args.run is None : 
                    print(f"\nThe parameters for run index ({run_index}) and  simulation ({simulation}) are: \n\n{run_parameter}\n")

    @classmethod
    def get_all_scenarios(cls,simulation):
        cls.configfile_arguments(simulation)
        keys = list(cls.config_args.keys())
        values = list(cls.config_args.values())
        all_combinations = list(itertools.product(*values))
        scenarios = [
            dict(zip(keys, combination)) for combination in all_combinations
        ]
        return scenarios

    @classmethod
    def get_run_number(cls):
        if cls.parsed_args.configurations == None:
            simulations = cls.configfile.sections() 
        else :
            simulations = [cls.parsed_args.configurations]

        for simulation in simulations:
            cls.run_index = 0  # Reset index when the scenario changes
            scenarios = cls.get_all_scenarios(simulation)
            for scenario in scenarios:
                repeat = cls.parsed_args.repetition or int(scenario['repetition'])
                for rep in range(repeat):
                    cls.runs[( cls.run_index, simulation)] = scenario
                    cls.run_index += 1