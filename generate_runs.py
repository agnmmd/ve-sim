from input_manager import InputManager
import os

class GenerateRuns:



    @classmethod
    def initialize_txt_file(cls, text, file_path = 'runs.txt' ):
        
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                f.write("")  
        with open(file_path, 'a') as f:
            f.write(text)

    @classmethod
    def get_scenarios_generate_run_commands(cls):
        if InputManager.parsed_args.configfile :
            for (run_index , scenario ), run in InputManager.runs.items():
                cls.initialize_txt_file(cls.initialize_command(run_index , scenario))
        # else:
        #     cls.initialize_txt_file(cls.initialize_command(InputManager.parsed_args.run , InputManager.parsed_args.configurations))


    # TODO : make sure to have customize command like :python3  generate_runs.py -cf config_file.ini -r -4 -rep 3 -c Simulation4 

    @classmethod
    def runs_guide(cls):
        if InputManager.parsed_args.configfile :
            for (run_index, scenario ), run in InputManager.runs.items():
                cls.initialize_txt_file(f"run_index {run_index} , {scenario}  : {run}\n" , 'runs_guide.txt')


    @classmethod
    def initialize_command(cls,run_index , scenario):
        command = [
            f"-r {run_index}",
            f"-p {InputManager.parsed_args.policy}" if InputManager.parsed_args.policy else "",
            f"-rep {InputManager.parsed_args.repetition}" if InputManager.parsed_args.repetition else "",
            f"-cf {InputManager.parsed_args.configfile}" if InputManager.parsed_args.configfile else "",
            f"-c {InputManager.parsed_args.configurations or scenario}",
            f"-tg {InputManager.parsed_args.task_generation}" if InputManager.parsed_args.task_generation else "",
            f"-s {InputManager.parsed_args.start}" if InputManager.parsed_args.start else "",
            f"-dur {InputManager.parsed_args.duration}" if InputManager.parsed_args.duration else "",
            f"-tc {InputManager.parsed_args.task_complexity}" if InputManager.parsed_args.task_complexity else "",
            f"-tp {InputManager.parsed_args.task_priority}" if InputManager.parsed_args.task_priority else "",
            f"-td {InputManager.parsed_args.task_deadline}" if InputManager.parsed_args.task_deadline else "",
            f"-cp {InputManager.parsed_args.car_processing_power}" if InputManager.parsed_args.car_processing_power else ""
        ]
        
        # Filter out any empty strings and join them with spaces
        text = " ".join(cmd for cmd in command if cmd)
        return f". python3 main.py {text}\n"

if __name__ == '__main__' : 
    InputManager.parse_arguments() 
    InputManager.load_config()
    InputManager.get_run_number()
    if InputManager.parsed_args.dry_run :
        InputManager.show_parameter()
        

    else :
        GenerateRuns.get_scenarios_generate_run_commands()
        GenerateRuns.runs_guide()
