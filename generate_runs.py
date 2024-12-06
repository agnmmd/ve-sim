from input_manager import InputManager

if __name__ == '__main__' :
    command_line_args = InputManager.parse_command_line_arguments()
    all_runs = InputManager.compile_all_runs_from_configfile(command_line_args.configfile)

    for (run, sim_config),parameters in all_runs.items():
        # python3 main.py -r 2 -cf config_file.ini -c Simulation1
        print(f". python3 main.py -r {run} -cf {command_line_args.configfile} -c {sim_config}")
