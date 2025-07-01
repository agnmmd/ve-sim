from sim import Sim
from car import Car
from task import Task
from scheduler import Scheduler
from traci_manager import TraciManager
from input_manager import InputManager
from policy import *
import traci
from traci_annotation import TraciAnnotation
# from policy_factory import get_policy
from rl import *
from class_factory import load_class
from stats import Statistics

def setup_traci():
    traci_mgr = TraciManager()
    # roi_min_x = Sim.get_parameter('roi_min_x')
    # roi_min_y = Sim.get_parameter('roi_min_y')
    # roi_max_x = Sim.get_parameter('roi_max_x')
    # roi_max_y = Sim.get_parameter('roi_max_y')
    # traci_mgr.set_rois([(roi_min_x, roi_min_y, roi_max_x, roi_max_y)])
    sumo_binary = Sim.get_parameter('sumo_binary')
    sumo_cfg = Sim.get_parameter('sumo_cfg')
    sumo_step_length = Sim.get_parameter('step_length')
    sumo_cmd = [sumo_binary, "-c", sumo_cfg, "--quit-on-end", "--step-length", str(sumo_step_length)]#, "--start"])
    # --start # Start the simulation immediately after loading (no need to press the start button)
    # --quit-on-end # Quit the simulation gui in the end automatically once the simulation is finished
    # --step-length TIME # Defines the step duration in seconds

    traci.start(sumo_cmd)
    Sim.get_env().process(traci_mgr.execute_one_time_step())

    ##################################################
    # drawer = TraciAnnotation()

    # # Add a rectangle using bottom-left and top-right coordinates
    # # 6430,7180-6562,7257
    # bottom_left = (6430,7180)
    # top_right = (6562,7257)
    # drawer.add_rectangle('rectangle1', bottom_left, top_right)

    # # # Add a circle
    # drawer.add_circle('circle1', center=(6430,7180), radius=1500)

    # # # Draw all shapes in the SUMO simulation
    # drawer.draw_shapes()
    ##################################################
    
    return traci_mgr

def run_sim():
    rl_env = load_class(Sim.get_parameter("rl_environment"))
    agent = load_class(Sim.get_parameter("rl_agent"), rl_env=rl_env)
    policy_name = Sim.get_parameter("policy")

    if  policy_name == "DQNPolicy":
        policy = load_class(Sim.get_parameter("policy"), gymenv=rl_env, agent=agent)
    else:
        policy = load_class(Sim.get_parameter("policy"))

    traci_mgr = setup_traci()
    scheduler = Scheduler(traci_mgr, policy)

    # Start Scheduling
    Sim.get_env().process(scheduler.schedule_tasks())
    end = Sim.get_parameter('end')
    Sim.get_env().run(until=end+1)

    # Print statistics for static cars that haven't been removed by dwell time
    for car in scheduler.static_cars:
        car.finish()

def train():
    
    rl_env = load_class(Sim.get_parameter("rl_environment"))
    agent = load_class(Sim.get_parameter("rl_agent"), rl_env=rl_env)

    for episode in range(InputManager.command_line_args.episodes):
        policy_name = Sim.get_parameter("policy")

        if policy_name == "DQNPolicy":
            policy = load_class(Sim.get_parameter("policy"), gymenv=rl_env, agent=agent)
        else:
            raise NameError(f"Policy is not defined correctly")
        
        Sim.reset(episode)
        traci_mgr = setup_traci()
        scheduler = Scheduler(traci_mgr, policy)
        rl_env.reset()

        # Start Scheduling
        Sim.get_env().process(scheduler.schedule_tasks())
        end = Sim.get_parameter('end')
        Sim.get_env().run(until=end+1)

        # NOTE: Episode ends here

        # Print statistics for static cÖars that haven't been removed by dwell time
        for car in scheduler.static_cars:
            car.finish()

        print(f"Episode {episode}: Total Reward: {policy.get_episode_reward()}")
        Statistics.save_episode_stats(policy.get_episode_reward(), policy.get_episode_best_selection_ratio(), policy.get_episode_action_count())

        # Decay epsilon after each episode
        agent.decay_epsilon_exp(episode)

        # Periodically update the target network
        if episode % agent.target_update_freq == 0:
            agent.target_network.load_state_dict(agent.q_network.state_dict())

    agent.save_model("./training-dqn.pth")

if __name__ == "__main__":

    if not InputManager.dry_run():
        if InputManager.command_line_args.train:
            train()
        else:
            run_sim()



# TODO: Optional: In the Scheduler add a list (self.task_queue) that holds all the tasks; Also, the tasks can be subscribed automatically to it
# TODO: Optional: Use the schedule as a log for the schedule mapping (car, time) to task.
# NOTE: Due to the order of the iteration there is an order of tasks. Car 0 might be favored.
# NOTE: A task should be at one entity, there should not be multiple copies of a task;
# NOTE: Initiation of the tasks currently is happening in the Traci; Alternative: move to __init__ of Car?
# NOTE: A record of the the car that processes a task is not stored in the Task object for Statistic purposes. Maybe integrate.
# NOTE: Currently the Stats module exists separately, maybe create the Stats module inside Sim in the constructor...
# NOTE: Sim configuration is updated with e staticmethod. Maybe generate a Sim() object with all necessary config parameters in the constuctor.

