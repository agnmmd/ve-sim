def print_color(string, color_code, argument=""):
        # Black (Gray): 90m
        # Red: 91m
        # Green: 92m
        # Yellow: 93m
        # Blue: 94m
        # Magenta: 95m
        # Cyan: 96m
        # White: 97m
        print(f"\033[{color_code}m{string} {argument}\033[0m")

def calculate_completion_time(current_time, car, task):
        waiting_time = car.get_remaining_time() + car.calculate_waiting_time()
        processing_time = car.calculate_processing_time(task)
        completion_time = waiting_time + processing_time

        print(f"Evaluating Car {car.id} for Task {task.id}:")
        print(f"  Current Time: {current_time}")
        print(f"  Waiting Time: {waiting_time}")
        print(f"  Processing Time: {processing_time}")
        print(f"  Relative Completion Time: {completion_time}")
        print(f"  Task Time of Arrival: {task.time_of_arrival}")
        print(f"  Task Deadline: {task.deadline}")
        print(f"  Estimated Task Completion Time: {current_time + completion_time}")

        return completion_time

def before_deadline(current_time, task, completion_time):
    return (current_time + completion_time) <= (task.time_of_arrival + task.deadline)