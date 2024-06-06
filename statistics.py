import csv
import os

class Statistics:
    @staticmethod
    def save_task_stats(task):
        filename = 'task_statistics.csv'
        file_exists = os.path.isfile(filename)

        with open(filename, 'a', newline='') as csvfile:
            fieldnames = ['Task ID', 'Source Car ID', 'Time of Arrival', 'Deadline', 'Priority', 'Complexity', 'Processing Start', 'Processing End']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerow({
                'Task ID': task.id,
                'Source Car ID': task.source_car.id,
                'Time of Arrival': task.time_of_arrival,
                'Deadline': task.deadline,
                'Priority': task.priority,
                'Complexity': task.complexity,
                'Processing Start': task.processing_start,
                'Processing End': task.processing_end,
            })

    @staticmethod
    def save_car_stats(car):
        filename = 'car_statistics.csv'
        file_exists = os.path.isfile(filename)

        with open(filename, 'a', newline='') as csvfile:
            fieldnames = ['Car ID', 'Generated Tasks', 'Processed Tasks', 'Successful Tasks', 'Total Processing Time', 'Lifetime']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerow({
                'Car ID': car.id,
                'Generated Tasks': len(car.generated_tasks),
                'Processed Tasks': car.processed_tasks_count,
                'Successful Tasks': car.successful_tasks,
                'Total Processing Time': car.total_processing_time,
                'Lifetime': Sim.env.now - car.time_of_arrival
            })
