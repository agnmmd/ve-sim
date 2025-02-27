import numpy as np
import matplotlib.pyplot as plt
from pulp import LpMinimize, LpProblem, LpVariable, lpSum, LpBinary

# Parameters
np.random.seed(0)  # For reproducibility
numberOfProcessors = 11
numberOfTasks = 40

# Random processing times (equivalent to processingTime in MATLAB)
processingTime = np.array([10, 7, 2, 5, 3, 4, 7, 6, 4, 3, 1]).reshape(-1, 1) * np.random.rand(numberOfProcessors, numberOfTasks)

# Create the optimization problem
prob = LpProblem("Task_Scheduling", LpMinimize)

# Create decision variables: process[i][j] = 1 if task j is assigned to processor i
process = [[LpVariable(f'process_{i}_{j}', cat=LpBinary) for j in range(numberOfTasks)] for i in range(numberOfProcessors)]

# Makespan (objective function to minimize)
makespan = LpVariable("makespan", lowBound=0)

# Add objective function: minimize the makespan
prob += makespan

# Add constraints
# 1. Each task must be assigned to exactly one processor
for j in range(numberOfTasks):
    prob += lpSum([process[i][j] for i in range(numberOfProcessors)]) == 1

# 2. The makespan must be greater than or equal to the total processing time of each processor
for i in range(numberOfProcessors):
    prob += makespan >= lpSum([process[i][j] * processingTime[i][j] for j in range(numberOfTasks)])

# Solve the problem
prob.solve()

# Get the task assignment solution
processval = np.array([[process[i][j].varValue for j in range(numberOfTasks)] for i in range(numberOfProcessors)])

# Calculating the optimal schedule and task processing times
maxlen = int(np.max(np.sum(processval, axis=1)))  # Width of the schedule matrix
optimalSchedule = np.zeros((numberOfProcessors, maxlen), dtype=int)
ptime = np.zeros_like(optimalSchedule, dtype=float)

for i in range(numberOfProcessors):
    schedi = np.where(processval[i, :] == 1)[0]
    optimalSchedule[i, :len(schedi)] = schedi + 1  # +1 to match MATLAB 1-indexing
    ptime[i, :len(schedi)] = processingTime[i, schedi]

# Plotting

# Plotting the stacked bar chart
fig, ax = plt.subplots()

# Create the stacked bar plot
for i in range(ptime.shape[1]):  # Iterate through tasks in each processor
    ax.bar(np.arange(1, numberOfProcessors + 1), ptime[:, i], bottom=np.sum(ptime[:, :i], axis=1))

# Adding labels and title
ax.set_xlabel('Processor Number')
ax.set_ylabel('Processing Time')
ax.set_title('Task Assignments to Processors')

# Annotating the bars with task numbers
for i in range(numberOfProcessors):
    for j in range(maxlen):
        if optimalSchedule[i, j] > 0:  # Only annotate assigned tasks
            ax.text(i + 1, np.sum(ptime[i, :j+1]), str(optimalSchedule[i, j]),
                    va='top', ha='center', fontsize=10, color='w')

# Show the plot
plt.xticks(np.arange(1, numberOfProcessors + 1))  # Set x-ticks to processor numbers
plt.show()


