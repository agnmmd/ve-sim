# VE-SIM: RL-Based Task Scheduling for Vehicular Edge Computing

<img src="figures/components.png" alt="Simulation Framework Components" width="450" />

A simulation framework for Reinforcement Learning (RL) based task scheduling in vehicular networks, integrating realistic traffic simulation with deep learning capabilities.

## Key Features

- **Integrated Simulation**: Combines `SimPy` (discrete-event) with `SUMO` (traffic)
- **RL-Ready Framework**: PyTorch-based Agent and Environment
- **Realistic Modeling**: `Car` mobility + `Task` workloads with deadlines
- **Flexible Scheduling**: `Scheduler` supports heuristic and RL policies

## Framework Components

### 1. Vehicles (`Car` class)
- Mobile compute units with:
  - Processing power, Task generation, Dynamic mobility via SUMO (`TraCI`)
- Handles task execution

### 2. Tasks (`Task` class)
- Computational workloads with:
  - Complexity, Deadline, Priority

### 3. Scheduler (`Scheduler` class)
- Core decision-making component:
  - Maintains system state (`cars`, `tasks`)
  - Implements policy matching
  - Supports:
    - Heuristics (EDF, priority-based)
    - RL policies

## RL Integration

| Component              | Class               | Functionality                         |
|------------------------|---------------------|---------------------------------------|
| **RL Environment**     | `TaskSchedulingEnv` | Gymnasium interface for state/actions |
| **DQN Agent**          | `DQNAgent`          | Learns scheduling policy              |
| **Neural Network**     | `DQN`               | Policy approximation                  |
| **Experience Replay**  | `ReplayBuffer`      | Training data storage                 |

## Dependencies

- SUMO (1.23.1)
- Python (3.11.10)
- for Python libraries see 'requirements.txt'

## Getting Started

```bash
python3 main.py -cf config.ini -c RL-training --episodes 2000  --train
