# Revenue Simulation

A resource allocation simulation system for optimizing computing power distribution between states and actions.

# Features

- Resource allocation algorithm with greedy score-based prioritization
- CSV data processing for states and actions
- Expert model scoring system
- Parallel prediction result processing
- Progress tracking with tqdm integration

# Usage
1. Prepare input files:
    - Expert model score data (marca-expert-data.csv)
    - Experiment model allocate result data (experiment-data.csv)
2. Run main simulation:
    - python main.py

# Configuration
## Key configuration files:
- Action Definitions:
```bash
a1,a2,a3,a4,a5,an,num
```
- State Definitions:
```bash
uv,w,g,i,sn,num
```
# Key Functions
allocate_actions()
Implements core allocation logic:
```bash
def allocate_actions(state_dict, action_dict, value_list):
    # Greedy allocation based on sorted scores
    # Maintains remaining resources in n_si_remaining/q_aj_remaining
    # Returns assignment tuples and final counts
```
