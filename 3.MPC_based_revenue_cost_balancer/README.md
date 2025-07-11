# MPC-based Revenue-Cost Balancer

The MPC-based Revenue-Cost Balancer uses a neural network to predict system states and optimizes lambda through MPC to maintain CPU usage within specified thresholds while maximizing resource utilization.

## System Overview

The system consists of two main components:
1. **System Model**: A feed-forward neural network that predicts states based on current system state and control inputs (lambda values)
2. **MPC Controller**: An optimization-based controller that computes optimal lambda values to maintain CPU usage within target thresholds


## Project Structure

```
dynamic_compute_allocation/
├── marca/
│   ├── main_mpc.py          # Main MPC controller and simulation script
│   ├── mpc_solver.py        # MPC solver with multiple optimization methods
│   ├── system_model.py      # Neural network system model implementation
│   ├── Dataset.py           # Dataset handling and preprocessing
│   ├── train.py            # Neural network training script
│   ├── utils.py            # Configuration and utility functions
│   └── sample_data.csv     # Raw dataset
``` 

## Implementation Details

### Dataset Handling (Dataset.py)
- String column encoding
- Train/test split
- Min-max scaling for numerical features

### System Model (system_model.py)
- Feed-forward neural network with configurable architecture
- Input: System state + Control inputs
- Output: System state
- Configurable activation functions and hidden layer sizes

### MPC Solver (mpc_solver.py)
- Multiple optimization methods:
  - Sample-based optimization
  - SLSQP (Sequential Least Squares Programming)
  - BFGS (Broyden-Fletcher-Goldfarb-Shanno)
  - Gradient Descent
- Time-aware control with configurable time cycles

## Usage

### Training the Model

```bash
python train.py --root-dir /path/to/data \
                --saved-modules-dir saved_modules \
                --model-path model_params_100.pkl \
                --data-file sample_data.csv \
                --processed-data-file processed_data.csv \
                --minmax-data-file processed_data.npy \
                --train-ratio 0.8 \
                --num-epochs 1000 \
                --train-batch-size 1024 \
                --test-batch-size 1024 \
                --hidden-units 256 256 256 \
                --activation relu \
                --output-activation sigmoid
```

### Running the MPC Controller

```bash
python main_mpc.py --root-dir /path/to/data \
                   --saved-modules-dir saved_modules \
                   --model-path model_params_100.pkl \
                   --processed-data-file processed_data.csv \
                   --pred-h 15 \
                   --sim-h 30 \
                   --Cm 46.0 \
                   --Cthre 44.0 \
                   --cpu1-max 100 \
                   --cpu1-min 0 \
                   --cpu2-max 100 \
                   --cpu2-min 0
```

## Key Parameters

### System Model Parameters
- `--input-shape`: Input dimension (default: 30)
- `--output-shape`: Output dimension (default: 2)
- `--hidden-units`: Hidden layer sizes (default: [256, 256, 256])
- `--activation`: Hidden layer activation (default: 'relu')
- `--output-activation`: Output layer activation (default: 'sigmoid')

### MPC Parameters
- `--d-C`: CPU usage metrics dimension (default: 2)
- `--d-s`: System state dimension (default: 28)
- `--d-a`: Number of lambda values (default: 2)
- `--loc`: CPU usage metrics starting index (default: 7)
- `--Cm`: Target CPU usage threshold (default: 46.0)
- `--Cthre`: Critical CPU usage threshold (default: 44.0)
- `--pred-h`: Prediction horizon (default: 15)
- `--sim-h`: Simulation horizon (default: 30)
- `--alpha`: State weight coefficient (default: 0.4)
- `--beta`: Reference weight coefficient (default: 4.0)

### Time Parameters
- `--day-cycle`: Day of week cycle length (default: 7)
- `--minute-cycle`: Time slot cycle length (default: 20)
- `--day-column`: Day of week column index (default: 2)
- `--time-column`: Time slot column index (default: 3)

### CPU Thresholds
- `--cpu1-max`: Maximum CPU usage for CPU 1 (default: 100)
- `--cpu1-min`: Minimum CPU usage for CPU 1 (default: 0)
- `--cpu2-max`: Maximum CPU usage for CPU 2 (default: 100)
- `--cpu2-min`: Minimum CPU usage for CPU 2 (default: 0)

## Data Format

The input data should be in CSV format with the following structure:
- String columns: `idc`, `group`, `dt` (for identification)
- Time columns: Day of week and time slot
- System state columns: Various metrics including CPU usage
- Control input columns: `lambda_value1`, `lambda_value2`

## Output Metrics

The MPC controller provides two key performance metrics:
1. **Utilization Rate**: CPU usage relative to the target
2. **Overutilization Rate**: CPU usage exceeding the target

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- Pandas
- SciPy
- scikit-learn

Install dependencies:
```bash
pip install torch numpy pandas scipy scikit-learn
```
