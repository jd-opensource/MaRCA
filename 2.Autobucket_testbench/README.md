# Autobucket Testbench Framework

Implementation of Multi-gate Mixture-of-Experts architecture with Deep & Cross Network for multi-task learning scenarios.

## Key Features <a name="key-features"></a>
### Core Components
- Hybrid architecture combining DCN and MMoE
- Dual-task prediction with shared feature processing
- Automated data preprocessing pipeline
- Flexible expert network configuration

## Configuration
### Training Configuration:
- EPOCHS: Number of training epochs.
- BATCH_SIZE: The number of samples per training batch.

### Algorithm Parameters:
- STATE_SPACE: Dimensionality of the state space.
- RE_ACTION_SPACE: List of dimensions for retrieval action space.

### Optimizer Configuration:
- LEARNING_RATE: The learning rate used by the optimizer.

## Training Process
1. Prepare Data:
The training data should be in a CSV file. Each row in the CSV represents a state, action, and label. The load_data function in main.py will handle the loading and processing of this data.

2. Modify config.py:
In config.py, modify the parameters to suit your experiment. 

3. Run main.py:
After adjusting the configuration, you can run the training with the following command:
python main.py

4. Monitor Training:
Training progress will be printed during each epoch, including loss and any other relevant metrics.
