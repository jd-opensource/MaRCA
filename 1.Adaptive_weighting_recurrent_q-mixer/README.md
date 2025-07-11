# Multi-Agent Reinforcement Learning Models

This project contains multiple reinforcement learning models such as AWRQ, DQN, DoubleDQN, DRQN, EMDQN, REMDQN, QMIX, VDN, WEIGHTED_QMIX. The models are implemented with TensorFlow and use different configurations for training, optimization, and network architecture.

## Requirements

Make sure you have the Python libraries in requirement.txt

## Project Structure
- config.py: This file contains all configurable parameters, including model type, training parameters, optimizer settings, and other network-specific settings.

- model.py: This file contains the implementation of different models like AWRQ, DQN, DRQN, and QMIX, among others. You can modify the model architecture here if needed.

- main.py: This is the entry point for running the training. You can modify the parameters in config.py and then execute main.py to train the selected model.

- Data: The training data is expected to be in CSV format. The load_data function in main.py reads the data, processes the state, actions, and rewards for training.

- Evaluation：This repository contains a Python script for evaluating machine learning model predictions. It loads prediction results, calculates performance metrics, and generates reports in the form of tab-separated values (TSV) files.
## Configuration

### Model Configuration:
- MODEL: Specifies which model to train. Choose from a list of available models (AWRQ, DQN, DoubleDQN, DRQN, EMDQN, REMDQN, QMIX, VDN, WEIGHTED_QMIX).
- ABLATION_FLAG: If True, uses the ablation version of the selected model.
- ABLATION_MODEL: If using an ablation model, this will specify which variant of AWRQ to use.

### Training Configuration:
- EPOCHS: Number of training epochs.
- BATCH_SIZE: The number of samples per training batch.
- TARGET_UPDATE_FREQ: Frequency at which target networks are updated.

### Algorithm Parameters:
- STATE_SPACE: Dimensionality of the state space.
- RE_ACTION_SPACE: List of dimensions for retrieval action space.
- PRE_ACTION_SPACE: List of dimensions for ranking action space.
- SINGLE_AGENT_ACTION_DIM: Total action dimension for a single agent.
- HIDDEN_SIZE: The size of the hidden layers in the networks.
- ENSEMBLE_SIZE: Number of ensemble models used for computation.
- EMBED_DIM: Embedding dimension for the hypernetwork.
- GAMMA: Discount factor for reward calculation.
### Optimizer Configuration:
- OPTIMIZER: The optimizer to use for training. Can be 'Adam', 'SGD', 'RMSprop', etc.
- LEARNING_RATE: The learning rate used by the optimizer.
- CLIP_GRADIENT: Whether to clip gradients to avoid exploding gradients.
- GRADIENT_CLIP_NORM: The threshold for gradient clipping.

### Network Configuration:
- ACTIVATION_FUNCTION: The activation function used in the network layers.
- USE_BATCH_NORM: Whether to apply batch normalization in the network.
- DROPOUT_RATE: Dropout rate used for regularization.
- WEIGHT_INITIALIZER: Method used to initialize the weights in the network layers.
### Loss Function Configuration:
- LOSS_FUNCTION: The loss function used for training. Can be 'mse' (Mean Squared Error), 'mae' (Mean Absolute Error), or 'huber' (Huber loss).


## Training Process
1. Prepare Data:
The training data should be in a CSV file. Each row in the CSV represents a state, action, and reward. The load_data function in main.py will handle the loading and processing of this data.

2. Modify config.py:
In config.py, modify the parameters to suit your experiment. Set the MODEL to one of the available models, adjust the optimizer, learning rate, etc. You can also toggle the use of ablation models.

3. Run main.py:
After adjusting the configuration, you can run the training with the following command:
python main.py
4. Monitor Training:
Training progress will be printed during each epoch, including loss and any other relevant metrics.

## Available Models
- AWRQ: A model based on DRQN with additional components like hypernet for weighting.

- DQN: A simple deep Q-network using a fully connected network architecture.

- DoubleDQN: A variant of DQN that mitigates the overestimation bias by using two networks.

- DRQN: A recurrent deep Q-network using GRU cells for sequence data.

- EMDQN: An extension of DQN where the ensemble Q-values are averaged.

- REMDQN: Uses a weighted average of Q-values from multiple heads in the ensemble.

- QMIX: A model that mixes Q-values from multiple agents.

- VDN: A variant of QMIX where the Q-values are simply summed.

- WEIGHTED_QMIX: A weighted version of QMIX.

### Model-specific Parameters
- Make sure to change MODEL to the correct model name (e.g., DQN, AWRQ, DoubleDQN, etc.). Also, if you're using any ablation models (e.g., AWRQ_w/o_AW), set ABLATION_FLAG and ABLATION_MODEL accordingly.

## Conclusion
This repository allows you to experiment with multiple RL models by modifying the configuration file and running main.py. The modular design makes it easy to switch between models, adjust training parameters, and modify optimizer settings.