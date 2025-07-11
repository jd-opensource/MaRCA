
# Model parameters
MODEL = 'AWRQ'
# choice = ['AWRQ', 'DQN', 'DoubleDQN', 'DRQN', 'EMDQN', 'REMDQN', 'QMIX','VDN','WEIGHTED_QMIX']
ABLATION_FLAG=False
ABLATION_MODEL='AWRQ_w/o_AW'
# choice = ['AWRQ_w/o_AW','AWRQ_w/o_SMC','AWRQ_w/o_VGCA']

# Training parameters
EPOCHS = 50
BATCH_SIZE = 256
TARGET_UPDATE_FREQ = 100

# Algo common parameters
STATE_SPACE = 4
RE_ACTION_SPACE = [4, 5, 5, 2]
PRE_ACTION_SPACE = [5]
SINGLE_AGENT_ACTION_DIM = sum(RE_ACTION_SPACE) + sum(PRE_ACTION_SPACE)
HIDDEN_SIZE = 128
ENSEMBLE_SIZE = 5
EMBED_DIM = 128
GAMMA = 0.99

# Optimizer parameters
OPTIMIZER = 'Adam'
# choice = ['Adam','SGD','RMSprop']
LEARNING_RATE = 1e-2
CLIP_GRADIENT = True
GRADIENT_CLIP_NORM = 1.0

# Network parameters
ACTIVATION_FUNCTION = 'relu'
# choice = ['relu', 'tanh', 'sigmoid', 'leaky_relu']
USE_BATCH_NORM = False
DROPOUT_RATE = 0.2
WEIGHT_INITIALIZER = 'glorot_uniform'

#Loss parameters
LOSS_FUNCTION = 'mse'
# choice = ['mse', 'mae', 'huber']


