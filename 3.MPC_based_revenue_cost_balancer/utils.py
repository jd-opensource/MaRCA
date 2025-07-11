import argparse

def get_args():
    parser = argparse.ArgumentParser(description='MPC controller parameters and model training settings')
    
    # Directory and path settings
    parser.add_argument('--root-dir', type=str, default=None,
                       help='Root directory for data and model files. If None, uses current directory')
    parser.add_argument('--saved-modules-dir', type=str, default='saved_modules',
                       help='Directory name for saving model checkpoints')
    parser.add_argument('--model-path', type=str, default='model_params_100.pkl',
                       help='Name of the model checkpoint file to load')
    parser.add_argument('--data-file', type=str, default='sample_data.csv',
                       help='Name of the input data file')
    parser.add_argument('--processed-data-file', type=str, default='processed_data.csv',
                       help='Name of the processed data file (csv)')
    parser.add_argument('--minmax-data-file', type=str, default='processed_data.npy',
                       help='Name of the processed data file (npy)')
    parser.add_argument('--encoder-file', type=str, default='str_encoders_dict.joblib',
                       help='Name of the string encoder dictionary file')
    
    # Dataset parameters
    parser.add_argument('--str-columns', type=str, nargs='+', default=['idc', 'group', 'dt'],
                       help='List of string columns that need encoding')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Ratio of data to use for training (0-1)')
    parser.add_argument('--lambda-columns', type=str, nargs='+', default=['lambda_value1', 'lambda_value2'],
                       help='List of lambda columns that should not be normalized')
    parser.add_argument('--day-cycle', type=int, default=7,
                       help='Cycle length for day of week (e.g., 7 for weekly cycle)')
    parser.add_argument('--minute-cycle', type=int, default=20,
                       help='Cycle length for time slots (e.g., 20 for 20-minute intervals)')
    parser.add_argument('--day-column', type=int, default=2,
                       help='Column index for day of week in the data')
    parser.add_argument('--time-column', type=int, default=3,
                       help='Column index for time slot in the data')
    
    # System model training parameters
    parser.add_argument('--input-shape', default=30, type=int, 
                       help='Input dimension of model')
    parser.add_argument('--output-shape', default=2, type=int, 
                       help='Output dimension of model')
    parser.add_argument('--train-batch-size', default=1024, type=int, 
                       help='Batch size for model training')
    parser.add_argument('--test-batch-size', default=1024, type=int,
                       help='Batch size for model validation/testing')
    parser.add_argument('--num-epochs', default=1000, type=int, 
                       help='Total number of training epochs')
    parser.add_argument('--save-rate', default=10, type=int, 
                       help='Frequency of model checkpoint saving (in epochs)')
    parser.add_argument('--hidden-units', default=[256, 256, 256], type=int, nargs='+',
                       help='List of hidden layer sizes')
    parser.add_argument('--activation', default='relu', type=str,
                       help='Activation function for hidden layers (relu/sigmoid/tanh/leaky_relu)')
    parser.add_argument('--output-activation', default='sigmoid', type=str,
                       help='Activation function for output layer (relu/sigmoid/tanh/leaky_relu)')

    # MPC solver parameters
    parser.add_argument('--d-C', default=2, type=int, 
                       help='Dimension of CPU usage metrics')
    parser.add_argument('--d-s', default=28, type=int, 
                       help='Dimension of system state')
    parser.add_argument('--d-a', default=2, type=int, 
                       help='Number of lambda values')
    parser.add_argument('--loc', default=7, type=int, 
                       help='Starting index of CPU usage metrics')
    parser.add_argument('--Cm', default=46., type=float, 
                       help='Target CPU usage threshold (budget limit)')
    parser.add_argument('--Cthre', default=44., type=float, 
                       help='Critical CPU usage threshold (safety limit)')
    parser.add_argument('--lmd_lb', default=0., type=float, 
                       help='Lower bound for lambda values')
    parser.add_argument('--lmd_ub', default=20., type=float, 
                       help='Upper bound for lambda values')
    parser.add_argument('--pred-h', default=15, type=int, 
                       help='Prediction horizon')
    parser.add_argument('--sim-h', default=30, type=int,
                       help='Simulation horizon')
    parser.add_argument('--alpha', default=0.4, type=float, 
                       help='Weight coefficient for state')
    parser.add_argument('--beta', default=4, type=float, 
                       help='Weight coefficient for reference')
    
    # CPU usage thresholds
    parser.add_argument('--cpu1-max', default=100, type=float,
                       help='Maximum CPU usage for CPU 1')
    parser.add_argument('--cpu1-min', default=0, type=float,
                       help='Minimum CPU usage for CPU 1')
    parser.add_argument('--cpu2-max', default=100, type=float,
                       help='Maximum CPU usage for CPU 2')
    parser.add_argument('--cpu2-min', default=0, type=float,
                       help='Minimum CPU usage for CPU 2')

    return parser.parse_args()
