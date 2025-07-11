from mpc_solver import MPC_Solver
from utils import get_args
import time
import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main(root_dir, state, args): 
    model_path = os.path.join(root_dir, args.saved_modules_dir, args.model_path)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    mpc_solver = MPC_Solver(
                    args,
                    d_C=args.d_C,
                    d_s=args.d_s,
                    d_a=args.d_a, 
                    loc=args.loc,
                    Cm=args.Cm,
                    Cthre=args.Cthre,
                    lmd_lbound=args.lmd_lb,
                    lmd_ubound=args.lmd_ub, 
                    pred_h=args.pred_h, 
                    alpha=args.alpha,
                    beta=args.beta,
                    model_path=model_path,
                    root_dir=root_dir)
    
    utl_rate = 0.
    overutl_rate = 0.
    for i in range(args.sim_h):
        mpc_solver.x = state
        logger.info(f'Current time: {i}min')
        start_time = time.time()
        lmbd = mpc_solver.solve_sample(state)
        end_time = time.time()
        logger.info(f'Solving time per step: {end_time - start_time:.2f}s')

        out = mpc_solver.model_sml(state, lmbd).reshape(-1, )
        state_next = state.copy()
        state_next[args.loc:args.loc+args.d_C] = out[:args.d_C]

        time_increment = 1.0 / (args.minute_cycle - 1)  
        state_next[args.time_column] = (state[args.time_column] + time_increment) % 1.0
        state = state_next

        cpu_ranges = np.array([
            [args.cpu1_max, args.cpu1_min],
            [args.cpu2_max, args.cpu2_min]
        ])
        Ct = out[:args.d_C] * (cpu_ranges[:, 0] - cpu_ranges[:, 1]) + cpu_ranges[:, 1]
        logger.info(f'CPU usage: {Ct}')
        utl_rate += Ct[0] / args.Cm if Ct[0] <= args.Cm else 1.
        overutl_rate += (Ct[0] - args.Cm) / args.Cm if Ct[0] > args.Cm else 0.

    return utl_rate / args.sim_h, overutl_rate / args.sim_h

if __name__ == '__main__':
    args = get_args()
    
    if args.root_dir is None:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), './'))
    else:
        root_dir = os.path.abspath(args.root_dir)
    
    out_data_path = os.path.join(root_dir, args.processed_data_file)
    if not os.path.exists(out_data_path):
        raise FileNotFoundError(f"Data file not found at: {out_data_path}")
    
    df = pd.read_csv(out_data_path)
    
    for col in args.lambda_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataset")
    
    idxs = []
    i = 0
    max_index = len(df) - 1
    
    while i < max_index:
        lambda_values = df.iloc[i][args.lambda_columns].values
        if np.sum(lambda_values) != 0: 
            idxs.append(i)
        i += 1
    
    if not idxs:
        raise ValueError("No valid initial states found")
    
    if len(idxs) > 10:
        idxs = idxs[:10]
            
    for exp_idx in idxs:
        logger.info(f"\nProcessing experiment with initial state at index {exp_idx}")
        init_state = df.iloc[exp_idx].to_numpy()[1:]
        init_state = init_state[:args.d_s]
        try:
            utl_rate, overutl_rate = main(root_dir, init_state, args)
            logger.info(f'Experiment {exp_idx}: Utilization rate = {utl_rate:.4f}, Overutilization rate = {overutl_rate:.4f}')
        except Exception as e:
            logger.error(f"Error processing experiment {exp_idx}: {str(e)}", exc_info=True)

    