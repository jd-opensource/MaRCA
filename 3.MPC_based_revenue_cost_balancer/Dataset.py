from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from typing import List, Dict, Tuple, Optional

class SystemDataset(Dataset):
    """Dataset class for system state and control data.
    
    This dataset handles:
    1. Loading and preprocessing of raw data
    2. String column encoding
    3. Min-max normalization of numerical columns
    4. Train/test split
    5. Lambda value preservation (no normalization)
    
    Args:
        root_dir (str): Root directory containing the data files
        args: Configuration arguments containing dataset parameters
        mode (str): Either 'train' or 'test' to specify the dataset split
    """
    
    def __init__(self, root_dir, args, mode = 'train'):
        if mode not in ['train', 'test']:
            raise ValueError(f"mode must be 'train' or 'test', got {mode}")
            
        self.root_dir = root_dir
        self.mode = mode
        self.args = args
        
        self._load_data()
        self._encode_string_columns()
        self._normalize_data()
        self._split_data()
        
    def _load_data(self) -> None:
        data_path = os.path.join(self.root_dir, self.args.data_file)
        self.raw_data = pd.read_csv(data_path)
        self.column_names = self.raw_data.columns.tolist()
        
    def _encode_string_columns(self) -> None:
        self.str_encoders: Dict[str, LabelEncoder] = {}
        for col in self.args.str_columns:
            if col not in self.raw_data.columns:
                raise ValueError(f"String column {col} not found in data")
            encoder = LabelEncoder()
            self.raw_data[col] = encoder.fit_transform(self.raw_data[col])
            self.str_encoders[col] = encoder
            
        encoder_path = os.path.join(self.root_dir, self.args.encoder_file)
        joblib.dump(self.str_encoders, encoder_path)
        
    def _normalize_data(self) -> None:
        self.data = self.raw_data.to_numpy().astype(np.float32)

        self.data[:, self.args.day_column] = self.data[:, self.args.day_column] % self.args.day_cycle 
        self.data[:, self.args.time_column] = self.data[:, self.args.time_column] % self.args.minute_cycle
        
        minmax_data = []
        for i, col in enumerate(self.column_names):
            minmax_data.append([np.max(self.data[:, i]), np.min(self.data[:, i])])
            if col in self.args.lambda_columns:
                continue  
            self.data[:, i] = self._minmax_normalize(self.data[:, i], minmax_data[-1])
            
        minmax_path = os.path.join(self.root_dir, self.args.minmax_data_file)
        np.save(minmax_path, np.array(minmax_data))
        
        processed_path = os.path.join(self.root_dir, self.args.processed_data_file)
        df_out = pd.DataFrame(self.data, columns=self.column_names)
        df_out.to_csv(processed_path, index=False)
        
    def _minmax_normalize(self, data: np.ndarray, minmax: List[float]) -> np.ndarray:
        return (data - minmax[1]) / (minmax[0] - minmax[1] + 1e-9)
        
    def _split_data(self) -> None:
        np.random.shuffle(self.data)
        split_idx = int(len(self.data) * self.args.train_ratio)
        
        if self.mode == 'train':
            data_split = self.data[:split_idx]
        else:
            data_split = self.data[split_idx:]
            
        self.inputs = data_split[:, :self.args.input_shape]
        self.labels = data_split[:, self.args.input_shape:self.args.input_shape + self.args.output_shape]

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.inputs[index], self.labels[index]