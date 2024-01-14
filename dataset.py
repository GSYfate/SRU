import numpy as np
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, max_len):
        self.max_len = max_len

    def __len__(self):
        return self.max_len - 3 


    def __getitem__(self, idx):
        input_seq = np.array([[idx + 1, idx + 2, idx + 3]], dtype=np.float32).reshape(3, 1)
        input_seq = torch.from_numpy(input_seq)

        label = np.array([[ idx + 2, idx + 3, idx + 4]], dtype=np.float32).reshape(3, 1)
        label = torch.from_numpy(label)

        return {
            'inputs': input_seq,
            'label': label
        }