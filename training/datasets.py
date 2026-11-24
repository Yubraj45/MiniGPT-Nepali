
import numpy as np
import torch
from torch.utils.data import Dataset


class GPTDataset(Dataset):

    def __init__(self, file_path, block_size=256):

        # Load token IDs
        data = np.load(file_path)

        # Convert token IDs to PyTorch long integers
        self.data = torch.from_numpy(
            data.astype(np.int64)
        )

        self.block_size = block_size

    def __len__(self):

        # Number of complete input-target chunks
        return (len(self.data) - 1) // self.block_size

    def __getitem__(self, index):

        # Starting position of this chunk
        start = index * self.block_size

        # Input sequence
        x = self.data[
            start : start + self.block_size
        ]

        # Target sequence, shifted by one position
        y = self.data[
            start + 1 : start + self.block_size + 1
        ]

        return x, y
