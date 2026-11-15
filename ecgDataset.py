import torch
from torch.utils.data import Dataset


class ECGDataset(Dataset):
    def __init__(self, np_array):
        super().__init__()
        self.seq_array = torch.tensor(np_array, dtype=torch.float32)
    def __len__(self):
        return len(self.seq_array)
    def __getitem__(self, index):
        return self.seq_array[index]