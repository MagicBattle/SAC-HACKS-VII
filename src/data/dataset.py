import torch
from torch.utils.data import Dataset
import numpy as np

class ASLDataset(Dataset):
    """
    Dataset for static ASL signs (single frame, 63D vectors).
    """
    def __init__(self, data_path=None, labels_path=None):
        if data_path and labels_path:
            self.data = np.load(data_path)
            self.labels = np.load(labels_path)
        else:
            self.data = np.random.randn(1000, 63).astype(np.float32)
            self.labels = np.random.randint(0, 26, (1000,))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


class ASLDynamicDataset(Dataset):
    """
    Dataset for dynamic/motion ASL signs (sequences of frames).
    Each sample is a sequence of normalized landmark vectors (seq_len, 63).
    """
    def __init__(self, data_path=None, labels_path=None, seq_len=30):
        self.seq_len = seq_len
        if data_path and labels_path:
            # data shape: (num_samples, seq_len, 63)
            self.data = np.load(data_path, allow_pickle=True)
            self.labels = np.load(labels_path)
        else:
            self.data = np.random.randn(100, seq_len, 63).astype(np.float32)
            self.labels = np.random.randint(0, 2, (100,))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = self.data[idx]
        # Pad or truncate to fixed seq_len
        if len(seq) < self.seq_len:
            pad = np.zeros((self.seq_len - len(seq), 63), dtype=np.float32)
            seq = np.concatenate([seq, pad], axis=0)
        elif len(seq) > self.seq_len:
            seq = seq[:self.seq_len]
        x = torch.tensor(seq, dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
