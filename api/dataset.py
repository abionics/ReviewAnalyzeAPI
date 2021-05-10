import numpy as np
import torch
from torch.utils.data import Dataset


class ReviewsDataset(Dataset):
    def __init__(self, encoded: list):
        self.__encoded = encoded

    def __len__(self):
        return len(self.__encoded)

    def __getitem__(self, idx: int):
        tensor = torch.from_numpy(self.__encoded[idx][0].astype(np.int32))
        return tensor, self.__encoded[idx][1]
