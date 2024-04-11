import numpy as np
import os
import sys
import pickle
import traceback
from torch.utils.data import Dataset

def rotate_points(points, angel):
    rotated = np.zeros(points.shape, dtype=points.dtype)
    rotated[...,0] = np.cos(np.deg2rad(angel)) * points[...,0] - np.sin(np.deg2rad(angel)) * points[...,1]
    rotated[...,1] = np.sin(np.deg2rad(angel)) * points[...,0] + np.cos(np.deg2rad(angel)) * points[...,1]
    return rotated

class CachedDataset(Dataset):
    def __init__(self, dataset, cache_folder=None):
        super().__init__()
        self.dataset = dataset
        self.cache = {}
        self.cache_folder = cache_folder

        if self.cache_folder is not None:
            os.makedirs(self.cache_folder, exist_ok=True)
            for filename in os.listdir(self.cache_folder):
                try:
                    idx = int(filename)
                    with open(os.path.join(self.cache_folder, filename), 'rb') as f:
                        data = pickle.load(f)
                    self.cache[idx] = data
                except Exception as e:
                    
                    traceback.print_exc()
                    print(f'filename: {filename}')
                    continue

    def __getitem__(self, idx):
        if idx not in self.cache:
            self.cache[idx] = self.dataset[idx]
            if self.cache_folder is not None:
                with open(os.path.join(self.cache_folder, str(idx)), 'wb') as f:
                    pickle.dump(self.cache[idx], f)

        return self.cache[idx]

    def __len__(self):
        return len(self.dataset)
