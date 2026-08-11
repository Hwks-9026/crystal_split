import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
import diffraction_sim

class DiffractionDataset(Dataset):
    """
    PyTorch Dataset Class for using simulated diffraction patterns.
    Dynamically requests byte streams from diffraction_sim library,
    formats them into standard tensors, applies downsampling.
    """
    def __init__(self, sim_size=1024, target_size=256, epoch_size=1000):
        self.sim_size = sim_size
        self.target_size = target_size
        self.epoch_size = epoch_size

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        """
        Fetches a raw byte stream, formats the composite input, and processes the target masks.
        Applies area interpolation for downsampling and max pooling to dilate target peaks.
        """
        composite_bytes, mask_bytes_list, metadata = diffraction_sim.generate_sample(self.sim_size)

        composite_np = np.frombuffer(composite_bytes, dtype=np.uint8).copy()
        composite_np = composite_np.reshape((1, 1, self.sim_size, self.sim_size))
        
        x = torch.from_numpy(composite_np).float() / 255.0
        x = F.interpolate(x, size=(self.target_size, self.target_size), mode='area').squeeze(0)

        masks_np = np.zeros((1, 2, self.sim_size, self.sim_size), dtype=np.uint8)
        for i, mask_bytes in enumerate(mask_bytes_list[:2]):
            masks_np[0, i] = np.frombuffer(mask_bytes, dtype=np.uint8).copy().reshape(self.sim_size, self.sim_size)

        y = torch.from_numpy(masks_np).float() / 255.0
        y = F.interpolate(y, size=(self.target_size, self.target_size), mode='area').squeeze(0)
        
        y = (y > 0.0).float()
        y = F.max_pool2d(y, kernel_size=3, stride=1, padding=1)
        
        return x, y, metadata
