import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
import numpy as np
import diffraction_sim

class DiffractionDataset(Dataset):
    def __init__(self, sim_size=1024, target_size=256, epoch_size=1000):
        self.sim_size = sim_size
        self.target_size = target_size
        self.epoch_size = epoch_size

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        composite_bytes, mask_bytes_list = diffraction_sim.generate_sample(self.sim_size)

        # --- COMPOSITE IMAGE ---
        composite_np = np.frombuffer(composite_bytes, dtype=np.uint8).copy()
        composite_np = composite_np.reshape((1, self.sim_size, self.sim_size))
        
        # Keep background at 0.0 and bright spots near 1.0 
        # (Remember to REMOVE "inputs = 1.0 - inputs" from your training loop!)
        x = torch.from_numpy(composite_np.astype(np.float32) / 255.0)
        
        x = x.unsqueeze(0) 
        x = F.interpolate(x, size=(self.target_size, self.target_size), mode='area')
        x = x.squeeze(0)

        # --- MASKS ---
        mask_arrays = []
        for mask_bytes in mask_bytes_list:
            mask_np = np.frombuffer(mask_bytes, dtype=np.uint8).copy()
            mask_np = mask_np.reshape((1, self.sim_size, self.sim_size))
            mask_arrays.append(mask_np)

        # Pad with empty masks up to exactly 3 channels
        while len(mask_arrays) < 3:
            mask_arrays.append(np.zeros((1, self.sim_size, self.sim_size), dtype=np.uint8))

        masks_np = np.concatenate(mask_arrays, axis=0) # Shape is now guaranteed to be (3, sim_size, sim_size)
        y = torch.from_numpy(masks_np).float() / 255.0 
        
        y = y.unsqueeze(0)
        y = F.interpolate(y, size=(self.target_size, self.target_size), mode='area')
        y = y.squeeze(0)
        
        y = (y > 0.0).float() 
        y = F.max_pool2d(y, kernel_size=3, stride=1, padding=1)
        return x, y
