import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from glob import glob
import torch

class CustomDataset(Dataset):
    def __init__(self, dataset_folder, mode, transform=None):
        self.dataset_folder = dataset_folder
        self.mode = mode
        self.transform = transform

        self.images = sorted(glob(f"{dataset_folder}{mode}/**/*leftImg8bit.png", recursive=True))
        self.masks = sorted(glob(f"{dataset_folder}{mode}/**/*labelIds.png", recursive=True))

    def __len__(self):
        return len(self.masks)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        image = np.array(image, dtype=np.uint8)

        mask = Image.open(self.masks[idx]).convert("L")
        mask = np.array(mask, dtype=np.uint8)

        if self.transform:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()

        return image, mask
        
