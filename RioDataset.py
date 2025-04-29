import os

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class RioDataset(Dataset):

    def __init__(self, tiff_dir, geojson_dir, transform=None):
        self.tiff_files = [os.path.join(tiff_dir, f) for f in os.listdir(tiff_dir) if f.endswith('.tiff')]
        self.geojson_files = {}
        
        # Match geojson files to tiff files
        for tiff_file in self.tiff_files:
            base_name = os.path.basename(tiff_file).split('.')[0]
            geojson_file = os.path.join(geojson_dir, f"{base_name}.geojson")
            if os.path.exists(geojson_file):
                self.geojson_files[tiff_file] = geojson_file
        
        # Keep only tiff files that have matching geojson files
        self.tiff_files = [f for f in self.tiff_files if f in self.geojson_files]
        
        self.transform = transform
        
    def __len__(self):
        return len(self.tiff_files)
    
    def __getitem__(self, idx):
        tiff_path = self.tiff_files[idx]
        geojson_path = self.geojson_files[tiff_path]
        
        # Read the image
        with rasterio.open(tiff_path) as src:
            # Read all bands and transpose to channels-first format
            image = src.read()
            # Convert from (C, H, W) to (H, W, C) for transforms
            image = np.transpose(image, (1, 2, 0))
        
        # Create the mask from GeoJSON
        mask = geojson_to_mask(geojson_path, tiff_path)
        
        # Apply transforms if any
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            # Convert to PyTorch tensors
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0  # Normalize to [0, 1]
            mask = torch.from_numpy(mask).unsqueeze(0).float()  # Add channel dimension
        
        return image, mask
