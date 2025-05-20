import glob
import os

import geopandas as gpd
import numpy as np
import rasterio
import torch
from rasterio import features
from shapely.geometry import mapping
from torch.utils.data import Dataset
from torchvision import transforms


class CreateDataset_3band(Dataset):
    def __init__(self, data_dir, mask_dir, transform=None):
        """
        Args:
            data_dir: Directory containing both .tiff and .geojson files
            transform: Optional transform to be applied
        """
        self.data_dir = data_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # Get all tiff files
        self.tiff_files = glob.glob(os.path.join(data_dir, "3band_*.tif"))

        # Create mapping between tiff files and corresponding geojson files
        self.valid_pairs = []
        for tiff_path in self.tiff_files:
            tiff_filename = os.path.basename(tiff_path)

            if tiff_filename.startswith("3band_"):
                tiff_filename = os.path.basename(tiff_path)
                aoi_part = tiff_filename.replace("3band_", "").replace(".tif", "")
                mask_filename = f"mask_{aoi_part}.tif"
                mask_path = os.path.join(mask_dir, mask_filename)

                if os.path.exists(mask_path):
                    self.valid_pairs.append((tiff_path, mask_path))

        print(f"Found {len(self.valid_pairs)} valid image-mask pairs")

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        tiff_path, mask_path = self.valid_pairs[idx]

        # Read image
        with rasterio.open(tiff_path) as src:
            image = src.read()
            image = np.transpose(image, (1, 2, 0))  # C,H,W to H,W,C

        with rasterio.open(mask_path) as msk:
            mask = msk.read(1)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

            # Fix error where Albumentations converts the mask to shape [H, W] but BCEWithLogitsLoss expect same shape of input and target [1, H, W]
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)

            # Fix error of BCEWithLogitsLoss that expect input and target to be Float
            mask = mask.float()
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0).float()

        return image, mask
