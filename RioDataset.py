import os
import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np
from torchvision import transforms
import glob
import geopandas as gpd
from shapely.geometry import mapping

class RioDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir: Directory containing both .tiff and .geojson files
            transform: Optional transform to be applied
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Get all tiff files
        self.tiff_files = glob.glob(os.path.join(data_dir, "*.tif"))

        # Create mapping between tiff files and corresponding geojson files
        self.valid_pairs = []
        for tiff_path in self.tiff_files:
            tiff_filename = os.path.basename(tiff_path)
        
            if tiff_filename.startswith("3band_"):
                aoi_part = tiff_filename.replace("3band_", "").replace(".tif", "")
                geojson_name = f"Geo_{aoi_part}.geojson"
                geojson_path = os.path.join(data_dir, geojson_name)

                if os.path.exists(geojson_path):
                    self.valid_pairs.append((tiff_path, geojson_path))
        
        print(f"Found {len(self.valid_pairs)} valid image-mask pairs")
    
    def __len__(self):
        return len(self.valid_pairs)
    
    def __getitem__(self, idx):
        tiff_path, geojson_path = self.valid_pairs[idx]
        
        # Read image
        with rasterio.open(tiff_path) as src:
            # Read all available bands
            image = src.read()
            # Move from (C,H,W) to (H,W,C) for compatibility with transforms
            image = np.transpose(image, (1, 2, 0))
            # Get metadata for mask creation
            transform = src.transform
            height, width = src.height, src.width
        
        # Create mask from GeoJSON
        mask = np.zeros((height, width), dtype=np.uint8)
        gdf = gpd.read_file(geojson_path)
        
        if not gdf.empty:
            shapes = [(mapping(geom), 1) for geom in gdf.geometry]
            mask = rasterio.features.rasterize(
                shapes,
                out_shape=(height, width),
                transform=transform,
                fill=0,
                dtype=np.uint8
            )
        
        # Apply transforms if specified
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

            # Fix error where Albumentations converts the mask to shape [H, W] but BCEWithLogitsLoss expect same shape of input and target [1, H, W]
            if mask.dim() == 2
                mask = mask.unsqueeze(0)

            # Fix error of BCEWithLogitsLoss that expect input and target to be Float
            mask = mask.float()
        else:
            # Convert to torch tensors
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0).float()
        
        return image, mask