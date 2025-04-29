import numpy as np
import glob
import random
import torch
from shapely.geometry import mapping
import albumentations as A
import geopandas as gpd
from albumentations.pytorch import ToTensorV2
from RioDataset import RioDataset
from torch.utils.data import Dataset, DataLoader, Subset, RandomSampler

def create_data_loaders(data_dir, batch_size=4, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Creates train, validation and test data loaders from a single directory
    without physically moving files.
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    
    # Define transforms
    train_transform = A.Compose([
        A.Resize(448, 448),  # Make divisible by 32 (for 5-depth UNet)
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(),
        ToTensorV2()
    ])
    
    val_test_transform = A.Compose([
        A.Resize(448, 448),
        A.Normalize(),
        ToTensorV2()
    ])
    
    # Create full dataset without transforms initially (to split indices)
    full_dataset = RioDataset(data_dir, transform=None)
    
    # Generate indices for the split
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    random.shuffle(indices)
    
    # Calculate split sizes
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    
    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    # Create datasets with appropriate transforms
    train_dataset = RioDataset(data_dir, transform=train_transform)
    val_dataset = RioDataset(data_dir, transform=val_test_transform)
    test_dataset = RioDataset(data_dir, transform=val_test_transform)
    
    # Create subsets
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    test_subset = Subset(test_dataset, test_indices)
    
    print(f"Dataset split: train={len(train_subset)}, val={len(val_subset)}, test={len(test_subset)}")

    # Create sampler
    sampler = RandomSampler(train_subset)
    
    # Create data loaders
    num_workers = 2
    train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader, full_dataset