import glob
import os
import random

import albumentations as A
import geopandas as gpd
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from shapely.geometry import mapping
from torch.utils.data import DataLoader, Dataset, RandomSampler, Subset

from RioDataset import RioDataset


def create_data_loaders(
    data_dir,
    batch_size=4,
    num_workers=4,
):
    """
    Creates train, validation and test data loaders from a single directory
    without physically moving files.
    """
    # Define transforms
    train_transform = A.Compose(
        [
            A.Resize(448, 448),  # Make divisible by 32 (for 5-depth UNet)
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(),
            ToTensorV2(),
        ]
    )

    val_test_transform = A.Compose([A.Resize(448, 448), A.Normalize(), ToTensorV2()])

    # Dataset paths
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    # Create datasets
    train_dataset = RioDataset(train_dir, transform=train_transform)
    val_dataset = RioDataset(val_dir, transform=val_test_transform)
    test_dataset = RioDataset(test_dir, transform=val_test_transform)

    # Create subsets
    # train_subset = Subset(train_dataset, train_indices)
    # val_subset = Subset(val_dataset, val_indices)
    # test_subset = Subset(test_dataset, test_indices)

    # Create sampler
    sampler = RandomSampler(train_dataset)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
