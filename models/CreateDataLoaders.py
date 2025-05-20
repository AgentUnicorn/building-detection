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

from models.CreateDataset import CreateDataset_3band
from preprocess.Transformer import getTrainTransform, getValidTransform


def CreateDataLoaders(
    data_dir,
    batch_size,
    num_workers,
):
    """
    Creates train, validation and test data loaders from a single directory
    without physically moving files.
    """
    # Define transforms
    train_transform = getTrainTransform()
    val_test_transform = getValidTransform()

    # Dataset paths
    train_dir = os.path.join(data_dir, "train/3band")
    train_mask_dir = os.path.join(data_dir, "train/mask")
    val_dir = os.path.join(data_dir, "val/3band")
    val_mask_dir = os.path.join(data_dir, "val/mask")

    # Create datasets
    train_dataset = CreateDataset_3band(
        data_dir=train_dir, mask_dir=train_mask_dir, transform=train_transform
    )
    val_dataset = CreateDataset_3band(
        data_dir=val_dir, mask_dir=val_mask_dir, transform=val_test_transform
    )

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

    return train_loader, val_loader
