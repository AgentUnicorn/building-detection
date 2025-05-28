import os

import matplotlib.pyplot as plt
import numpy as np
import rasterio


def visualizeMask(mask_path):

    with rasterio.open(mask_path) as src:
        mask = src.read(1)

    plt.figure(figsize=(8, 8))
    plt.imshow(mask, cmap="viridis")
    plt.title("Poly Mask")
    plt.axis("off")
    plt.show()


def visualizePrediction(prediction):
    plt.imshow(prediction, cmap="gray")
    plt.title("Predicted Building Mask")
    plt.axis("off")
    plt.show()


def visualizeOriginalWithMask(image_path, prediction):
    with rasterio.open(image_path) as src:
        image = src.read([1, 2, 3])
        image = image.transpose(1, 2, 0)
        image = image / image.max()

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.imshow(prediction, cmap="Reds", alpha=0.5)  # Overlay in red
    plt.title("Prediction Overlay")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def visualize_with_multiple_masks(
    image_path, mask1=None, mask2=None, mask3=None, save_path=None
):
    with rasterio.open(image_path) as src:
        image = src.read([1, 2, 3]).transpose(1, 2, 0)
        image = image / image.max()

    plt.figure(figsize=(16, 8))

    # Original image
    plt.subplot(1, 4, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    # Mask 1
    if mask1 is not None:
        mask_path1, title1 = mask1
        plt.subplot(1, 4, 2)
        plt.imshow(image)
        plt.imshow(mask_path1, cmap="Reds", alpha=0.5)
        plt.title(title1)
        plt.axis("off")

    # Mask 2
    if mask2 is not None:
        mask_path2, title2 = mask2
        plt.subplot(1, 4, 3)
        plt.imshow(image)
        plt.imshow(mask_path2, cmap="Greens", alpha=0.5)
        plt.title(title2)
        plt.axis("off")

    # Mask 3
    if mask3 is not None:
        mask_path3, title3 = mask3
        plt.subplot(1, 4, 4)
        plt.imshow(image)
        plt.imshow(mask_path3, cmap="Blues", alpha=0.5)
        plt.title(title3)
        plt.axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)

    plt.show()
    plt.close()
