import os
import random

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from PIL import Image


def visualize_training_metrics(history, save_path=None, figsize=(15, 10)):
    """
    Visualize training metrics from history dictionary.

    Args:
        history (dict): Dictionary containing metric lists
        save_path (str, optional): Path to save the plot
        figsize (tuple): Figure size (width, height)
    """

    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle("Training Metrics Overview", fontsize=16, fontweight="bold")

    # Flatten axes for easier indexing
    axes = axes.flatten()

    # Define colors and styles
    colors = {
        "train_loss": "#e74c3c",
        "val_loss": "#3498db",
        "dice": "#2ecc71",
        "iou": "#f39c12",
        "f1": "#9b59b6",
        "accuracy": "#1abc9c",
    }

    # Plot 1: Loss curves
    ax1 = axes[0]
    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(
        epochs,
        history["train_loss"],
        color=colors["train_loss"],
        linewidth=2,
        label="Training Loss",
        marker="o",
        markersize=4,
    )
    ax1.plot(
        epochs,
        history["val_loss"],
        color=colors["val_loss"],
        linewidth=2,
        label="Validation Loss",
        marker="s",
        markersize=4,
    )
    ax1.set_title("Training & Validation Loss", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Dice Score
    ax2 = axes[1]
    ax2.plot(
        epochs,
        history["dice"],
        color=colors["dice"],
        linewidth=2,
        marker="o",
        markersize=4,
    )
    ax2.set_title("Dice Score", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Dice Score")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    # Plot 3: IoU Score
    ax3 = axes[2]
    ax3.plot(
        epochs,
        history["iou"],
        color=colors["iou"],
        linewidth=2,
        marker="o",
        markersize=4,
    )
    ax3.set_title("IoU (Intersection over Union)", fontweight="bold")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("IoU Score")
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)

    # Plot 4: F1 Score
    ax4 = axes[3]
    ax4.plot(
        epochs, history["f1"], color=colors["f1"], linewidth=2, marker="o", markersize=4
    )
    ax4.set_title("F1 Score", fontweight="bold")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("F1 Score")
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1)

    # Plot 5: Accuracy
    ax5 = axes[4]
    ax5.plot(
        epochs,
        history["accuracy"],
        color=colors["accuracy"],
        linewidth=2,
        marker="o",
        markersize=4,
    )
    ax5.set_title("Accuracy", fontweight="bold")
    ax5.set_xlabel("Epoch")
    ax5.set_ylabel("Accuracy")
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, 1)

    # Plot 6: All performance metrics together
    ax6 = axes[5]
    ax6.plot(
        epochs,
        history["dice"],
        color=colors["dice"],
        linewidth=2,
        label="Dice",
        marker="o",
        markersize=3,
    )
    ax6.plot(
        epochs,
        history["iou"],
        color=colors["iou"],
        linewidth=2,
        label="IoU",
        marker="s",
        markersize=3,
    )
    ax6.plot(
        epochs,
        history["f1"],
        color=colors["f1"],
        linewidth=2,
        label="F1",
        marker="^",
        markersize=3,
    )
    ax6.plot(
        epochs,
        history["accuracy"],
        color=colors["accuracy"],
        linewidth=2,
        label="Accuracy",
        marker="d",
        markersize=3,
    )
    ax6.set_title("All Performance Metrics", fontweight="bold")
    ax6.set_xlabel("Epoch")
    ax6.set_ylabel("Score")
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(0, 1)

    # Adjust layout
    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    # Show plot
    plt.show()

    # Print summary statistics
    print("\n" + "=" * 50)
    print("TRAINING SUMMARY")
    print("=" * 50)

    for metric, values in history.items():
        if values:  # Check if list is not empty
            print(f"{metric.upper()}:")
            print(f"  Final: {values[-1]:.4f}")
            if metric in ["dice", "iou", "f1", "accuracy"]:
                print(
                    f"  Best:  {max(values):.4f} (Epoch {values.index(max(values)) + 1})"
                )
            elif "loss" in metric:
                print(
                    f"  Best:  {min(values):.4f} (Epoch {values.index(min(values)) + 1})"
                )
            print()


def visualize_metrics_compact(history, save_path=None, figsize=(12, 8)):
    """
    Create a more compact visualization with 2x2 layout.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle("Training Metrics", fontsize=14, fontweight="bold")

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss plot
    axes[0, 0].plot(
        epochs, history["train_loss"], "r-", label="Train Loss", linewidth=2
    )
    axes[0, 0].plot(epochs, history["val_loss"], "b-", label="Val Loss", linewidth=2)
    axes[0, 0].set_title("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Dice & IoU
    axes[0, 1].plot(epochs, history["dice"], "g-", label="Dice", linewidth=2)
    axes[0, 1].plot(epochs, history["iou"], "orange", label="IoU", linewidth=2)
    axes[0, 1].set_title("Dice & IoU")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1)

    # F1 Score
    axes[1, 0].plot(epochs, history["f1"], "purple", linewidth=2)
    axes[1, 0].set_title("F1 Score")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1)

    # Accuracy
    axes[1, 1].plot(epochs, history["accuracy"], "teal", linewidth=2)
    axes[1, 1].set_title("Accuracy")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def show_random_image_mask_pairs(image_dir, mask_dir, num_samples=5):
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".tif")]
    if len(image_files) == 0:
        print("No images found in the directory.")
        return

    selected_files = random.sample(image_files, min(num_samples, len(image_files)))

    for img_name in selected_files:
        # Get suffix after '3band_'
        if "3band_" not in img_name:
            print(f"Skipping unrecognized image name: {img_name}")
            continue

        suffix = img_name.split("3band_")[-1]
        mask_name = f"mask_{suffix}"
        image_path = os.path.join(image_dir, img_name)
        mask_path = os.path.join(mask_dir, mask_name)

        if not os.path.exists(mask_path):
            print(f"Mask not found for {img_name} -> Expected: {mask_name}")
            continue

        # Read .tif image using rasterio
        with rasterio.open(image_path) as src:
            image = src.read([1, 2, 3]).transpose(1, 2, 0)
            image = image / image.max()

        # Load mask as grayscale and normalize
        with rasterio.open(mask_path) as mask_src:
            mask = mask_src.read(1)  # Single channel
        if mask.max() > 1:
            mask = mask / 255.0

        # Plotting
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title(f"Original Image: {img_name}")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(image)
        plt.imshow(mask, cmap="gray")
        plt.title("Ground Truth Mask")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(image)
        plt.imshow(mask, cmap="Reds", alpha=0.5)
        plt.title("Original + Ground Truth Mask")
        plt.axis("off")

        plt.tight_layout()
        plt.show()


def show_random_image_mask_pairs(image_dir, mask_dir, num_samples=5):
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".tif")]
    if len(image_files) == 0:
        print("No images found in the directory.")
        return

    selected_files = random.sample(image_files, min(num_samples, len(image_files)))

    for img_name in selected_files:
        # Get suffix after '3band_'
        if "3band_" not in img_name:
            print(f"Skipping unrecognized image name: {img_name}")
            continue

        suffix = img_name.split("3band_")[-1]
        mask_name = f"mask_{suffix}"
        image_path = os.path.join(image_dir, img_name)
        mask_path = os.path.join(mask_dir, mask_name)

        if not os.path.exists(mask_path):
            print(f"Mask not found for {img_name} -> Expected: {mask_name}")
            continue

        # Read .tif image using rasterio
        with rasterio.open(image_path) as src:
            image = src.read([1, 2, 3]).transpose(1, 2, 0)
            image = image / image.max()

        # Load mask as grayscale and normalize
        with rasterio.open(mask_path) as mask_src:
            mask = mask_src.read(1)  # Single channel
        if mask.max() > 1:
            mask = mask / 255.0

        # Plotting
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title(f"Original Image: {img_name}")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(image)
        plt.imshow(mask, cmap="gray")
        plt.title("Ground Truth Mask")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(image)
        plt.imshow(mask, cmap="Reds", alpha=0.5)
        plt.title("Original + Ground Truth Mask")
        plt.axis("off")

        plt.tight_layout()
        plt.show()


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
