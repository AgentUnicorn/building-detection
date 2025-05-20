import matplotlib.pyplot as plt
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
