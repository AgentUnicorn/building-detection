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
