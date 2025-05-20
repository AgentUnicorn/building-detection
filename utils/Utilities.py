import numpy as np
import rasterio
import torch

from preprocess.Transformer import getValidTransform


def load_image(tif_path):
    with rasterio.open(tif_path) as src:
        image = src.read()
        image = np.transpose(image, (1, 2, 0))
    return image


def predict(model, image_path):
    image = load_image(image_path)

    transform = getValidTransform()
    augmented = transform(image=image)
    image_tensor = (
        augmented["image"]
        .unsqueeze(0)
        .to("cuda" if torch.cuda.is_available() else "cpu")
    )

    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.sigmoid(output)
        prediction = (prediction > 0.5).float().squeeze().cpu().numpy()
    return prediction
