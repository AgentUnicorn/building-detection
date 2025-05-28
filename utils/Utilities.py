import os

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import save_image

from preprocess.Transformer import getValidTransform


def load_image(tif_path):
    with rasterio.open(tif_path) as src:
        image = src.read()
        image = np.transpose(image, (1, 2, 0))
    return image


def predict(model, image_path, width, height):
    image = load_image(image_path)

    transform = getValidTransform(width, height)
    augmented = transform(image=image)
    image_tensor = (
        augmented["image"]
        .unsqueeze(0)
        .to("cuda" if torch.cuda.is_available() else "cpu")
    )

    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.sigmoid(output)
        # prediction = (prediction > 0.5).float().squeeze().cpu().numpy()
        prediction = (prediction > 0.5).float()

        with rasterio.open(image_path) as src:
            orig_height, orig_width = src.height, src.width

        prediction = (
            F.interpolate(prediction, size=(orig_height, orig_width), mode="bilinear")
            .squeeze()
            .cpu()
            .numpy()
        )

    return prediction


def save_prediction_as_png(prediction, output_path):
    """Save binary prediction mask as a PNG (0 and 255)"""
    binary_mask = (prediction > 0.5).astype(np.uint8) * 255
    Image.fromarray(binary_mask).save(output_path)


def batch_predict_and_save(model, image_dir, output_dir, width, height):
    os.makedirs(output_dir, exist_ok=True)

    tif_files = [f for f in os.listdir(image_dir) if f.endswith(".tif")]
    print(f"[INFO] Found {len(tif_files)} .tif images in: {image_dir}")

    for tif_file in tif_files:
        image_path = os.path.join(image_dir, tif_file)
        print(f"[Predicting] {tif_file}")

        prediction = predict(model, image_path, width, height)

        # Save the predicted mask
        mask_name = os.path.splitext(tif_file)[0] + ".png"
        mask_path = os.path.join(output_dir, mask_name)
        save_prediction_as_png(prediction, mask_path)

    print(f"[✓] All predictions saved to: {output_dir}")


def predict_and_save(model, image_path, output_path, width, height):
    filename = os.path.splitext(os.path.basename(image_path))[0]
    prediction = predict(model, image_path, width, height)

    # Save the predicted mask
    mask_name = filename + ".png"
    mask_path = os.path.join(output_path, mask_name)
    save_prediction_as_png(prediction, mask_path)
