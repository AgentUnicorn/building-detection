import os

import cv2
import geopandas as gpd
import numpy as np
import rasterio
import torch
from PIL import Image
from rasterio import features
from shapely.geometry import mapping
from tqdm import tqdm


def convert_geojsons_to_masks(geojson_dir, image_dir, mask_output_dir):
    """
    Converts all geojson files in a folder to raster masks based on matching TIFF images.

    Args:
        geojson_dir (str): Directory containing Geo_*.geojson files.
        image_dir (str): Directory containing 3band_*.tif images.
        mask_output_dir (str): Output directory for the generated masks.
    """
    os.makedirs(mask_output_dir, exist_ok=True)

    geojson_files = sorted(
        [f for f in os.listdir(geojson_dir) if f.endswith(".geojson")]
    )

    for geojson_file in tqdm(geojson_files, desc="Creating masks"):
        geojson_path = os.path.join(geojson_dir, geojson_file)

        aoi_part = geojson_file.replace("Geo_", "").replace(".geojson", "")
        tiff_file = f"3band_{aoi_part}.tif"
        tiff_path = os.path.join(image_dir, tiff_file)

        if not os.path.exists(tiff_path):
            print(f"[Warning] TIFF file not found for: {geojson_file}")
            continue

        with rasterio.open(tiff_path) as src:
            height, width = src.height, src.width
            transform = src.transform

        gdf = gpd.read_file(geojson_path)
        mask = np.zeros((height, width), dtype=np.uint8)

        if not gdf.empty:
            shapes = [(mapping(geom), 1) for geom in gdf.geometry]
            mask = features.rasterize(
                shapes,
                out_shape=(height, width),
                transform=transform,
                fill=0,
                dtype=np.uint8,
            )

        # Save mask
        output_path = os.path.join(mask_output_dir, f"mask_{aoi_part}.tif")
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=np.uint8,
            transform=transform,
        ) as dst:
            dst.write(mask, 1)


def create_sam_mask(predictor, image_path, unet_mask_path, sam_mask_output):
    with rasterio.open(image_path) as src:
        image = src.read([1, 2, 3]).transpose(1, 2, 0)
    predictor.set_image(image)

    # Load and binarize U-Net mask
    mask = cv2.imread(unet_mask_path, cv2.IMREAD_GRAYSCALE)
    binary_mask = (mask > 0.5).astype(np.uint8) * 255

    # Find contours -> bounding boxes
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) == 0:
        print(f"[Note] No buildings found")
        return

    boxes = [cv2.boundingRect(cnt) for cnt in contours]
    input_boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    # SAM prediction
    all_sam_masks = []
    for box in input_boxes:
        masks, _, _ = predictor.predict(box=box, multimask_output=False)
        all_sam_masks.append(masks[0])

    # Combine all SAM masks
    final_mask = np.any(all_sam_masks, axis=0).astype(np.uint8) * 255

    # Save as PNG
    cv2.imwrite(sam_mask_output, final_mask)


def create_sam_masks_from_unet_prediction(
    predictor, image_dir, unet_mask_dir, sam_mask_dir
):
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".tif")])

    for fname in tqdm(image_files, desc="SAM Ensemble"):
        image_path = os.path.join(image_dir, fname)
        base_name = os.path.splitext(fname)[0]
        unet_mask_path = os.path.join(unet_mask_dir, f"{base_name}.png")
        sam_mask_path = os.path.join(sam_mask_dir, f"{base_name}.png")

        if not os.path.exists(unet_mask_path):
            print(f"[Skip] Missing U-Net mask: {unet_mask_path}")
            continue

        with rasterio.open(image_path) as src:
            image = src.read([1, 2, 3]).transpose(1, 2, 0)
        predictor.set_image(image)

        # Load and binarize U-Net mask
        mask = cv2.imread(unet_mask_path, cv2.IMREAD_GRAYSCALE)
        binary_mask = (mask > 0.5).astype(np.uint8) * 255

        # Find contours -> bounding boxes
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(contours) == 0:
            print(f"[Note] No buildings found in {base_name}")
            continue

        boxes = [cv2.boundingRect(cnt) for cnt in contours]
        input_boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

        # SAM prediction
        all_sam_masks = []
        for box in input_boxes:
            masks, _, _ = predictor.predict(box=box, multimask_output=False)
            all_sam_masks.append(masks[0])

        # Combine all SAM masks
        final_mask = np.any(all_sam_masks, axis=0).astype(np.uint8) * 255

        # Save as PNG
        cv2.imwrite(sam_mask_path, final_mask)
