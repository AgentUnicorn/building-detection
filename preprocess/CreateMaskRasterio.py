import os

import geopandas as gpd
import numpy as np
import rasterio
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
