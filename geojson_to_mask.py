import geopandas as gpd
import numpy as np
import rasterio
import rasterio.features
from shapely.geometry import mapping


def geojson_to_mask(geojson_path, tiff_path, output_path=None):

    # Read the TIFF file to get the image dimensions and transform
    with rasterio.open(tiff_path) as src:
        height, width = src.height, src.width
        transform = src.transform
    
    # Read building polygons from GeoJSON
    gdf = gpd.read_file(geojson_path)
    
    # Create an empty mask
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Rasterize the polygons onto the mask
    if not gdf.empty:
        shapes = [(mapping(geom), 1) for geom in gdf.geometry]
        mask = rasterio.features.rasterize(shapes, 
                                          out_shape=(height, width),
                                          transform=transform,
                                          fill=0,
                                          dtype=np.uint8)
    
    # Save the mask if an output path is provided
    if output_path:
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=np.uint8,
            transform=transform,
        ) as dst:
            dst.write(mask, 1)
    
    return mask
