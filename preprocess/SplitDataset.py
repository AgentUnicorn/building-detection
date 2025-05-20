import os
import random
import shutil

from tqdm import tqdm


def SplitDataset(data_dir, target_dir, total_samples=None, seed=42):
    random.seed(seed)

    # Image folders
    image_3b = data_dir + "/3band"
    image_8b = data_dir + "/8band"
    geojson = data_dir + "/geojson"

    # Get sorted list of image files
    image_files = sorted([f for f in os.listdir(image_3b) if f.endswith(".tif")])
    random.shuffle(image_files)

    if total_samples is not None:
        image_files = image_files[:total_samples]

    # Compute sizes
    n = len(image_files)
    train_cutoff = int(0.85 * n)

    # Split into three groups (no shuffle yet)
    train_files = image_files[:train_cutoff]
    val_files = image_files[train_cutoff:]

    splits = {"train": train_files, "val": val_files}

    for split, files in splits.items():
        split_dir = os.path.join(target_dir, split)
        for sub in ["3band", "8band", "geojson"]:
            os.makedirs(os.path.join(split_dir, sub), exist_ok=True)

        print(f"\n[INFO] Copying {len(files)} files to '{split}/'...")

        for tif_file in tqdm(files, desc=f"Copying to {split}", unit="file"):
            # Copy TIFF
            src_img = os.path.join(image_3b, tif_file)
            target_3b = os.path.join(split_dir, "3band", tif_file)
            shutil.copy2(src_img, target_3b)

            # Find and copy matching 8band
            name_8b = tif_file.replace("3band_", "8band_")
            src_8b = os.path.join(image_8b, name_8b)
            target_8b = os.path.join(split_dir, "8band", tif_file)

            if os.path.exists(src_8b):
                shutil.copy2(src_8b, target_8b)
            else:
                print(f"[Warning] Missing 8band for: {tif_file}")

            # Find and copy matching GeoJSON
            geojson_name = tif_file.replace("3band_", "Geo_").replace(
                ".tif", ".geojson"
            )
            src_json = os.path.join(geojson, geojson_name)
            target_json = os.path.join(split_dir, "geojson", geojson_name)

            if os.path.exists(src_json):
                shutil.copy2(src_json, target_json)
            else:
                print(f"[Warning] Missing GeoJSON for: {tif_file}")

        print(f"[✓] {split} set created with {len(files)} image-mask pairs.")

    print(f"[Done] Dataset split into train/val/test folders under: {target_dir}")
