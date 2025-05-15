import os
import random
import shutil

from tqdm import tqdm


def split_dataset(image_dir, geojson_dir, output_root, total_samples=None, seed=42):
    random.seed(seed)

    # Get sorted list of image files
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".tif")])

    if total_samples is not None:
        image_files = image_files[:total_samples]

    # Compute sizes
    n = len(image_files)
    train_cutoff = int(0.7 * n)
    val_cutoff = int(0.85 * n)

    # Split into three groups (no shuffle yet)
    train_files = image_files[:train_cutoff]
    val_files = image_files[train_cutoff:val_cutoff]
    test_files = image_files[val_cutoff:]

    # Shuffle each group separately
    random.shuffle(train_files)
    random.shuffle(val_files)
    random.shuffle(test_files)

    splits = {"train": train_files, "val": val_files, "test": test_files}

    for split, files in splits.items():
        split_dir = os.path.join(output_root, split)
        os.makedirs(split_dir, exist_ok=True)

        print(f"\n[INFO] Copying {len(files)} files to '{split}/'...")

        for tif_file in tqdm(files, desc=f"Copying to {split}", unit="file"):
            # Copy TIFF
            src_img = os.path.join(image_dir, tif_file)
            dst_img = os.path.join(split_dir, tif_file)
            shutil.copy2(src_img, dst_img)

            # Find and copy matching GeoJSON
            geojson_name = tif_file.replace("3band_", "Geo_").replace(
                ".tif", ".geojson"
            )
            src_json = os.path.join(geojson_dir, geojson_name)
            dst_json = os.path.join(split_dir, geojson_name)

            if os.path.exists(src_json):
                shutil.copy2(src_json, dst_json)
            else:
                print(f"[Warning] Missing GeoJSON for: {tif_file}")

        print(f"[✓] {split} set created with {len(files)} image-mask pairs.")

    print(f"[Done] Dataset split into train/val/test folders under: {output_root}")
