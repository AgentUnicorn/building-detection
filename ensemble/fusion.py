import os

import numpy as np
import pydensecrf.densecrf as dcrf
import rasterio
from PIL import Image
from pydensecrf.utils import create_pairwise_bilateral, unary_from_softmax
from tqdm import tqdm


def apply_crf(original_image, mask_probs):
    H, W = mask_probs.shape
    d = dcrf.DenseCRF2D(W, H, 2)

    probs = np.stack([1 - mask_probs, mask_probs], axis=0)
    unary = unary_from_softmax(probs)
    d.setUnaryEnergy(unary)

    feats = create_pairwise_bilateral(
        sdims=(10, 10), schan=(10, 10, 10), img=original_image, chdim=2
    )
    d.addPairwiseEnergy(feats, compat=10)

    Q = d.inference(5)
    refined = np.argmax(Q, axis=0).reshape((H, W))

    return refined.astype(np.uint8)


def read_rgb_from_tif(tif_path):
    with rasterio.open(tif_path) as src:
        image = src.read([1, 2, 3])
        image = image.transpose(1, 2, 0).astype(np.uint8)
    return image


def batch_fusion(image_dir, unet_mask_dir, sam_mask_dir, fused_mask_dir):
    fused_union_dir = os.path.join(fused_mask_dir, "union")
    fused_weighted_dir = os.path.join(fused_mask_dir, "weighted")
    fused_crf_dir = os.path.join(fused_mask_dir, "crf")

    os.makedirs(fused_union_dir, exist_ok=True)
    os.makedirs(fused_weighted_dir, exist_ok=True)
    os.makedirs(fused_crf_dir, exist_ok=True)

    for filename in tqdm(os.listdir(unet_mask_dir), desc="Fusion masks"):
        if not filename.endswith(".png"):
            continue

        base_name = os.path.splitext(filename)[0]
        original_image_path = os.path.join(image_dir, f"{base_name}.tif")
        unet_mask_path = os.path.join(unet_mask_dir, f"{base_name}.png")
        sam_mask_path = os.path.join(sam_mask_dir, f"{base_name}.png")

        if not os.path.exists(sam_mask_path):
            print(f"[Skip] Missing SAM mask: {base_name}")
            continue

        unet_mask = np.array(Image.open(unet_mask_path)) / 255.0
        sam_mask = np.array(Image.open(sam_mask_path)) / 255.0

        # Union
        union_mask = np.logical_or(unet_mask, sam_mask).astype(np.uint8) * 255
        Image.fromarray(union_mask).save(os.path.join(fused_union_dir, filename))

        # Weighted Fusion
        alpha, beta = 0.6, 0.4
        weighted = alpha * unet_mask + beta * sam_mask
        weighted_mask = (weighted > 0.5).astype(np.uint8) * 255
        Image.fromarray(weighted_mask).save(os.path.join(fused_weighted_dir, filename))

        # CRF
        original_image = read_rgb_from_tif(original_image_path)
        crf_result = apply_crf(original_image, weighted)
        Image.fromarray(crf_result * 255).save(os.path.join(fused_crf_dir, filename))


def fusion(image_path, unet_mask_path, sam_mask_path, fused_mask_path):
    if not os.path.exists(sam_mask_path) or not os.path.exists(unet_mask_path):
        print("Missing SAM or U-Net mask")
        return

    unet_mask = np.array(Image.open(unet_mask_path)) / 255.0
    sam_mask = np.array(Image.open(sam_mask_path)) / 255.0
    filename = os.path.splitext(os.path.basename(unet_mask_path))[0]

    # Union
    union_mask = np.logical_or(unet_mask, sam_mask).astype(np.uint8) * 255
    Image.fromarray(union_mask).save(
        os.path.join(fused_mask_path, f"union_{filename}.png")
    )

    # Weighted Fusion
    alpha, beta = 0.6, 0.4
    weighted = alpha * unet_mask + beta * sam_mask
    weighted_mask = (weighted > 0.5).astype(np.uint8) * 255
    Image.fromarray(weighted_mask).save(
        os.path.join(fused_mask_path, f"weighted_{filename}.png")
    )

    # CRF
    original_image = read_rgb_from_tif(image_path)
    crf_result = apply_crf(original_image, weighted)
    Image.fromarray(crf_result * 255).save(
        os.path.join(fused_mask_path, f"crf_{filename}.png")
    )
