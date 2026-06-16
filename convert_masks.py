"""
CONVERT LABEL STUDIO EXPORT -> PNG MASKS
==========================================
Uses the OFFICIAL label-studio-sdk RLE decoder to convert
brush annotations into clean PNG mask files.

Input:
  - Label Studio JSON export file
  - Original patch PNG images (for size reference)

Output:
  ISRO_14/masks_seg/
      patch_00000_mask.png   <- 256x256 grayscale PNG
      ...
      pixel values:
        0 = Background (None)
        1 = Urban Dense
        2 = Hills Rural

Usage:
  python convert_masks.py "C:/path/to/labelstudio_export.json"
"""

import os
import sys
import re
import json
import numpy as np
from PIL import Image

try:
    from label_studio_sdk.converter.brush import decode_rle
except ImportError:
    print("ERROR: label-studio-sdk not installed.")
    print("Run: pip install label-studio-sdk")
    sys.exit(1)

# ─────────────────────────────────────────────
BASE_DIR     = r"C:\Users\shiva\OneDrive\Documents\ISRO_14"
SEG_IMG_DIR  = os.path.join(BASE_DIR, "seg_to_label")   # original PNGs uploaded to LS
MASKS_DIR    = os.path.join(BASE_DIR, "masks_seg")       # output masks
os.makedirs(MASKS_DIR, exist_ok=True)

# Class name -> pixel value mapping (must match your Label Studio config)
CLASS_TO_ID = {
    "Urban Dense": 1,
    "Hills Rural": 2,
}
# Background (unpainted) = 0

# If a class is painted over a region, this controls draw order priority.
# Higher number = painted LAST = wins in overlaps.
CLASS_PRIORITY = {
    "Hills Rural": 1,
    "Urban Dense": 2,
}


def strip_ls_prefix(img_name):
    """
    Label Studio prefixes uploaded filenames with an 8-char hex hash:
    'e209b8ea-patch_00186.png' -> 'patch_00186.png'
    """
    match = re.match(r'^[a-f0-9]{8}-(.+)$', img_name)
    if match:
        return match.group(1)
    return img_name


def decode_brush_mask(rle_list, width, height):
    """
    Decode a Label Studio brush RLE annotation into a binary mask
    using the official SDK decoder.

    Label Studio encodes brush masks as RGBA images flattened to RLE.
    We only need the alpha channel (channel index 3) which indicates
    painted (255) vs unpainted (0) pixels.

    Returns: (H, W) uint8 array, 1 = painted, 0 = not painted
    """
    rle_bytes = bytes(rle_list)
    decoded = decode_rle(rle_bytes)  # flat array of RGBA values

    expected_len = width * height * 4
    if len(decoded) != expected_len:
        # Try cropping/padding defensively
        if len(decoded) > expected_len:
            decoded = decoded[:expected_len]
        else:
            decoded = np.pad(decoded, (0, expected_len - len(decoded)))

    image = decoded.reshape(height, width, 4)
    alpha = image[:, :, 3]  # alpha channel = the actual painted mask
    mask = (alpha > 0).astype(np.uint8)
    return mask


def process_export(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    print(f"Found {len(data)} labeled images in export.")
    saved = 0
    skipped = 0

    for item in data:
        # Get image filename, handle LS hash prefix
        img_url = item.get("data", {}).get("image", "")
        img_name = os.path.basename(img_url.split("?")[0])
        img_name = img_name.replace("%20", " ")
        img_name = strip_ls_prefix(img_name)

        print(f"\nProcessing: {img_name}")

        # Locate source image (for width/height reference)
        src_img_path = os.path.join(SEG_IMG_DIR, img_name)
        if not os.path.exists(src_img_path):
            src_img_path = os.path.join(BASE_DIR, "patches_png", img_name)
        if not os.path.exists(src_img_path):
            print(f"  [SKIP] Source image not found: {img_name}")
            skipped += 1
            continue

        src_img = Image.open(src_img_path)
        W, H = src_img.size

        combined_mask = np.zeros((H, W), dtype=np.uint8)

        annotations = item.get("annotations", [])
        if not annotations:
            print(f"  [SKIP] No annotations found")
            skipped += 1
            continue

        annotation = annotations[0]
        results = annotation.get("result", [])
        if not results:
            print(f"  [SKIP] Empty annotation result")
            skipped += 1
            continue

        # Sort results by class priority so higher-priority classes
        # get painted last (and thus win pixel overlaps)
        def get_priority(result):
            labels = result.get("value", {}).get("brushlabels", ["zzz"])
            return CLASS_PRIORITY.get(labels[0], 0)

        brush_results = [r for r in results if r.get("type") == "brushlabels"]
        brush_results.sort(key=get_priority)

        any_painted = False
        for result in brush_results:
            labels = result.get("value", {}).get("brushlabels", [])
            if not labels:
                continue
            label_name = labels[0]
            class_id = CLASS_TO_ID.get(label_name, 0)
            if class_id == 0:
                print(f"  [WARN] Unknown label '{label_name}', skipping")
                continue

            rle = result.get("value", {}).get("rle", None)
            if rle is None:
                print(f"  [WARN] No RLE data for label '{label_name}'")
                continue

            orig_w = result.get("original_width", W)
            orig_h = result.get("original_height", H)

            brush_mask = decode_brush_mask(rle, orig_w, orig_h)

            # Resize if annotation resolution differs from image
            if (orig_w, orig_h) != (W, H):
                brush_pil = Image.fromarray(brush_mask * 255).resize(
                    (W, H), Image.NEAREST
                )
                brush_mask = (np.array(brush_pil) > 127).astype(np.uint8)

            painted_pixels = int(brush_mask.sum())
            print(f"  Label '{label_name}' (id={class_id}): "
                  f"{painted_pixels} / {W*H} pixels painted "
                  f"({painted_pixels/(W*H)*100:.1f}%)")

            if painted_pixels > 0:
                any_painted = True
            combined_mask[brush_mask == 1] = class_id

        if not any_painted:
            print(f"  [WARN] No pixels painted for this image — empty mask saved")

        mask_name = img_name.replace(".png", "_mask.png")
        mask_path = os.path.join(MASKS_DIR, mask_name)
        Image.fromarray(combined_mask).save(mask_path)
        unique_vals = np.unique(combined_mask)
        print(f"  Saved mask: {mask_path}")
        print(f"  Mask stats: unique values = {unique_vals}")
        saved += 1

    print(f"\n{'='*50}")
    print(f"DONE! Saved {saved} masks, skipped {skipped}.")
    print(f"Masks directory: {MASKS_DIR}")
    print(f"\nPixel value legend:")
    print(f"  0 = Background (None)")
    print(f"  1 = Urban Dense")
    print(f"  2 = Hills Rural")
    print(f"\nNext step: Run train_unet.py to train the segmentation model.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_masks.py <path_to_labelstudio_export.json>")
        sys.exit(1)

    json_path = sys.argv[1]
    if not os.path.exists(json_path):
        print(f"ERROR: JSON file not found: {json_path}")
        sys.exit(1)

    process_export(json_path)