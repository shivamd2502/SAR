"""
CONVERT PATCHES TO PNG FOR LABEL STUDIO
=========================================
Converts each 256x256x2 normalized .npy patch into a PNG image
that Label Studio (or any annotation tool) can display and annotate.

Two output modes:
  - "hh"   : grayscale image using only the HH channel (simplest)
  - "rgb"  : 3-channel composite (R=HH, G=HV, B=ratio) - more visual detail

Output:
  ISRO_14/patches_png/patch_00000.png
  ISRO_14/patches_png/patch_00001.png
  ...
"""

import os
import numpy as np
import glob
from pathlib import Path
from PIL import Image

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_DIR    = r"C:\Users\shiva\OneDrive\Documents\ISRO_14"
PATCHES_DIR = os.path.join(BASE_DIR, "patches")
OUTPUT_DIR  = os.path.join(BASE_DIR, "patches_png")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Choose mode: "hh" (grayscale) or "rgb" (3-channel composite)
MODE = "rgb"


def patch_to_image(patch, mode="rgb"):
    """
    Convert a (256, 256, 2) normalized [0,1] patch to a uint8 image array.
    """
    hh = patch[:, :, 0]
    hv = patch[:, :, 1]

    if mode == "hh":
        # Simple grayscale using HH channel only
        img = (hh * 255).astype(np.uint8)
        return img  # shape (256, 256) -> grayscale PNG

    elif mode == "rgb":
        # R=HH, G=HV, B=HH-HV difference (same as your overview composite)
        r = hh
        g = hv
        b = np.clip((r - g + 1) / 2, 0, 1)
        rgb = np.stack([r, g, b], axis=-1)
        img = (rgb * 255).astype(np.uint8)
        return img  # shape (256, 256, 3) -> RGB PNG

    else:
        raise ValueError(f"Unknown mode: {mode}")


def main():
    patch_files = sorted(glob.glob(os.path.join(PATCHES_DIR, "patch_*.npy")))
    print(f"Found {len(patch_files)} patches. Converting using mode='{MODE}'...")

    for i, patch_path in enumerate(patch_files):
        patch = np.load(patch_path)
        img_arr = patch_to_image(patch, mode=MODE)

        patch_name = Path(patch_path).stem
        out_path = os.path.join(OUTPUT_DIR, f"{patch_name}.png")

        Image.fromarray(img_arr).save(out_path)

        if (i + 1) % 500 == 0:
            print(f"  Converted {i+1}/{len(patch_files)}")

    print(f"\nDONE! {len(patch_files)} PNGs saved to: {OUTPUT_DIR}")
    print(f"Mode used: {MODE}")
    print("\nNext: upload this folder to Label Studio as a local image source,")
    print("or zip it and import via 'Upload Files'.")


if __name__ == "__main__":
    main()