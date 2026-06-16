"""
VISUAL CHECK — VERIFY MASKS LOOK CORRECT
===========================================
Overlays generated masks on top of original patches
so you can visually confirm the painted regions match
what you actually drew in Label Studio.
"""

import os
import numpy as np
import glob
import matplotlib.pyplot as plt
from PIL import Image

BASE_DIR     = r"C:\Users\shiva\OneDrive\Documents\ISRO_14"
SEG_IMG_DIR  = os.path.join(BASE_DIR, "seg_to_label")
MASKS_DIR    = os.path.join(BASE_DIR, "masks_seg")

N_SAMPLES = 9

# Color map: 0=background(black), 1=Urban Dense(red), 2=Hills Rural(green)
COLOR_MAP = {
    0: [0, 0, 0],
    1: [255, 0, 0],
    2: [0, 255, 0],
}


def mask_to_rgb(mask):
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in COLOR_MAP.items():
        rgb[mask == class_id] = color
    return rgb


def main():
    mask_files = sorted(glob.glob(os.path.join(MASKS_DIR, "*_mask.png")))
    sample = mask_files[:N_SAMPLES]

    fig, axes = plt.subplots(3, 6, figsize=(20, 10))
    fig.suptitle("Original (left) vs Mask Overlay (right) — Red=Urban, Green=Hills", fontsize=13)

    for i, mask_path in enumerate(sample):
        mask_name = os.path.basename(mask_path)
        img_name = mask_name.replace("_mask.png", ".png")
        img_path = os.path.join(SEG_IMG_DIR, img_name)

        if not os.path.exists(img_path):
            continue

        orig = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))
        mask_rgb = mask_to_rgb(mask)

        # Overlay: blend mask color on top of original
        overlay = (orig * 0.5 + mask_rgb * 0.5).astype(np.uint8)

        row = i // 3
        col = (i % 3) * 2

        axes[row, col].imshow(orig)
        axes[row, col].set_title(img_name, fontsize=8)
        axes[row, col].axis("off")

        axes[row, col + 1].imshow(overlay)
        axes[row, col + 1].set_title("Mask overlay", fontsize=8)
        axes[row, col + 1].axis("off")

    plt.tight_layout()
    out_path = os.path.join(MASKS_DIR, "mask_verification.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()