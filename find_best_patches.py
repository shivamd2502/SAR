"""
FIND BEST PATCHES TO LABEL FOR SEGMENTATION
=============================================
Scores all patches by visual content richness and
copies the top N most informative ones into a
separate folder for Label Studio upload.

Logic:
- High variance = more texture/detail = more interesting to label
- High mean = brighter = more likely to contain urban/structure
- We want patches that are NOT all black (empty) and NOT all uniform
"""

import os
import numpy as np
import glob
import shutil
from PIL import Image

# ─────────────────────────────────────────────
BASE_DIR       = r"C:\Users\shiva\OneDrive\Documents\ISRO_14"
PATCHES_PNG    = os.path.join(BASE_DIR, "patches_png")
SEG_LABEL_DIR  = os.path.join(BASE_DIR, "seg_to_label")   # folder to upload to Label Studio
N_TO_SELECT    = 50   # how many patches to select for labeling
os.makedirs(SEG_LABEL_DIR, exist_ok=True)


def score_patch(img_path):
    """
    Score a patch by how informative it is for labeling.
    Higher score = more worth labeling.
    """
    img = np.array(Image.open(img_path).convert("RGB")).astype(np.float32)

    # Variance across all pixels — high variance = more texture/detail
    variance = img.var()

    # Fraction of bright pixels (likely urban/structure)
    bright_fraction = (img.mean(axis=2) > 100).mean()

    # Penalize patches that are mostly black (no-data border patches)
    black_fraction = (img.mean(axis=2) < 5).mean()
    if black_fraction > 0.3:
        return -1  # skip mostly-black patches

    # Combined score: reward variance and some brightness
    score = variance * 0.7 + bright_fraction * 1000 * 0.3
    return score


def main():
    all_patches = sorted(glob.glob(os.path.join(PATCHES_PNG, "patch_*.png")))
    print(f"Found {len(all_patches)} patches. Scoring...")

    scores = []
    for i, p in enumerate(all_patches):
        score = score_patch(p)
        scores.append((score, p))
        if (i + 1) % 500 == 0:
            print(f"  Scored {i+1}/{len(all_patches)}")

    # Sort by score descending, pick top N
    scores.sort(key=lambda x: x[0], reverse=True)
    selected = scores[:N_TO_SELECT]

    print(f"\nTop {N_TO_SELECT} most informative patches selected.")
    print(f"Copying to: {SEG_LABEL_DIR}")

    for score, src_path in selected:
        fname = os.path.basename(src_path)
        dst_path = os.path.join(SEG_LABEL_DIR, fname)
        shutil.copy2(src_path, dst_path)

    print(f"\nDONE! {N_TO_SELECT} patches copied to {SEG_LABEL_DIR}")
    print(f"\nNext step:")
    print(f"  1. Open Label Studio → your segmentation project")
    print(f"  2. Import files from: {SEG_LABEL_DIR}")
    print(f"  3. Paint Urban Dense and Hills Rural regions on each patch")
    print(f"  4. Export as JSON when done")


if __name__ == "__main__":
    main()