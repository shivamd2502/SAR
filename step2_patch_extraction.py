"""
STEP 2: PATCH EXTRACTION
=========================
Cuts each large processed SAR scene (10301 x 10201 pixels)
into small 256×256 patches suitable for ML/DL training.

Input:  ISRO_14/processed/*.npy   (output of step1_preprocess.py)
Output: ISRO_14/patches/
            ├── patch_0000.npy    ← shape (256, 256, 2)  [HH, HV channels]
            ├── patch_0001.npy
            ├── ...
            └── patch_index.json  ← maps patch_id → scene, row, col, stats
"""

import os
import numpy as np
import glob
import json
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_DIR     = r"C:\Users\shiva\OneDrive\Documents\ISRO_14"   # ← CHANGE THIS  ###C:\Users\shiva\OneDrive\Documents\ISRO_14
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
PATCHES_DIR   = os.path.join(BASE_DIR, "patches")
os.makedirs(PATCHES_DIR, exist_ok=True)

PATCH_SIZE   = 256     # pixels × pixels per patch
STRIDE       = 256     # set < PATCH_SIZE for overlapping patches (e.g. 128 = 50% overlap)
                        # use STRIDE = 256 for non-overlapping (faster, fewer patches)

# Skip patches that are mostly zeros (e.g., edge/nodata areas)
MIN_VALID_FRACTION = 0.90   # patch must have ≥ 90% non-zero pixels to be kept


def is_valid_patch(patch):
    """
    Returns True if patch has enough valid (non-zero, non-NaN) pixels.
    Rejects edge patches that are mostly empty.
    """
    valid_pixels = np.sum(np.isfinite(patch) & (patch != 0))
    total_pixels = patch.size
    return (valid_pixels / total_pixels) >= MIN_VALID_FRACTION


def normalize_patch(patch):
    """
    Normalize patch values to [0, 1] range per channel.
    
    For SAR dB values (typically -25 to +5 dB):
    We clip to a fixed range to ensure consistency across scenes.
    
    Channel 0 (HH): clip to [-25, 5] dB → normalize to [0, 1]
    Channel 1 (HV): clip to [-30, 0] dB → normalize to [0, 1]
    (HV is usually 5-8 dB lower than HH)
    """
    norm = np.zeros_like(patch, dtype=np.float32)

    # HH channel: typical range -25 to +5 dB
    db_min_hh, db_max_hh = -25.0, 5.0
    norm[:, :, 0] = (np.clip(patch[:, :, 0], db_min_hh, db_max_hh) - db_min_hh) / (db_max_hh - db_min_hh)

    # HV channel: typical range -30 to 0 dB
    db_min_hv, db_max_hv = -30.0, 0.0
    norm[:, :, 1] = (np.clip(patch[:, :, 1], db_min_hv, db_max_hv) - db_min_hv) / (db_max_hv - db_min_hv)

    return norm


def extract_patches_from_scene(npy_path, patch_size, stride):
    """
    Extract all valid patches from one scene .npy file.
    Returns list of (patch_array, row_start, col_start) tuples.
    """
    scene = np.load(npy_path)   # shape: (H, W, 2)
    H, W, C = scene.shape
    print(f"  Scene shape: {H} x {W} x {C} channels")

    patches = []
    total_possible = 0
    skipped = 0

    for row in range(0, H - patch_size + 1, stride):
        for col in range(0, W - patch_size + 1, stride):
            total_possible += 1
            patch_raw = scene[row:row+patch_size, col:col+patch_size, :]  # (256, 256, 2)

            # Skip invalid patches (edge areas, nodata)
            if not is_valid_patch(patch_raw):
                skipped += 1
                continue

            # Normalize to [0, 1]
            patch_norm = normalize_patch(patch_raw)
            patches.append((patch_norm, row, col))

    print(f"  Patches: {len(patches)} valid / {total_possible} total ({skipped} skipped as invalid)")
    return patches


def main():
    # Load all processed .npy files
    npy_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "*.npy")))

    if not npy_files:
        print(f"ERROR: No .npy files found in {PROCESSED_DIR}")
        print("Run step1_preprocess.py first!")
        return

    print(f"Found {len(npy_files)} processed scene(s):")
    for f in npy_files:
        print(f"  - {os.path.basename(f)}")

    patch_index = []   # will store metadata for each saved patch
    patch_counter = 0

    for npy_path in npy_files:
        scene_name = Path(npy_path).stem
        print(f"\n{'='*60}")
        print(f"Extracting patches from: {scene_name}")
        print(f"{'='*60}")

        patches = extract_patches_from_scene(npy_path, PATCH_SIZE, STRIDE)

        for patch_norm, row, col in patches:
            patch_id = f"patch_{patch_counter:05d}"
            out_path = os.path.join(PATCHES_DIR, f"{patch_id}.npy")

            np.save(out_path, patch_norm)

            # Record index entry
            patch_index.append({
                "patch_id": patch_id,
                "scene": scene_name,
                "row_start": int(row),
                "col_start": int(col),
                "shape": list(patch_norm.shape),
                "HH_mean": float(patch_norm[:, :, 0].mean()),
                "HV_mean": float(patch_norm[:, :, 1].mean()),
            })
            patch_counter += 1

        print(f"  → Saved {len(patches)} patches from this scene.")

    # Save the patch index
    index_path = os.path.join(PATCHES_DIR, "patch_index.json")
    with open(index_path, "w") as f:
        json.dump(patch_index, f, indent=2)

    print(f"\n{'='*60}")
    print(f"DONE! Total patches saved: {patch_counter}")
    print(f"Patches directory: {PATCHES_DIR}")
    print(f"Patch index: {index_path}")
    print(f"\nEach patch shape: ({PATCH_SIZE}, {PATCH_SIZE}, 2)")
    print(f"  Channel 0 = HH (normalized 0–1)")
    print(f"  Channel 1 = HV (normalized 0–1)")
    print(f"\nNext step: Run step3_visualize.py to visually inspect patches")


if __name__ == "__main__":
    main()