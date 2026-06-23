"""
INSPECT _lia AND _mask TIF FILES
====================================
Quick diagnostic script to understand what's actually inside the
Local Incidence Angle (_lia.tif) and quality mask (_mask.tif) files
before integrating them into the preprocessing pipeline.

Prints:
    - band count, dtype, shape
    - value range / unique values
    - histogram of values
    - visual preview (downsampled)

Usage:
    python inspect_lia_mask.py
"""

import os
import glob
import numpy as np
import rasterio
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
BASE_DIR = r"C:\Users\shiva\OneDrive\Documents\ISRO_14"
RAW_DIR  = os.path.join(BASE_DIR, "raw")
OUT_DIR  = os.path.join(BASE_DIR, "inspect_lia_mask")
os.makedirs(OUT_DIR, exist_ok=True)


def inspect_tif(tif_path, label, downsample=10):
    print(f"\n{'='*60}")
    print(f"Inspecting: {label}")
    print(f"Path: {tif_path}")
    print(f"{'='*60}")

    with rasterio.open(tif_path) as src:
        print(f"Band count : {src.count}")
        print(f"Dtype      : {src.dtypes}")
        print(f"Shape      : {src.height} x {src.width}")
        print(f"CRS        : {src.crs}")
        print(f"Nodata     : {src.nodata}")

        # Read first band (downsampled for speed on large files)
        data = src.read(
            1,
            out_shape=(src.height // downsample, src.width // downsample)
        ).astype(np.float64)

    valid = data[np.isfinite(data)]
    print(f"\nValue stats (downsampled {downsample}x for speed):")
    print(f"  min    : {valid.min():.4f}")
    print(f"  max    : {valid.max():.4f}")
    print(f"  mean   : {valid.mean():.4f}")
    print(f"  median : {np.median(valid):.4f}")
    print(f"  std    : {valid.std():.4f}")

    unique_vals = np.unique(valid)
    n_unique = len(unique_vals)
    print(f"\nUnique values: {n_unique}")
    if n_unique <= 30:
        # Likely categorical (e.g. mask flags) — show counts
        print("Looks CATEGORICAL — value counts:")
        for v in unique_vals:
            count = (valid == v).sum()
            pct = count / valid.size * 100
            print(f"  value={v:<10g}  count={count:<10d}  ({pct:.1f}%)")
    else:
        print("Looks CONTINUOUS (e.g. an angle in degrees) — showing histogram.")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(label, fontsize=13)

    im = axes[0].imshow(data, cmap="viridis")
    axes[0].set_title(f"Spatial preview (downsampled {downsample}x)")
    axes[0].axis("off")
    plt.colorbar(im, ax=axes[0], shrink=0.7)

    axes[1].hist(valid.ravel(), bins=50, color="steelblue", edgecolor="none")
    axes[1].set_title("Value distribution")
    axes[1].set_xlabel("Pixel value")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    safe_label = label.replace(" ", "_").replace("/", "_")
    out_path = os.path.join(OUT_DIR, f"{safe_label}.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"\nSaved preview: {out_path}")
    plt.show()


def find_files(scene_folder):
    lia_candidates = glob.glob(os.path.join(scene_folder, "*_lia*"))
    mask_candidates = glob.glob(os.path.join(scene_folder, "*_mask*"))
    return lia_candidates, mask_candidates


def main():
    scene_folders = [d for d in glob.glob(os.path.join(RAW_DIR, "E04_SAR_*")) if os.path.isdir(d)]

    if not scene_folders:
        print(f"No scene folders found under {RAW_DIR}")
        return

    print(f"Found {len(scene_folders)} scene folder(s).")

    for scene_folder in scene_folders:
        scene_name = os.path.basename(scene_folder)
        print(f"\n\n{'#'*60}")
        print(f"# SCENE: {scene_name}")
        print(f"{'#'*60}")

        lia_files, mask_files = find_files(scene_folder)

        if lia_files:
            inspect_tif(lia_files[0], f"{scene_name[:30]}_LIA")
        else:
            print(f"\n[INFO] No _lia file found in {scene_folder}")

        if mask_files:
            inspect_tif(mask_files[0], f"{scene_name[:30]}_MASK")
        else:
            print(f"\n[INFO] No _mask file found in {scene_folder}")

    print(f"\n\nAll previews saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()