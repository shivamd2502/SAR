"""
STEP 3: VISUALIZE & VERIFY
============================
Visually inspect processed SAR patches to confirm the 
pipeline worked correctly before moving to ML.

Run this AFTER step1 and step2.

What this does:
  1. Plots a full processed scene (HH and HV side by side)
  2. Plots a random grid of 16 patches 
  3. Plots the pixel value distribution (histogram)
  4. Creates a false-color composite (useful for human interpretation)
"""

import os
import numpy as np
import glob
import json
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_DIR      = r"C:\Users\shiva\OneDrive\Documents\ISRO_14"   # ← CHANGE THIS
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
PATCHES_DIR   = os.path.join(BASE_DIR, "patches")

# How many sample patches to show in the grid
N_SAMPLE_PATCHES = 16


# ─────────────────────────────────────────────
# VISUALIZATION 1: Full scene overview
# ─────────────────────────────────────────────
def plot_scene_overview(npy_path, downsample=10):
    """
    Plot HH and HV channels of a full scene side by side.
    downsample=10 means show every 10th pixel (for speed with 10k×10k images)
    """
    scene = np.load(npy_path)  # (H, W, 2)
    scene_ds = scene[::downsample, ::downsample, :]  # downsample for display

    scene_name = Path(npy_path).stem

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Scene Overview: {scene_name[:60]}", fontsize=11)

    # HH channel (dB scale - values before normalization, so load from scene)
    im0 = axes[0].imshow(scene_ds[:, :, 0], cmap="gray", vmin=-25, vmax=5)
    axes[0].set_title("HH Channel (dB)")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], label="σ⁰ (dB)")

    # HV channel
    im1 = axes[1].imshow(scene_ds[:, :, 1], cmap="gray", vmin=-30, vmax=0)
    axes[1].set_title("HV Channel (dB)")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], label="σ⁰ (dB)")

    # False color composite
    # R = HH, G = HV, B = HH-HV ratio (highlights vegetation vs urban)
    r = (np.clip(scene_ds[:, :, 0], -25, 5) - (-25)) / 30.0      # HH normalized
    g = (np.clip(scene_ds[:, :, 1], -30, 0) - (-30)) / 30.0      # HV normalized
    b = np.clip((r - g + 1) / 2, 0, 1)                            # HH-HV difference
    rgb = np.stack([r, g, b], axis=-1)

    axes[2].imshow(rgb)
    axes[2].set_title("False Color Composite\n(R=HH, G=HV, B=ratio)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(PROCESSED_DIR, f"{scene_name}_overview.png"), dpi=150, bbox_inches="tight")
    print(f"  Saved overview: {scene_name}_overview.png")
    plt.show()


# ─────────────────────────────────────────────
# VISUALIZATION 2: Random patch grid
# ─────────────────────────────────────────────
def plot_patch_grid(patches_dir, n=16):
    """
    Load n random patches and show them in a grid.
    Each patch shows HH (left) and HV (right).
    """
    all_patches = sorted(glob.glob(os.path.join(patches_dir, "patch_*.npy")))
    if not all_patches:
        print("No patches found! Run step2_patch_extraction.py first.")
        return

    sample = random.sample(all_patches, min(n, len(all_patches)))
    cols = 8
    rows = (len(sample) * 2 + cols - 1) // cols  # 2 images per patch (HH + HV)

    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 2.5))
    fig.suptitle(f"Random Sample of {len(sample)} Patches (HH | HV per pair)", fontsize=13)

    axes_flat = axes.flatten()
    img_idx = 0

    for patch_path in sample:
        patch = np.load(patch_path)  # (256, 256, 2)
        patch_name = Path(patch_path).stem

        # HH
        if img_idx < len(axes_flat):
            axes_flat[img_idx].imshow(patch[:, :, 0], cmap="gray", vmin=0, vmax=1)
            axes_flat[img_idx].set_title(f"{patch_name}\nHH", fontsize=7)
            axes_flat[img_idx].axis("off")
            img_idx += 1

        # HV
        if img_idx < len(axes_flat):
            axes_flat[img_idx].imshow(patch[:, :, 1], cmap="gray", vmin=0, vmax=1)
            axes_flat[img_idx].set_title(f"HV", fontsize=7)
            axes_flat[img_idx].axis("off")
            img_idx += 1

    # Hide unused axes
    for j in range(img_idx, len(axes_flat)):
        axes_flat[j].axis("off")

    plt.tight_layout()
    out_path = os.path.join(patches_dir, "patch_sample_grid.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"  Saved patch grid: {out_path}")
    plt.show()


# ─────────────────────────────────────────────
# VISUALIZATION 3: Histogram of pixel values
# ─────────────────────────────────────────────
def plot_histogram(patches_dir, n_patches=200):
    """
    Sample pixel values from n random patches and plot histograms.
    Good way to confirm normalization is working correctly.
    Expected: values concentrated in [0, 1] range.
    """
    all_patches = sorted(glob.glob(os.path.join(patches_dir, "patch_*.npy")))
    sample = random.sample(all_patches, min(n_patches, len(all_patches)))

    hh_vals = []
    hv_vals = []

    for p in sample:
        patch = np.load(p)
        # Sample 100 random pixels per patch (for speed)
        h, w = patch.shape[:2]
        ri = np.random.randint(0, h, 100)
        ci = np.random.randint(0, w, 100)
        hh_vals.extend(patch[ri, ci, 0].tolist())
        hv_vals.extend(patch[ri, ci, 1].tolist())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Pixel Value Distributions (from {len(sample)} patches)", fontsize=13)

    axes[0].hist(hh_vals, bins=100, color="steelblue", alpha=0.8, edgecolor="none")
    axes[0].set_title("HH Channel (normalized)")
    axes[0].set_xlabel("Pixel value (0=min dB, 1=max dB)")
    axes[0].set_ylabel("Count")
    axes[0].axvline(np.mean(hh_vals), color="red", linestyle="--", label=f"Mean={np.mean(hh_vals):.3f}")
    axes[0].legend()

    axes[1].hist(hv_vals, bins=100, color="darkorange", alpha=0.8, edgecolor="none")
    axes[1].set_title("HV Channel (normalized)")
    axes[1].set_xlabel("Pixel value (0=min dB, 1=max dB)")
    axes[1].set_ylabel("Count")
    axes[1].axvline(np.mean(hv_vals), color="red", linestyle="--", label=f"Mean={np.mean(hv_vals):.3f}")
    axes[1].legend()

    plt.tight_layout()
    out_path = os.path.join(patches_dir, "pixel_distribution.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"  Saved histogram: {out_path}")
    plt.show()


# ─────────────────────────────────────────────
# VISUALIZATION 4: Patch index summary
# ─────────────────────────────────────────────
def print_dataset_summary(patches_dir):
    """
    Print a summary of all patches: counts per scene, total patches.
    """
    index_path = os.path.join(patches_dir, "patch_index.json")
    if not os.path.exists(index_path):
        print("patch_index.json not found. Run step2_patch_extraction.py first.")
        return

    with open(index_path) as f:
        index = json.load(f)

    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"Total patches: {len(index)}")

    # Count per scene
    from collections import Counter
    scene_counts = Counter(p["scene"] for p in index)
    print("\nPatches per scene:")
    for scene, count in sorted(scene_counts.items()):
        print(f"  {scene[:55]:55s} → {count:5d} patches")

    # Overall stats
    hh_means = [p["HH_mean"] for p in index]
    hv_means = [p["HV_mean"] for p in index]
    print(f"\nHH mean across all patches: {np.mean(hh_means):.4f} ± {np.std(hh_means):.4f}")
    print(f"HV mean across all patches: {np.mean(hv_means):.4f} ± {np.std(hv_means):.4f}")
    print(f"\nPatch shape: 256 × 256 × 2 channels")
    print(f"Storage per patch: ~{256*256*2*4 / 1024:.0f} KB")
    print(f"Total storage (approx): ~{len(index) * 256*256*2*4 / 1024 / 1024:.1f} MB")


def main():
    print("="*60)
    print("STEP 3: VISUALIZATION & VERIFICATION")
    print("="*60)

    # --- Scene overviews ---
    npy_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "*.npy")))
    if npy_files:
        print(f"\n[1] Plotting scene overviews for {len(npy_files)} scene(s)...")
        for npy_path in npy_files[:2]:  # limit to first 2 scenes to save time
            plot_scene_overview(npy_path, downsample=10)
    else:
        print("No processed .npy files found. Run step1 first.")

    # --- Patch grid ---
    print("\n[2] Plotting random patch grid...")
    plot_patch_grid(PATCHES_DIR, n=N_SAMPLE_PATCHES)

    # --- Histograms ---
    print("\n[3] Plotting pixel value histograms...")
    plot_histogram(PATCHES_DIR, n_patches=200)

    # --- Summary ---
    print("\n[4] Dataset summary:")
    print_dataset_summary(PATCHES_DIR)

    print("\n" + "="*60)
    print("Verification complete!")
    print("If the images look reasonable, your dataset is ready for ML.")
    print("Next step: Decide on classification vs segmentation, then label data.")
    print("="*60)


if __name__ == "__main__":
    main()