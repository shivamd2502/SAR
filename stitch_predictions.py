"""
STITCH SEGMENTATION PREDICTIONS INTO FULL SCENE
===================================================
Runs the trained U-Net on every 256x256 patch belonging to a
chosen scene, then reassembles the individual predicted masks
back into one full-resolution segmentation map matching the
original scene's dimensions (e.g. 10301 x 10201).

Uses patches/patch_index.json (created by step2_patch_extraction.py)
to know exactly where each patch belongs (row_start, col_start, scene).

Input:
    ISRO_14/patches/patch_index.json       <- patch -> (scene, row, col) map
    ISRO_14/patches_png/patch_XXXXX.png    <- patch images (for the model)
    ISRO_14/processed/<scene>.npy          <- only used to get full scene H,W
    ISRO_14/model_seg/best_unet.pth        <- trained segmentation model

Output:
    ISRO_14/stitched/<scene>_full_mask.png       <- full-res class-id mask (0/1/2)
    ISRO_14/stitched/<scene>_full_mask_rgb.png   <- full-res colorized mask
    ISRO_14/stitched/<scene>_overlay.png         <- mask blended over original

Usage:
    python stitch_predictions.py
    python stitch_predictions.py --scene E04_SAR_MRS_03JUN2026_154005259612_23736_STUC00ZTD
"""

import os
import sys
import json
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.transforms as T

try:
    import segmentation_models_pytorch as smp
except ImportError:
    print("ERROR: segmentation_models_pytorch not installed.")
    print("Run: pip install segmentation-models-pytorch")
    sys.exit(1)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_DIR       = r"C:\Users\shiva\OneDrive\Documents\ISRO_14"
PATCHES_DIR    = os.path.join(BASE_DIR, "patches")
PATCHES_PNG    = os.path.join(BASE_DIR, "patches_png")
PROCESSED_DIR  = os.path.join(BASE_DIR, "processed")
MODEL_PATH     = os.path.join(BASE_DIR, "model_seg", "best_unet.pth")
OUT_DIR        = os.path.join(BASE_DIR, "stitched")
os.makedirs(OUT_DIR, exist_ok=True)

PATCH_SIZE  = 256
NUM_CLASSES = 3
CLASS_NAMES = ["Background", "Urban Dense", "Hills Rural"]
COLOR_MAP   = np.array([
    [30, 30, 30],     # 0 Background - dark gray (instead of pure black, easier to see scene edge)
    [220, 40, 40],     # 1 Urban Dense - red
    [40, 200, 90],     # 2 Hills Rural - green
], dtype=np.uint8)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
def load_model():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,   # weights come from checkpoint, not ImageNet, on load
        in_channels=3,
        classes=NUM_CLASSES,
    ).to(DEVICE)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Model loaded: {MODEL_PATH}")
    print(f"Checkpoint val_iou: {checkpoint.get('val_iou', 'n/a')}")
    return model


def get_transform():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


# ─────────────────────────────────────────────
# PREDICT ONE PATCH
# ─────────────────────────────────────────────
@torch.no_grad()
def predict_patch(model, transform, img_path):
    img = Image.open(img_path).convert("RGB")
    if img.size != (PATCH_SIZE, PATCH_SIZE):
        img = img.resize((PATCH_SIZE, PATCH_SIZE), Image.BILINEAR)
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    logits = model(tensor)
    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return pred


# ─────────────────────────────────────────────
# LOAD PATCH INDEX, FILTER BY SCENE
# ─────────────────────────────────────────────
def load_patch_index(scene_filter=None):
    index_path = os.path.join(PATCHES_DIR, "patch_index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"patch_index.json not found at {index_path}. "
            "This is created by step2_patch_extraction.py."
        )
    with open(index_path) as f:
        index = json.load(f)

    scenes_available = sorted(set(p["scene"] for p in index))
    print(f"\nScenes found in patch_index.json:")
    for s in scenes_available:
        n = sum(1 for p in index if p["scene"] == s)
        print(f"  {s}  ({n} patches)")

    if scene_filter is None:
        scene_filter = scenes_available[0]
        print(f"\nNo --scene given, defaulting to first scene: {scene_filter}")

    matched = [p for p in index if p["scene"] == scene_filter]

    if not matched:
        raise ValueError(f"No patches found for scene '{scene_filter}'. "
                          f"Available: {scenes_available}")

    print(f"\nUsing scene: {scene_filter}  ({len(matched)} patches)")
    return scene_filter, matched


def get_scene_dimensions(scene_name):
    """
    Get full (H, W) of the original scene from its processed .npy file.
    Falls back to inferring from max patch row/col + patch size if not found.
    """
    npy_path = os.path.join(PROCESSED_DIR, f"{scene_name}.npy")
    if os.path.exists(npy_path):
        arr = np.load(npy_path, mmap_mode="r")  # mmap = don't load full array into RAM
        H, W = arr.shape[0], arr.shape[1]
        print(f"Scene dimensions from {scene_name}.npy: {H} x {W}")
        return H, W
    return None  # caller will infer from patch index instead


# ─────────────────────────────────────────────
# STITCH
# ─────────────────────────────────────────────
def stitch_scene(scene_name, patches_meta, model, transform):
    # Determine canvas size
    dims = get_scene_dimensions(scene_name)
    if dims is None:
        max_row = max(p["row_start"] for p in patches_meta) + PATCH_SIZE
        max_col = max(p["col_start"] for p in patches_meta) + PATCH_SIZE
        H, W = max_row, max_col
        print(f"Scene dimensions inferred from patch grid: {H} x {W}")
    else:
        H, W = dims

    full_mask = np.zeros((H, W), dtype=np.uint8)
    # Track coverage so unfilled (no-data / edge) regions stay distinguishable
    covered = np.zeros((H, W), dtype=bool)

    print(f"\nRunning predictions and stitching {len(patches_meta)} patches...")
    for i, p in enumerate(patches_meta):
        patch_id = p["patch_id"]
        row, col = p["row_start"], p["col_start"]

        img_path = os.path.join(PATCHES_PNG, f"{patch_id}.png")
        if not os.path.exists(img_path):
            print(f"  [WARN] Missing PNG for {patch_id}, skipping")
            continue

        pred = predict_patch(model, transform, img_path)  # (256,256) uint8

        row_end = min(row + PATCH_SIZE, H)
        col_end = min(col + PATCH_SIZE, W)
        ph = row_end - row
        pw = col_end - col

        full_mask[row:row_end, col:col_end] = pred[:ph, :pw]
        covered[row:row_end, col:col_end] = True

        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(patches_meta)} patches placed...")

    n_uncovered = (~covered).sum()
    if n_uncovered > 0:
        pct = n_uncovered / covered.size * 100
        print(f"\n[INFO] {n_uncovered} pixels ({pct:.1f}%) not covered by any patch "
              f"(edges trimmed by non-overlapping stride extraction) — left as Background.")

    return full_mask, covered


# ─────────────────────────────────────────────
# SAVE OUTPUTS
# ─────────────────────────────────────────────
def save_outputs(scene_name, full_mask, covered):
    # 1. Raw class-id mask (for further processing / GIS tools)
    mask_path = os.path.join(OUT_DIR, f"{scene_name}_full_mask.png")
    Image.fromarray(full_mask).save(mask_path)
    print(f"\nSaved class-id mask: {mask_path}")

    # 2. Colorized RGB version (for visual inspection)
    rgb = COLOR_MAP[full_mask]
    rgb_path = os.path.join(OUT_DIR, f"{scene_name}_full_mask_rgb.png")
    Image.fromarray(rgb).save(rgb_path)
    print(f"Saved colorized mask: {rgb_path}")

    # 3. Downsampled preview (full-res files can be huge, e.g. 10301x10201)
    preview = Image.fromarray(rgb)
    preview.thumbnail((2000, 2000), Image.NEAREST)
    preview_path = os.path.join(OUT_DIR, f"{scene_name}_preview.png")
    preview.save(preview_path)
    print(f"Saved downsampled preview: {preview_path}")

    # 4. Class coverage stats
    print(f"\nClass coverage over stitched scene:")
    total = full_mask.size
    for cid, name in enumerate(CLASS_NAMES):
        count = (full_mask == cid).sum()
        print(f"  {name:15s}: {count:>12,} px  ({count/total*100:5.1f}%)")

    return rgb_path


def plot_overview(scene_name, rgb_path):
    rgb = np.array(Image.open(rgb_path))
    ds = max(1, rgb.shape[0] // 1200)
    rgb_small = rgb[::ds, ::ds]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb_small)
    ax.set_title(f"Stitched segmentation — {scene_name}\n"
                 f"(downsampled {ds}x for display)", fontsize=11)
    ax.axis("off")

    # Legend
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=np.array(COLOR_MAP[i]) / 255, label=CLASS_NAMES[i])
               for i in range(NUM_CLASSES)]
    ax.legend(handles=handles, loc="upper right", fontsize=9)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"{scene_name}_overview.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved overview figure: {out}")
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default=None,
                         help="Scene name to stitch (folder name under raw/, "
                              "matches the 'scene' field in patch_index.json). "
                              "If omitted, uses the first scene found.")
    args = parser.parse_args()

    print("=" * 60)
    print("STITCH SEGMENTATION PREDICTIONS INTO FULL SCENE")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    scene_name, patches_meta = load_patch_index(args.scene)

    print("\nLoading trained U-Net model...")
    model = load_model()
    transform = get_transform()

    full_mask, covered = stitch_scene(scene_name, patches_meta, model, transform)

    rgb_path = save_outputs(scene_name, full_mask, covered)

    print("\nGenerating overview plot...")
    plot_overview(scene_name, rgb_path)

    print("\n" + "=" * 60)
    print("DONE!")
    print(f"Outputs saved in: {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()