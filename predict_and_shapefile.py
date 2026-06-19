"""
PREDICT CLASSES ON FULL SCENE + GENERATE QGIS SHAPEFILE
===========================================================
Runs the trained EfficientNet-B0 classifier on EVERY patch of a
chosen scene (not just labeled ones), then writes the predicted
class for each patch into a georeferenced shapefile for QGIS.

This is different from generate_patch_shapefile.py, which only
showed your manually-assigned ground-truth labels. This script
shows what the MODEL predicts across the entire scene, including
the ~2900 patches you never manually labeled.

Input:
    ISRO_14/model/best_model.pth           <- trained classifier
    ISRO_14/model/label_encoder.json       <- class name <-> id mapping
    ISRO_14/patches_png/patch_XXXXX.png    <- patch images
    ISRO_14/patches/patch_index.json       <- row_start/col_start per patch
    ISRO_14/raw/<scene_folder>/hh.tif      <- for CRS + geotransform

Output:
    ISRO_14/qgis/predicted_<scene>.shp  (+ .shx .dbf .prj)
        One polygon per patch, attributes:
            patch_id, scene, row_start, col_start,
            pred_label, confidence, prob_none, prob_urban, prob_hills

Usage:
    python predict_and_shapefile.py
    python predict_and_shapefile.py --scene E04_SAR_MRS_03JUN2026_154005259612_23736_STUC00ZTD
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

try:
    import rasterio
except ImportError:
    print("ERROR: rasterio not installed. Run: pip install rasterio")
    sys.exit(1)

try:
    import geopandas as gpd
    from shapely.geometry import Polygon
except ImportError:
    print("ERROR: geopandas/shapely not installed.")
    print("Run: pip install geopandas shapely fiona pyproj")
    sys.exit(1)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_DIR     = r"C:\Users\shiva\OneDrive\Documents\ISRO_14"
RAW_DIR      = os.path.join(BASE_DIR, "raw")
PATCHES_DIR  = os.path.join(BASE_DIR, "patches")
PATCHES_PNG  = os.path.join(BASE_DIR, "patches_png")
MODEL_DIR    = os.path.join(BASE_DIR, "model")
MODEL_PATH   = os.path.join(MODEL_DIR, "best_model.pth")
LABEL_PATH   = os.path.join(MODEL_DIR, "label_encoder.json")
QGIS_DIR     = os.path.join(BASE_DIR, "qgis")
os.makedirs(QGIS_DIR, exist_ok=True)

PATCH_SIZE = 256
IMG_SIZE   = 224   # classifier's expected input size

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ─────────────────────────────────────────────
# LOAD CLASSIFIER MODEL
# ─────────────────────────────────────────────
def load_model(num_classes):
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes)
    )
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    print(f"Model loaded: {MODEL_PATH}")
    print(f"Training val accuracy: {checkpoint.get('val_acc', 'n/a')}")
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


@torch.no_grad()
def predict_patch(model, transform, img_path, class_names):
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(probs.argmax())
    return class_names[pred_idx], probs


# ─────────────────────────────────────────────
# LOAD PATCH INDEX FOR A SCENE
# ─────────────────────────────────────────────
def load_patch_index(scene_filter=None):
    index_path = os.path.join(PATCHES_DIR, "patch_index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"patch_index.json not found at {index_path}")

    with open(index_path) as f:
        index = json.load(f)

    scenes_available = sorted(set(p["scene"] for p in index))
    print("\nScenes found in patch_index.json:")
    for s in scenes_available:
        n = sum(1 for p in index if p["scene"] == s)
        print(f"  {s}  ({n} patches)")

    if scene_filter is None:
        scene_filter = scenes_available[0]
        print(f"\nNo --scene given, defaulting to: {scene_filter}")

    matched = [p for p in index if p["scene"] == scene_filter]
    if not matched:
        raise ValueError(f"No patches found for scene '{scene_filter}'. "
                          f"Available: {scenes_available}")

    print(f"\nUsing scene: {scene_filter}  ({len(matched)} patches)")
    return scene_filter, matched


# ─────────────────────────────────────────────
# FIND ORIGINAL TIF (for CRS + transform)
# ─────────────────────────────────────────────
def find_scene_tif(scene_name):
    candidates = glob.glob(os.path.join(RAW_DIR, scene_name + "*"))
    if not candidates:
        all_folders = [d for d in glob.glob(os.path.join(RAW_DIR, "*")) if os.path.isdir(d)]
        candidates = [d for d in all_folders if os.path.basename(d).startswith(scene_name[:40])]
    if not candidates:
        raise FileNotFoundError(f"Could not find raw scene folder for '{scene_name}' under {RAW_DIR}")

    scene_folder = candidates[0]
    hh_path = os.path.join(scene_folder, "hh.tif")
    if not os.path.exists(hh_path):
        raise FileNotFoundError(f"hh.tif not found in {scene_folder}")
    return hh_path


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default=None,
                         help="Scene name to predict + shapefile (matches "
                              "patch_index.json 'scene' field). Defaults to "
                              "first scene found if omitted.")
    args = parser.parse_args()

    print("=" * 60)
    print("PREDICT CLASSES ON FULL SCENE + GENERATE SHAPEFILE")
    print("=" * 60)

    # Load label encoder
    with open(LABEL_PATH) as f:
        label_enc = json.load(f)
    class_names = list(label_enc.keys())
    print(f"\nClasses: {class_names}")

    # Load model + transform
    model = load_model(len(class_names))
    transform = get_transform()

    # Load patch index for chosen scene
    scene_name, patches_meta = load_patch_index(args.scene)

    # Find GeoTIFF for CRS/transform
    tif_path = find_scene_tif(scene_name)
    print(f"\nUsing GeoTIFF for CRS/transform: {tif_path}")

    with rasterio.open(tif_path) as src:
        geo_transform = src.transform
        crs = src.crs
        print(f"CRS: {crs}")

        print(f"\nRunning predictions on {len(patches_meta)} patches...")
        records = []
        class_counts = {c: 0 for c in class_names}

        for i, p in enumerate(patches_meta):
            patch_id = p["patch_id"]
            row, col = p["row_start"], p["col_start"]

            img_path = os.path.join(PATCHES_PNG, f"{patch_id}.png")
            if not os.path.exists(img_path):
                print(f"  [WARN] Missing PNG for {patch_id}, skipping")
                continue

            pred_label, probs = predict_patch(model, transform, img_path, class_names)
            class_counts[pred_label] += 1

            # Pixel corners -> map coordinates (same logic as generate_patch_shapefile.py)
            r0, c0 = row, col
            r1, c1 = row + PATCH_SIZE, col + PATCH_SIZE

            x_ul, y_ul = geo_transform * (c0, r0)
            x_ur, y_ur = geo_transform * (c1, r0)
            x_lr, y_lr = geo_transform * (c1, r1)
            x_ll, y_ll = geo_transform * (c0, r1)

            poly = Polygon([(x_ul, y_ul), (x_ur, y_ur), (x_lr, y_lr), (x_ll, y_ll)])
            x_center, y_center = geo_transform * (c0 + PATCH_SIZE / 2, r0 + PATCH_SIZE / 2)

            record = {
                "patch_id": patch_id,
                "scene": scene_name[:40],
                "row_start": row,
                "col_start": col,
                "pred_label": pred_label,
                "confidence": float(probs.max()),
                "x_center": x_center,
                "y_center": y_center,
                "geometry": poly,
            }
            # Add per-class probability columns (shapefile field names max 10 chars)
            for j, cname in enumerate(class_names):
                short_name = "p_" + "".join(ch for ch in cname if ch.isalnum())[:7]
                record[short_name] = float(probs[j])

            records.append(record)

            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(patches_meta)} patches predicted...")

    print(f"\nPrediction summary:")
    total = sum(class_counts.values())
    for cname, count in class_counts.items():
        pct = count / total * 100 if total else 0
        print(f"  {cname:15s}: {count:5d} patches ({pct:5.1f}%)")

    # Build GeoDataFrame and reproject to WGS84 for QGIS/Google Satellite overlay
    gdf = gpd.GeoDataFrame(records, crs=crs)
    gdf_wgs84 = gdf.to_crs(epsg=4326)

    safe_scene_name = "".join(c if c.isalnum() else "_" for c in scene_name[:40])
    out_path = os.path.join(QGIS_DIR, f"predicted_{safe_scene_name}.shp")
    gdf_wgs84.to_file(out_path)

    print(f"\n{'='*60}")
    print(f"DONE!")
    print(f"Shapefile saved: {out_path}")
    print(f"Total patches: {len(gdf_wgs84)}")
    print(f"\nOpen this .shp file in QGIS, then Symbology -> Categorized -> ")
    print(f"Value: 'pred_label' -> Classify, to color-code by predicted class.")
    print(f"Field 'confidence' shows the model's certainty (0-1) for each patch.")
    print("=" * 60)


if __name__ == "__main__":
    main()