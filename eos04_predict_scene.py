#!/usr/bin/env python3
"""
eos04_predict_scene.py
======================
Runs the trained EOS04Classifier on the COMPLETE OPT1_sigma0 GeoTIFF scene.

What this script does
---------------------
1.  Loads the trained model  (best_model.pth)
2.  Loads feature_stats.json (for SAR feature normalisation — same as training)
3.  Loads label_encoder.json (class ID → class name mapping)
4.  Loads the full OPT1_sigma0 composite GeoTIFF  (float32, 3-band)
5.  Cuts the scene into 256×256 patches on the SAME grid used during training
    PLUS handles edge patches (padded with zeros) so every pixel is covered
6.  For each patch:
      a. Computes SAR features on-the-fly (mean_r, mean_g, hh_hv_rat, rvi_mean)
         from the patch pixel values  — same features used during training
      b. Normalises them using feature_stats.json
      c. Runs the model → predicted class ID + confidence score
7.  Stitches all patch predictions back into a full-scene classification map
8.  Saves four output files:
      prediction_classmap.tif    ← GeoTIFF: each pixel = class ID (uint8)
                                    georeferenced, open directly in QGIS
      prediction_confidence.tif  ← GeoTIFF: each pixel = max softmax prob (float32)
      prediction_rgb.png         ← colourised PNG for quick visual check
      prediction_report.csv      ← per-patch: patch_id, class, confidence, row, col

Colour coding in prediction_rgb.png
-------------------------------------
  Dense_forest  →  dark green   (0, 128, 0)
  Mix           →  yellow       (255, 200, 0)
  Urban         →  red          (220, 50, 50)
  Unclear       →  grey         (150, 150, 150)
  No prediction →  black        (0, 0, 0)

Usage
-----
    python eos04_predict_scene.py

    # or override paths explicitly
    python eos04_predict_scene.py ^
        --composite-tif "composites\03JUN2026\OPT1_R-HH-sigma0_G-HV-sigma0_B-ratio-sigma0.tif" ^
        --model-dir     "model\EOS04" ^
        --output-dir    "predictions\03JUN2026" ^
        --patch-size 256 --batch-size 32

Requirements
------------
    pip install torch torchvision rasterio Pillow numpy pandas tqdm
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("predict")

try:
    import rasterio
    from rasterio.transform import rowcol
except ImportError:
    print("ERROR: pip install rasterio"); sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw): return x   # graceful fallback if tqdm not installed


# =============================================================================
# PATHS  — edit if needed
# =============================================================================
BASE_DIR       = r"D:\ISRO_14"
COMPOSITE_TIF  = os.path.join(
    BASE_DIR, "composites", "03JUN2026",
    "OPT1_R-HH-sigma0_G-HV-sigma0_B-ratio-sigma0.tif")
MODEL_DIR      = os.path.join(BASE_DIR, "model", "EOS04")
OUTPUT_DIR     = os.path.join(BASE_DIR, "predictions", "03JUN2026")

PATCH_SIZE     = 256
BATCH_SIZE     = 32
IMG_SIZE       = 224        # model input size (resize from 256)
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Colour map for visualisation  class_id → (R, G, B)
COLOUR_MAP = {
    0: (0,   128,  0),    # Dense_forest  dark green
    1: (255, 200,  0),    # Mix           yellow
    2: (220,  50, 50),    # Urban         red
    3: (150, 150,150),    # Unclear       grey
    -1:(0,     0,  0),    # no prediction black (edge / outside scene)
}

SAR_FEATURE_COLS = ["mean_r", "mean_g", "hh_hv_rat", "rvi_mean"]


# =============================================================================
# 1.  RE-DEFINE MODEL  (must match training script exactly)
# =============================================================================

class EOS04Classifier(nn.Module):
    """Exact copy of the model definition from eos04_train.py."""
    def __init__(self, num_classes: int, num_sar_features: int = 4):
        super().__init__()
        backbone = models.efficientnet_b0(weights=None)   # no pretrain at inference
        for param in backbone.parameters():
            param.requires_grad = False
        self.cnn_features = backbone.features
        self.pool         = backbone.avgpool
        self.cnn_dim      = backbone.classifier[1].in_features   # 1280
        self._phase2_params = list(backbone.features[-2:].parameters())

        self.sar_branch = nn.Sequential(
            nn.Linear(num_sar_features, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )
        self.sar_dim = 32

        fusion_dim = self.cnn_dim + self.sar_dim
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, images, sar_features):
        x = self.cnn_features(images)
        x = self.pool(x)
        x = x.flatten(1)
        s = self.sar_branch(sar_features)
        return self.classifier(torch.cat([x, s], dim=1))

    def unfreeze_for_phase2(self):
        for param in self._phase2_params:
            param.requires_grad = True


# =============================================================================
# 2.  LOAD MODEL + METADATA
# =============================================================================

def load_model(model_dir: str):
    """Loads best_model.pth, label_encoder.json and feature_stats.json."""

    # label encoder
    enc_path = os.path.join(model_dir, "label_encoder.json")
    with open(enc_path) as f:
        label_enc = json.load(f)            # {"Dense_forest":0, "Mix":1, ...}
    id_to_class = {v: k for k, v in label_enc.items()}
    num_classes  = len(label_enc)
    log.info("Classes: %s", label_enc)

    # feature stats (z-score normalisation)
    stats_path = os.path.join(model_dir, "feature_stats.json")
    with open(stats_path) as f:
        feat_stats = json.load(f)
    log.info("Feature stats loaded from %s", stats_path)

    # model weights
    ckpt_path = os.path.join(model_dir, "best_model.pth")
    ckpt      = torch.load(ckpt_path, map_location=DEVICE)
    model     = EOS04Classifier(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    log.info("Model loaded from %s  (val_acc=%.3f at epoch %d)",
             ckpt_path, ckpt.get("val_acc", 0), ckpt.get("epoch", 0))

    return model, id_to_class, feat_stats


# =============================================================================
# 3.  IMAGE TRANSFORM  (val transform — no augmentation)
# =============================================================================

def get_val_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std= [0.229, 0.224, 0.225]),
    ])


# =============================================================================
# 4.  SAR FEATURE EXTRACTION FROM A PATCH
#     Same 4 features as training: mean_r, mean_g, hh_hv_rat, rvi_mean
#     Computed directly from the float32 GeoTIFF patch values.
# =============================================================================

def extract_sar_features(patch_float32: np.ndarray) -> np.ndarray:
    """
    patch_float32 : (3, H, W) float32
      Band 1 = HH σ⁰ dB     → mean_r
      Band 2 = HV σ⁰ dB     → mean_g
      Band 3 = HH/HV ratio dB → hh_hv_rat
      (OPT1 has no RVI — rvi_mean = 0, matches training fill value)

    Returns (4,) float32 array: [mean_r, mean_g, hh_hv_rat, rvi_mean]
    """
    r, g, b = patch_float32[0], patch_float32[1], patch_float32[2]
    valid_r = r[np.isfinite(r)]
    valid_g = g[np.isfinite(g)]
    valid_b = b[np.isfinite(b)]

    mean_r    = float(np.mean(valid_r)) if valid_r.size > 0 else 0.0
    mean_g    = float(np.mean(valid_g)) if valid_g.size > 0 else 0.0
    hh_hv_rat = float(np.mean(valid_b)) if valid_b.size > 0 else 0.0
    rvi_mean  = 0.0     # OPT1 does not have RVI — consistent with training

    return np.array([mean_r, mean_g, hh_hv_rat, rvi_mean], dtype=np.float32)


def normalise_features_batch(feat_batch: np.ndarray,
                              feat_stats: dict) -> torch.Tensor:
    """
    feat_batch : (N, 4) float32
    Returns    : (N, 4) float32 tensor (z-scored)
    """
    out = feat_batch.copy()
    for i, col in enumerate(SAR_FEATURE_COLS):
        mu  = feat_stats[col]["mean"]
        std = feat_stats[col]["std"]
        out[:, i] = (out[:, i] - mu) / std
    return torch.tensor(out, dtype=torch.float32)


# =============================================================================
# 5.  PATCH GRID GENERATOR
#     Generates (row_start, col_start, row_end, col_end, padded?)
#     Covers the FULL scene — edge patches are zero-padded to 256×256.
# =============================================================================

def patch_grid(H: int, W: int, patch_size: int, stride: int):
    """
    Yields (r0, c0, r1, c1, needs_pad) for every patch position.
    For interior patches: needs_pad = False, r1-r0 = c1-c0 = patch_size.
    For edge patches:     needs_pad = True, actual slice may be smaller.
    """
    r0 = 0
    while r0 < H:
        r1 = min(r0 + patch_size, H)
        c0 = 0
        while c0 < W:
            c1    = min(c0 + patch_size, W)
            needs = (r1 - r0 < patch_size) or (c1 - c0 < patch_size)
            yield r0, c0, r1, c1, needs
            c0 += stride
        r0 += stride


def pad_patch(patch: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    Zero-pads patch (C, h, w) to (C, target_h, target_w).
    Padding is added to the bottom and right (matches the grid convention).
    """
    C, h, w = patch.shape
    out = np.zeros((C, target_h, target_w), dtype=patch.dtype)
    out[:, :h, :w] = patch
    return out


# =============================================================================
# 6.  CONVERT FLOAT32 PATCH TO UINT8 PIL IMAGE
#     Applies the same 2–98% stretch used when generating training PNGs
#     so the model sees visually identical images at inference.
# =============================================================================

def float_patch_to_pil(patch_f32: np.ndarray) -> Image.Image:
    """
    patch_f32 : (3, H, W)  float32
    Returns   : PIL Image RGB (H, W, 3) uint8
    """
    hwc = np.zeros((patch_f32.shape[1], patch_f32.shape[2], 3), dtype=np.uint8)
    for i in range(3):
        ch    = patch_f32[i]
        valid = ch[np.isfinite(ch)]
        if valid.size == 0:
            continue
        lo, hi = np.percentile(valid, 2), np.percentile(valid, 98)
        if hi <= lo:
            continue
        stretched        = np.clip((ch - lo) / (hi - lo) * 255.0, 0, 255)
        stretched[~np.isfinite(ch)] = 0
        hwc[:, :, i]     = stretched.astype(np.uint8)
    return Image.fromarray(hwc, mode="RGB")


# =============================================================================
# 7.  MAIN PREDICTION LOOP
# =============================================================================

def predict_scene(composite_tif: str, model_dir: str,
                  output_dir: str, patch_size: int = 256,
                  batch_size: int = 32) -> None:

    os.makedirs(output_dir, exist_ok=True)
    stride = patch_size   # no overlap at inference (faster, cleaner boundaries)

    # ── load model ─────────────────────────────────────────────────────────
    log.info("Loading model ...")
    model, id_to_class, feat_stats = load_model(model_dir)
    transform = get_val_transform()
    num_classes = len(id_to_class)
    log.info("id_to_class: %s", id_to_class)

    # ── load composite raster ───────────────────────────────────────────────
    log.info("Loading composite: %s", composite_tif)
    with rasterio.open(composite_tif) as src:
        scene = src.read([1, 2, 3]).astype(np.float32)   # (3, H, W)
        meta  = src.meta.copy()
        crs   = src.crs
        tf    = src.transform
    C, H, W = scene.shape
    log.info("Scene shape: C=%d  H=%d  W=%d", C, H, W)

    # ── output arrays ───────────────────────────────────────────────────────
    class_map  = np.full((H, W), -1,  dtype=np.int8)    # class ID per pixel
    conf_map   = np.zeros((H, W),     dtype=np.float32) # confidence per pixel

    # ── build patch list ────────────────────────────────────────────────────
    grid = list(patch_grid(H, W, patch_size, stride))
    log.info("Total patches to predict: %d  (including edge patches)", len(grid))

    # ── batched prediction ──────────────────────────────────────────────────
    records = []   # for CSV report
    img_batch  = []
    feat_batch = []
    meta_batch = []   # (r0, c0, r1, c1) for stitching

    def run_batch():
        """Runs inference on the accumulated batch and fills class_map."""
        if not img_batch:
            return
        imgs  = torch.stack(img_batch).to(DEVICE)             # (N,3,224,224)
        feats = normalise_features_batch(
                    np.stack(feat_batch), feat_stats).to(DEVICE)  # (N,4)

        with torch.no_grad():
            logits = model(imgs, feats)                        # (N, num_classes)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()

        pred_ids = probs.argmax(axis=1)   # (N,)
        confs    = probs.max(axis=1)      # (N,)

        for i, (r0, c0, r1, c1) in enumerate(meta_batch):
            pid   = int(pred_ids[i])
            conf  = float(confs[i])
            class_map[r0:r1, c0:c1] = pid
            conf_map [r0:r1, c0:c1] = conf
            records.append({
                "patch_id":   f"r{r0:05d}_c{c0:05d}",
                "row_start":  r0,
                "col_start":  c0,
                "class_id":   pid,
                "class_name": id_to_class[pid],
                "confidence": round(conf, 4),
            })

        img_batch.clear()
        feat_batch.clear()
        meta_batch.clear()

    log.info("Running inference ...")
    for r0, c0, r1, c1, needs_pad in tqdm(grid, desc="Predicting patches"):

        # extract patch from float32 scene
        patch = scene[:, r0:r1, c0:c1]          # (3, h, w) — may be < 256

        # skip patches that are entirely NaN / outside scene
        valid_pct = np.isfinite(patch).mean()
        if valid_pct < 0.10:
            # leave as -1 (no prediction) in class_map
            continue

        # pad edge patches to full patch_size
        if needs_pad:
            patch = pad_patch(patch, patch_size, patch_size)

        # SAR features from float32 values
        feat = extract_sar_features(patch)
        feat_batch.append(feat)

        # convert to uint8 PIL for CNN (same stretch as training PNGs)
        pil_img = float_patch_to_pil(patch)
        img_t   = transform(pil_img)
        img_batch.append(img_t)

        # remember location for stitching — use actual (non-padded) extent
        meta_batch.append((r0, c0, r1, c1))

        if len(img_batch) >= batch_size:
            run_batch()

    run_batch()   # flush remaining

    # ── save outputs ────────────────────────────────────────────────────────
    log.info("Saving outputs to %s ...", output_dir)

    # 1. Classification map GeoTIFF (class IDs, georeferenced)
    classmap_path = os.path.join(output_dir, "prediction_classmap.tif")
    out_meta = meta.copy()
    out_meta.update(count=1, dtype="int8", nodata=-1, compress="deflate")
    with rasterio.open(classmap_path, "w", **out_meta) as dst:
        dst.write(class_map, 1)
    log.info("  Class map  : %s", classmap_path)

    # 2. Confidence map GeoTIFF (softmax probability of predicted class)
    conf_path = os.path.join(output_dir, "prediction_confidence.tif")
    conf_meta = meta.copy()
    conf_meta.update(count=1, dtype="float32", nodata=0.0, compress="deflate")
    with rasterio.open(conf_path, "w", **conf_meta) as dst:
        dst.write(conf_map, 1)
    log.info("  Confidence : %s", conf_path)

    # 3. Colourised RGB PNG
    rgb_img = np.zeros((H, W, 3), dtype=np.uint8)
    for cid, colour in COLOUR_MAP.items():
        mask = class_map == cid
        rgb_img[mask] = colour
    Image.fromarray(rgb_img, "RGB").save(
        os.path.join(output_dir, "prediction_rgb.png"))
    log.info("  RGB png    : %s",
             os.path.join(output_dir, "prediction_rgb.png"))

    # 4. Per-patch CSV report
    csv_path = os.path.join(output_dir, "prediction_report.csv")
    pd.DataFrame(records).to_csv(csv_path, index=False)
    log.info("  Report CSV : %s", csv_path)

    # ── class distribution summary ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PREDICTION SUMMARY")
    print("=" * 60)
    total_pixels = H * W
    for cid, cname in id_to_class.items():
        n   = int((class_map == cid).sum())
        pct = 100.0 * n / total_pixels
        print(f"  {cname:<18} class_id={cid}  pixels={n:>10,}  ({pct:.1f}%)")
    no_pred = int((class_map == -1).sum())
    print(f"  {'No prediction':<18} class_id=-1  pixels={no_pred:>10,} "
          f"  ({100.0*no_pred/total_pixels:.1f}%)")
    print(f"\n  Total pixels : {total_pixels:,}")
    print(f"  Total patches: {len(records)}")
    print(f"\n  Mean confidence : {conf_map[conf_map > 0].mean():.4f}")
    print("=" * 60)
    print(f"""
OUTPUT FILES
------------
  {classmap_path}
      → Open in QGIS as raster layer
        Style: Paletted/Unique values
        0 = Dense_forest (dark green)
        1 = Mix          (yellow)
        2 = Urban        (red)
       -1 = No data      (transparent)

  {conf_path}
      → Open in QGIS; style with Singleband pseudocolor
        Low (0.5) = uncertain,  High (1.0) = very confident

  {os.path.join(output_dir, "prediction_rgb.png")}
      → Quick preview image

  {csv_path}
      → Per-patch class + confidence table
""")


# =============================================================================
# CLI
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Predict land-cover classes on full EOS-04 OPT1_sigma0 scene.")
    p.add_argument("--composite-tif", default=COMPOSITE_TIF,
                    help="Path to OPT1 composite GeoTIFF (float32, 3-band)")
    p.add_argument("--model-dir",     default=MODEL_DIR,
                    help="Folder containing best_model.pth + JSON files")
    p.add_argument("--output-dir",    default=OUTPUT_DIR,
                    help="Where to save prediction outputs")
    p.add_argument("--patch-size",    type=int, default=PATCH_SIZE)
    p.add_argument("--batch-size",    type=int, default=BATCH_SIZE,
                    help="Patches per GPU batch (reduce if CUDA OOM)")
    args = p.parse_args()

    print("=" * 60)
    print("EOS-04 SCENE PREDICTION")
    print(f"  Composite : {args.composite_tif}")
    print(f"  Model dir : {args.model_dir}")
    print(f"  Output    : {args.output_dir}")
    print(f"  Device    : {DEVICE}")
    print("=" * 60)

    predict_scene(
        composite_tif = args.composite_tif,
        model_dir     = args.model_dir,
        output_dir    = args.output_dir,
        patch_size    = args.patch_size,
        batch_size    = args.batch_size,
    )


if __name__ == "__main__":
    main()
