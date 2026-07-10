#!/usr/bin/env python3
"""
eos04_predict_scene_128.py
==========================
Full-scene prediction using the 128×128 patch model trained on
L2B terrain-normalized Gamma0 composites.

Identical logic to eos04_predict_scene.py with two changes:
  1. PATCH_SIZE = 128  (was 256)
  2. Default paths point to L2B model and composites

What this script does
---------------------
1.  Loads model/EOS04_128_L2B/best_model.pth
2.  Loads feature_stats.json + label_encoder.json
3.  Loads the full OPT1 L2B composite GeoTIFF (float32, 3-band)
4.  Cuts the scene into 128×128 patches (same grid as training)
    Edge patches are zero-padded so every pixel is covered
5.  For each patch:
      a. Computes SAR features on-the-fly (mean_r, mean_g, hh_hv_rat, rvi_mean)
      b. Normalises using feature_stats.json
      c. Runs model → predicted class ID + confidence
6.  Stitches all predictions back into a full-scene classification map
7.  Saves four output files:
      prediction_classmap.tif    ← GeoTIFF: each pixel = class ID (int8)
      prediction_confidence.tif  ← GeoTIFF: max softmax prob (float32)
      prediction_rgb.png         ← colourised PNG preview
      prediction_report.csv      ← per-patch: class, confidence, row, col

Expected patch count
---------------------
Scene: 10301 × 10201 pixels
128×128 grid: ~80 rows × ~79 cols = ~6320 patches
(vs ~1640 patches for 256×256)
→ ~4× finer spatial resolution in the prediction map

Colour coding
-------------
  Dense_forest  →  dark green   (0, 128, 0)
  Mix           →  yellow       (255, 200, 0)
  Urban         →  red          (220, 50, 50)
  No prediction →  black        (0, 0, 0)

Usage
-----
    python eos04_predict_scene_128.py

    # with explicit paths
    python eos04_predict_scene_128.py ^
        --composite-tif "composites_L2B\\03JUN2026\\OPT1_R-HH-sigma0_G-HV-sigma0_B-ratio-sigma0.tif" ^
        --model-dir     "model\\EOS04_128_L2B" ^
        --output-dir    "predictions_128_L2B\\03JUN2026" ^
        --patch-size 128 --batch-size 64
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
log = logging.getLogger("predict_128")

try:
    import rasterio
except ImportError:
    print("ERROR: pip install rasterio"); sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw): return x


# =============================================================================
# PATHS  — only change here if needed
# =============================================================================
BASE_DIR      = r"D:\ISRO_14"
COMPOSITE_TIF = os.path.join(
    BASE_DIR, "composites_L2B", "03JUN2026",
    "OPT1_R-HH-sigma0_G-HV-sigma0_B-ratio-sigma0.tif")
MODEL_DIR     = os.path.join(BASE_DIR, "model", "EOS04_128_L2B")
OUTPUT_DIR    = os.path.join(BASE_DIR, "predictions_128_L2B", "03JUN2026")

PATCH_SIZE    = 128     # ← key change from 256 version
BATCH_SIZE    = 64      # can be higher than 256 version (smaller patches)
IMG_SIZE      = 224     # model always resizes to 224×224 internally
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAR_FEATURE_COLS = ["mean_r", "mean_g", "hh_hv_rat", "rvi_mean"]

# Colour map: class_id → (R, G, B)
COLOUR_MAP = {
    0:  (0,   128,  0),    # Dense_forest  dark green
    1:  (255, 200,  0),    # Mix           yellow
    2:  (220,  50, 50),    # Urban         red
    -1: (0,     0,  0),    # no prediction black
}


# =============================================================================
# 1.  MODEL DEFINITION  (must match eos04_train_128.py exactly)
# =============================================================================

class EOS04Classifier(nn.Module):
    def __init__(self, num_classes: int, num_sar_features: int = 4):
        super().__init__()
        backbone = models.efficientnet_b0(weights=None)
        for param in backbone.parameters():
            param.requires_grad = False
        self.cnn_features   = backbone.features
        self.pool           = backbone.avgpool
        self.cnn_dim        = backbone.classifier[1].in_features   # 1280
        self._phase2_params = list(backbone.features[-2:].parameters())
        self.sar_branch = nn.Sequential(
            nn.Linear(num_sar_features, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )
        self.sar_dim  = 32
        fusion_dim    = self.cnn_dim + self.sar_dim    # 1312
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

def load_model(model_dir: str) -> tuple:
    """Returns (model, id_to_class, feat_stats)."""
    with open(os.path.join(model_dir, "label_encoder.json")) as f:
        label_enc = json.load(f)
    with open(os.path.join(model_dir, "feature_stats.json")) as f:
        feat_stats = json.load(f)

    id_to_class = {v: k for k, v in label_enc.items()}
    num_classes  = len(label_enc)

    ckpt  = torch.load(os.path.join(model_dir, "best_model.pth"),
                       map_location=DEVICE, weights_only=False)
    model = EOS04Classifier(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    log.info("Model loaded — classes=%s  val_acc=%.3f  epoch=%d",
             list(label_enc.keys()),
             ckpt.get("val_acc", 0), ckpt.get("epoch", 0))

    return model, id_to_class, feat_stats


# =============================================================================
# 3.  IMAGE TRANSFORM  (val transform — no augmentation)
# =============================================================================

VAL_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std= [0.229, 0.224, 0.225]),
])


# =============================================================================
# 4.  SAR FEATURE EXTRACTION FROM A FLOAT32 PATCH
#     For OPT1 L2B:
#       Band 1 = HH terrain-normalized Gamma0 dB  → mean_r
#       Band 2 = HV terrain-normalized Gamma0 dB  → mean_g
#       Band 3 = HH/HV ratio dB                   → hh_hv_rat
#       rvi_mean = 0.0  (OPT1 has no RVI channel)
# =============================================================================

def extract_sar_features(patch_f32: np.ndarray) -> np.ndarray:
    """
    patch_f32 : (3, H, W) float32 from the L2B composite GeoTIFF.
    Returns (4,) float32: [mean_r, mean_g, hh_hv_rat, rvi_mean].
    """
    r, g, b = patch_f32[0], patch_f32[1], patch_f32[2]
    vr = r[np.isfinite(r)];  mean_r    = float(np.mean(vr)) if vr.size > 0 else 0.0
    vg = g[np.isfinite(g)];  mean_g    = float(np.mean(vg)) if vg.size > 0 else 0.0
    vb = b[np.isfinite(b)];  hh_hv_rat = float(np.mean(vb)) if vb.size > 0 else 0.0
    return np.array([mean_r, mean_g, hh_hv_rat, 0.0], dtype=np.float32)


def normalise_sar(raw: np.ndarray, feat_stats: dict) -> torch.Tensor:
    """Z-score normalise (4,) array → (1, 4) tensor."""
    out = raw.copy()
    for i, col in enumerate(SAR_FEATURE_COLS):
        out[i] = (out[i] - feat_stats[col]["mean"]) / feat_stats[col]["std"]
    return torch.tensor(out, dtype=torch.float32).unsqueeze(0)


# =============================================================================
# 5.  FLOAT32 PATCH → UINT8 PIL IMAGE
#     Same 2–98% stretch used when generating training PNGs so the model
#     sees visually identical images at inference time.
# =============================================================================

def float_patch_to_pil(patch_f32: np.ndarray) -> Image.Image:
    """patch_f32: (3, H, W) → PIL Image RGB (H, W, 3) uint8."""
    hwc = np.zeros((patch_f32.shape[1], patch_f32.shape[2], 3), dtype=np.uint8)
    for i in range(3):
        ch    = patch_f32[i]
        valid = ch[np.isfinite(ch)]
        if valid.size == 0:
            continue
        lo, hi = np.percentile(valid, 2), np.percentile(valid, 98)
        if hi <= lo:
            continue
        stretched       = np.clip((ch - lo) / (hi - lo) * 255.0, 0, 255)
        stretched[~np.isfinite(ch)] = 0
        hwc[:, :, i]    = stretched.astype(np.uint8)
    return Image.fromarray(hwc, mode="RGB")


# =============================================================================
# 6.  PATCH GRID GENERATOR
#     Covers the FULL scene including edge patches (zero-padded).
# =============================================================================

def patch_grid(H: int, W: int, patch_size: int, stride: int):
    """Yields (r0, c0, r1, c1, needs_pad) for every patch position."""
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
    """Zero-pad (C, h, w) to (C, target_h, target_w)."""
    C, h, w = patch.shape
    out = np.zeros((C, target_h, target_w), dtype=patch.dtype)
    out[:, :h, :w] = patch
    return out


# =============================================================================
# 7.  MAIN PREDICTION LOOP
# =============================================================================

def predict_scene(composite_tif: str, model_dir: str,
                  output_dir: str, patch_size: int = 128,
                  batch_size: int = 64) -> None:

    os.makedirs(output_dir, exist_ok=True)
    stride = patch_size   # no overlap at inference

    # ── load model ─────────────────────────────────────────────────────────
    log.info("Loading 128×128 model from %s ...", model_dir)
    model, id_to_class, feat_stats = load_model(model_dir)
    num_classes = len(id_to_class)
    log.info("id_to_class: %s", id_to_class)

    # ── load composite raster ───────────────────────────────────────────────
    log.info("Loading composite: %s", composite_tif)
    with rasterio.open(composite_tif) as src:
        scene = src.read([1, 2, 3]).astype(np.float32)   # (3, H, W)
        meta  = src.meta.copy()
    C, H, W = scene.shape
    log.info("Scene shape: C=%d  H=%d  W=%d", C, H, W)

    # ── output arrays ───────────────────────────────────────────────────────
    class_map = np.full((H, W), -1,  dtype=np.int8)
    conf_map  = np.zeros((H, W),     dtype=np.float32)

    # ── build full patch list ───────────────────────────────────────────────
    grid  = list(patch_grid(H, W, patch_size, stride))
    total = len(grid)
    log.info("Total patches: %d  (patch=%d  stride=%d)", total, patch_size, stride)

    # ── batched prediction ──────────────────────────────────────────────────
    records    = []
    img_batch  = []
    feat_batch = []
    meta_batch = []

    def run_batch():
        if not img_batch:
            return
        imgs  = torch.stack(img_batch).to(DEVICE)
        feats = torch.cat(feat_batch, dim=0).to(DEVICE)

        with torch.no_grad():
            probs = torch.softmax(model(imgs, feats), dim=1).cpu().numpy()

        pred_ids = probs.argmax(axis=1)
        confs    = probs.max(axis=1)

        for i, (r0, c0, r1, c1) in enumerate(meta_batch):
            pid  = int(pred_ids[i])
            conf = float(confs[i])
            class_map[r0:r1, c0:c1] = pid
            conf_map [r0:r1, c0:c1] = conf
            records.append({
                "patch_id":   f"r{r0:05d}_c{c0:05d}",
                "row_start":  r0,
                "col_start":  c0,
                "patch_size": patch_size,
                "class_id":   pid,
                "class_name": id_to_class[pid],
                "confidence": round(conf, 4),
            })

        img_batch.clear()
        feat_batch.clear()
        meta_batch.clear()

    log.info("Running inference ...")
    for r0, c0, r1, c1, needs_pad in tqdm(grid, desc="Predicting (128×128)"):

        patch = scene[:, r0:r1, c0:c1]

        # skip patches that are entirely outside the scene
        if np.isfinite(patch).mean() < 0.10:
            continue

        if needs_pad:
            patch = pad_patch(patch, patch_size, patch_size)

        # SAR features from float32 values
        raw_sar = extract_sar_features(patch)
        feat_t  = normalise_sar(raw_sar, feat_stats)   # (1, 4)

        # image
        img_t   = VAL_TF(float_patch_to_pil(patch))   # (3, 224, 224)

        img_batch.append(img_t)
        feat_batch.append(feat_t)
        meta_batch.append((r0, c0, r1, c1))

        if len(img_batch) >= batch_size:
            run_batch()

    run_batch()   # flush remaining

    # ── save outputs ────────────────────────────────────────────────────────
    log.info("Saving outputs to %s ...", output_dir)

    # 1. Classification map GeoTIFF
    classmap_path = os.path.join(output_dir, "prediction_classmap.tif")
    out_meta = meta.copy()
    out_meta.update(count=1, dtype="int8", nodata=-1, compress="deflate")
    with rasterio.open(classmap_path, "w", **out_meta) as dst:
        dst.write(class_map, 1)
    log.info("  Class map  : %s", classmap_path)

    # 2. Confidence map GeoTIFF
    conf_path = os.path.join(output_dir, "prediction_confidence.tif")
    conf_meta = meta.copy()
    conf_meta.update(count=1, dtype="float32", nodata=0.0, compress="deflate")
    with rasterio.open(conf_path, "w", **conf_meta) as dst:
        dst.write(conf_map, 1)
    log.info("  Confidence : %s", conf_path)

    # 3. Colourised RGB PNG
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for cid, colour in COLOUR_MAP.items():
        rgb[class_map == cid] = colour
    Image.fromarray(rgb, "RGB").save(
        os.path.join(output_dir, "prediction_rgb.png"))
    log.info("  RGB png    : %s",
             os.path.join(output_dir, "prediction_rgb.png"))

    # 4. Per-patch CSV report
    csv_path = os.path.join(output_dir, "prediction_report.csv")
    pd.DataFrame(records).to_csv(csv_path, index=False)
    log.info("  Report CSV : %s", csv_path)

    # ── class distribution summary ──────────────────────────────────────────
    total_px = H * W
    print("\n" + "=" * 60)
    print("PREDICTION SUMMARY  (128×128 patch model)")
    print("=" * 60)
    for cid, cname in id_to_class.items():
        n   = int((class_map == cid).sum())
        pct = 100.0 * n / total_px
        print(f"  {cname:<18} class_id={cid}  "
              f"pixels={n:>10,}  ({pct:.1f}%)")
    no_pred = int((class_map == -1).sum())
    print(f"  {'No prediction':<18} class_id=-1  "
          f"pixels={no_pred:>10,}  ({100*no_pred/total_px:.1f}%)")
    print(f"\n  Total pixels      : {total_px:,}")
    print(f"  Total patches used: {len(records):,}")
    print(f"  Mean confidence   : {conf_map[conf_map > 0].mean():.4f}")
    valid_conf = conf_map[conf_map > 0]
    high_conf  = int((valid_conf >= 0.80).sum())
    print(f"  High conf (≥0.80) : {high_conf:,} patches "
          f"({100*high_conf/max(len(records),1):.1f}%)")
    print("=" * 60)
    print(f"""
HOW TO OPEN IN QGIS
--------------------
1. Layer → Add Raster Layer → {classmap_path}
   Symbology → Paletted / Unique Values → Classify
   Set colours:
     0 (Dense_forest) → dark green  #008000
     1 (Mix)          → yellow      #FFC800
     2 (Urban)        → red         #DC3232
    -1 (No data)      → transparent

2. Layer → Add Raster Layer → {conf_path}
   Symbology → Singleband pseudocolor
   Color ramp: RdYlGn  Min=0.33  Max=1.0
   Low confidence patches will appear red.

3. Compare with 256×256 prediction:
   Load predictions\\03JUN2026\\prediction_classmap.tif
   Toggle both layers to see where finer 128×128 patches
   capture urban edges and forest boundaries better.
""")


# =============================================================================
# CLI
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Predict land-cover on full EOS-04 scene "
                    "using 128×128 patch model (L2B Gamma0).")
    p.add_argument("--composite-tif", default=COMPOSITE_TIF,
                    help="OPT1 L2B composite GeoTIFF (float32, 3-band)")
    p.add_argument("--model-dir",     default=MODEL_DIR,
                    help="Folder with best_model.pth + JSON files")
    p.add_argument("--output-dir",    default=OUTPUT_DIR)
    p.add_argument("--patch-size",    type=int, default=PATCH_SIZE)
    p.add_argument("--batch-size",    type=int, default=BATCH_SIZE,
                    help="Patches per GPU batch (reduce if CUDA OOM)")
    args = p.parse_args()

    print("=" * 60)
    print("EOS-04 SCENE PREDICTION  (128×128 patch model)")
    print(f"  Composite : {args.composite_tif}")
    print(f"  Model dir : {args.model_dir}")
    print(f"  Patch size: {args.patch_size} × {args.patch_size} pixels")
    print(f"  Batch size: {args.batch_size}")
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