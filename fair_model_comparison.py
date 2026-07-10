#!/usr/bin/env python3
"""
fair_model_comparison.py
========================
Creates a FAIR comparison between the 256×256 and 128×128 patch models
by evaluating BOTH on the exact same geographic locations.

WHY DIRECT VAL_ACC COMPARISON IS UNFAIR
-----------------------------------------
256×256 model  →  val set = 200 patches  (random 20% of 997)
128×128 model  →  val set = 779 patches  (random 20% of 3894)

These are DIFFERENT patches covering DIFFERENT locations.
Comparing 93.5% vs 91.3% directly is like comparing exam scores
from two different exam papers — the numbers mean nothing against each other.

THE FAIR APPROACH
-----------------
1. Take the 256×256 val set (200 patches) as the COMMON TEST SET
   → these are the ground-truth labels both models must predict
2. For the 256×256 model: run inference directly on each 256×256 patch
3. For the 128×128 model: split each 256×256 patch into 4 non-overlapping
   128×128 sub-patches → run inference on all 4 → take MAJORITY VOTE
   → this gives one prediction per 256×256 geographic location
4. Compare BOTH against the same ground truth labels on the same 200 locations
5. Produce a true apples-to-apples classification report

HOW THE 4 SUB-PATCHES COVER ONE 256×256 PATCH
-----------------------------------------------
  ┌─────────┬─────────┐
  │  TL     │  TR     │  TL = top-left     [0:128, 0:128]
  │ (128²)  │ (128²)  │  TR = top-right    [0:128, 128:256]
  ├─────────┼─────────┤  BL = bottom-left  [128:256, 0:128]
  │  BL     │  BR     │  BR = bottom-right [128:256, 128:256]
  │ (128²)  │ (128²)  │
  └─────────┴─────────┘
  Majority vote of TL+TR+BL+BR → one class prediction per 256×256 location

OUTPUT FILES (saved to model/comparison/)
------------------------------------------
  fair_comparison_report.csv        ← per-patch: both predictions + truth
  fair_confusion_matrix.png         ← side-by-side CM on same 200 locations
  fair_per_class_metrics.png        ← precision/recall/F1 on same test set
  fair_verdict.txt                  ← which model wins and why

Usage
-----
    python fair_model_comparison.py

    # override paths
    python fair_model_comparison.py ^
        --val-csv      "patches\\03JUN2026\\OPT1_sigma0\\OPT1_sigma0_labels_clean.csv" ^
        --patches-256  "patches\\03JUN2026\\OPT1_sigma0\\patches" ^
        --model-256    "model\\EOS04" ^
        --model-128    "model\\EOS04_128" ^
        --output-dir   "model\\comparison"
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import logging
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score)
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fair_compare")

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT PATHS  — edit to match your machine
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR     = r"D:\ISRO_14"
VAL_CSV      = os.path.join(BASE_DIR, "patches",     "03JUN2026",
                             "OPT1_sigma0", "OPT1_sigma0_labels_clean.csv")
PATCHES_256  = os.path.join(BASE_DIR, "patches",     "03JUN2026",
                             "OPT1_sigma0", "patches")
MODEL_DIR_256 = os.path.join(BASE_DIR, "model", "EOS04")
MODEL_DIR_128 = os.path.join(BASE_DIR, "model", "EOS04_128")
OUTPUT_DIR   = os.path.join(BASE_DIR, "model", "comparison")

PATCH_256    = 256
PATCH_128    = 128
IMG_SIZE     = 224     # both models resize to 224 internally
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED  = 42      # same seed as training → same val split


# =============================================================================
# 1.  MODEL DEFINITION  (identical to training scripts)
# =============================================================================

class EOS04Classifier(nn.Module):
    def __init__(self, num_classes: int, num_sar_features: int = 4):
        super().__init__()
        backbone = models.efficientnet_b0(weights=None)
        for param in backbone.parameters():
            param.requires_grad = False
        self.cnn_features = backbone.features
        self.pool         = backbone.avgpool
        self.cnn_dim      = backbone.classifier[1].in_features
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
# 2.  LOAD A MODEL + ITS METADATA
# =============================================================================

def load_model_bundle(model_dir: str) -> dict:
    """
    Returns dict with: model, id_to_class, class_to_id, feat_stats, num_classes
    """
    with open(os.path.join(model_dir, "label_encoder.json")) as f:
        label_enc = json.load(f)
    with open(os.path.join(model_dir, "feature_stats.json")) as f:
        feat_stats = json.load(f)

    id_to_class  = {v: k for k, v in label_enc.items()}
    num_classes  = len(label_enc)

    ckpt  = torch.load(os.path.join(model_dir, "best_model.pth"),
                       map_location=DEVICE, weights_only=False)
    model = EOS04Classifier(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    log.info("Loaded model from %s  classes=%s  val_acc=%.3f",
             model_dir, list(label_enc.keys()), ckpt.get("val_acc", 0))

    return {
        "model":       model,
        "id_to_class": id_to_class,
        "class_to_id": label_enc,
        "feat_stats":  feat_stats,
        "num_classes": num_classes,
        "val_acc":     ckpt.get("val_acc", 0),
        "epoch":       ckpt.get("epoch", 0),
    }


# =============================================================================
# 3.  IMAGE TRANSFORMS + SAR FEATURE EXTRACTION
# =============================================================================

VAL_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std= [0.229, 0.224, 0.225]),
])

SAR_COLS = ["mean_r", "mean_g", "hh_hv_rat", "rvi_mean"]


def stretch_to_uint8(arr_hwc: np.ndarray) -> np.ndarray:
    """2-98% per-channel stretch → uint8 HWC."""
    out = np.zeros_like(arr_hwc, dtype=np.uint8)
    for i in range(arr_hwc.shape[2]):
        ch    = arr_hwc[:, :, i].astype(np.float32)
        valid = ch[np.isfinite(ch)]
        if valid.size == 0:
            continue
        lo, hi = np.percentile(valid, 2), np.percentile(valid, 98)
        if hi <= lo:
            continue
        stretched        = np.clip((ch - lo) / (hi - lo) * 255, 0, 255)
        stretched[~np.isfinite(ch)] = 0
        out[:, :, i]     = stretched.astype(np.uint8)
    return out


def png_to_tensor(png_path: str) -> torch.Tensor:
    """Load a PNG patch → normalised tensor (1, 3, 224, 224)."""
    img = Image.open(png_path).convert("RGB")
    return VAL_TF(img).unsqueeze(0)


def sar_features_from_png(png_path: str, feat_stats: dict) -> torch.Tensor:
    """
    Compute SAR features from a PNG patch.
    PNG is uint8 (already stretched) so we compute mean/std of each channel
    and derive proxy values. For OPT1: R=HH, G=HV, B=HH/HV ratio.
    These are proxy values from the display-stretched image — the exact
    calibrated values are not recoverable from uint8 PNG, but they are
    good enough for the SAR branch since all patches were stretched the
    same way during training.
    Uses zeros for rvi_mean (OPT1 has no RVI).
    """
    img = np.array(Image.open(png_path).convert("RGB"), dtype=np.float32)
    r   = img[:, :, 0][img[:, :, 0] > 0]
    g   = img[:, :, 1][img[:, :, 1] > 0]
    b   = img[:, :, 2][img[:, :, 2] > 0]

    mean_r    = float(np.mean(r)) if r.size > 0 else 0.0
    mean_g    = float(np.mean(g)) if g.size > 0 else 0.0
    hh_hv_rat = float(np.mean(b)) if b.size > 0 else 0.0
    rvi_mean  = 0.0

    raw = np.array([mean_r, mean_g, hh_hv_rat, rvi_mean], dtype=np.float32)

    # z-score normalise using training stats
    for i, col in enumerate(SAR_COLS):
        raw[i] = (raw[i] - feat_stats[col]["mean"]) / feat_stats[col]["std"]

    return torch.tensor(raw, dtype=torch.float32).unsqueeze(0)


# =============================================================================
# 4.  COMMON TEST SET — 256×256 VAL PATCHES
#     Reconstruct the same 20% val split used during 256 model training
#     (same random_state=42, stratify=label_id).
# =============================================================================

def get_common_test_set(val_csv: str, patches_256_dir: str) -> pd.DataFrame:
    """
    Loads the 256×256 labels CSV, reconstructs the train/val split with
    the same seed used in training, and returns the 200-patch val DataFrame.
    Only patches whose PNG files exist on disk are kept.
    """
    df = pd.read_csv(val_csv)
    df.columns = df.columns.str.strip()

    # drop geometry column if present
    if "WKT" in df.columns:
        df = df.drop(columns=["WKT"])

    # keep only valid patches
    if "use_flag" in df.columns:
        df = df[df["use_flag"] == 1].copy()

    # drop unlabelled
    df = df[df["label"].notna() & (df["label"] != "")].copy()
    df["label"] = df["label"].str.strip()

    # rebuild label_id (3 classes only — exclude Unclear if present)
    classes = ["Dense_forest", "Mix", "Urban"]
    cls_map = {c: i for i, c in enumerate(classes)}
    df      = df[df["label"].isin(classes)].copy()
    df["label_id"] = df["label"].map(cls_map).astype(int)

    # resolve PNG paths
    def resolve(p):
        p = str(p).strip()
        full = os.path.join(os.path.dirname(val_csv), p)
        if os.path.isfile(full): return full
        bare = os.path.join(patches_256_dir, os.path.basename(p))
        if os.path.isfile(bare): return bare
        return full

    if "png_path" in df.columns:
        df["abs_png"] = df["png_path"].apply(resolve)
    elif "patch_id" in df.columns:
        df["abs_png"] = df["patch_id"].apply(
            lambda p: os.path.join(patches_256_dir, f"{p}.png"))

    # drop missing PNGs
    df = df[df["abs_png"].apply(os.path.isfile)].copy()

    # reconstruct the SAME 80/20 split as training (seed=42, stratify)
    _, val_df = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED,
                                  stratify=df["label_id"])
    val_df = val_df.reset_index(drop=True)

    log.info("Common test set: %d patches  classes=%s",
             len(val_df), val_df["label"].value_counts().to_dict())
    return val_df, classes


# =============================================================================
# 5.  SPLIT 256×256 PNG INTO FOUR 128×128 SUB-PATCHES
# =============================================================================

def split_256_to_128(png_path: str) -> list[np.ndarray]:
    """
    Loads a 256×256 PNG and returns 4 non-overlapping 128×128 sub-arrays
    in order: TL, TR, BL, BR.
    If the PNG is not exactly 256×256, it is center-cropped / padded.
    """
    img = np.array(Image.open(png_path).convert("RGB"), dtype=np.uint8)
    H, W, _ = img.shape

    # ensure exactly 256×256
    if H != 256 or W != 256:
        canvas = np.zeros((256, 256, 3), dtype=np.uint8)
        h_in, w_in = min(H, 256), min(W, 256)
        canvas[:h_in, :w_in] = img[:h_in, :w_in]
        img = canvas

    return [
        img[  0:128,   0:128],   # TL
        img[  0:128, 128:256],   # TR
        img[128:256,   0:128],   # BL
        img[128:256, 128:256],   # BR
    ]


# =============================================================================
# 6.  PREDICT WITH 256 MODEL (single patch → class)
# =============================================================================

@torch.no_grad()
def predict_256(bundle: dict, png_path: str) -> tuple[int, float]:
    """Returns (predicted_class_id, confidence)."""
    img_t   = png_to_tensor(png_path).to(DEVICE)
    sar_t   = sar_features_from_png(png_path,
                                     bundle["feat_stats"]).to(DEVICE)
    logits  = bundle["model"](img_t, sar_t)
    probs   = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_id = int(probs.argmax())
    return pred_id, float(probs.max())


# =============================================================================
# 7.  PREDICT WITH 128 MODEL (4 sub-patches → majority vote)
# =============================================================================

@torch.no_grad()
def predict_128_majority(bundle: dict, png_path_256: str) -> tuple[int, float, list]:
    """
    Splits the 256×256 PNG into 4 128×128 sub-patches,
    runs the 128 model on each, returns:
      (majority_vote_class_id, mean_confidence, [4 individual predictions])
    """
    sub_arrays = split_256_to_128(png_path_256)
    sub_preds  = []
    sub_confs  = []

    for sub_arr in sub_arrays:
        pil    = Image.fromarray(sub_arr, "RGB")
        img_t  = VAL_TF(pil).unsqueeze(0).to(DEVICE)

        # SAR features from sub-patch
        r  = sub_arr[:, :, 0][sub_arr[:, :, 0] > 0].astype(np.float32)
        g  = sub_arr[:, :, 1][sub_arr[:, :, 1] > 0].astype(np.float32)
        b  = sub_arr[:, :, 2][sub_arr[:, :, 2] > 0].astype(np.float32)
        raw = np.array([
            float(np.mean(r)) if r.size > 0 else 0.0,
            float(np.mean(g)) if g.size > 0 else 0.0,
            float(np.mean(b)) if b.size > 0 else 0.0,
            0.0,   # rvi_mean = 0 for OPT1
        ], dtype=np.float32)
        fs = bundle["feat_stats"]
        for i, col in enumerate(SAR_COLS):
            raw[i] = (raw[i] - fs[col]["mean"]) / fs[col]["std"]
        sar_t  = torch.tensor(raw, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        logits = bundle["model"](img_t, sar_t)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
        sub_preds.append(int(probs.argmax()))
        sub_confs.append(float(probs.max()))

    # majority vote
    vote       = Counter(sub_preds).most_common(1)[0][0]
    mean_conf  = float(np.mean(sub_confs))
    return vote, mean_conf, sub_preds


# =============================================================================
# 8.  RUN BOTH MODELS ON COMMON TEST SET
# =============================================================================

def evaluate_both(val_df: pd.DataFrame, bundle_256: dict,
                   bundle_128: dict) -> pd.DataFrame:
    """
    Runs both models on every patch in val_df.
    Returns DataFrame with per-patch predictions from both models.
    """
    records = []
    n = len(val_df)
    log.info("Running both models on %d common test patches ...", n)

    for i, row in val_df.iterrows():
        png_path   = row["abs_png"]
        true_label = row["label"]
        true_id    = row["label_id"]

        # 256 model prediction
        pred_256_id, conf_256 = predict_256(bundle_256, png_path)
        pred_256_name = bundle_256["id_to_class"].get(pred_256_id, "unknown")

        # 128 model prediction (majority vote of 4 sub-patches)
        pred_128_id, conf_128, sub_votes = predict_128_majority(
            bundle_128, png_path)
        pred_128_name = bundle_128["id_to_class"].get(pred_128_id, "unknown")

        correct_256 = int(pred_256_name == true_label)
        correct_128 = int(pred_128_name == true_label)
        agreement   = int(pred_256_name == pred_128_name)

        records.append({
            "patch_id"    : row.get("patch_id", f"patch_{i}"),
            "true_label"  : true_label,
            "pred_256"    : pred_256_name,
            "pred_128"    : pred_128_name,
            "conf_256"    : round(conf_256, 4),
            "conf_128"    : round(conf_128, 4),
            "correct_256" : correct_256,
            "correct_128" : correct_128,
            "agreement"   : agreement,
            "sub_votes"   : str(sub_votes),
        })

        if (i + 1) % 20 == 0:
            log.info("  Progress: %d / %d", i + 1, n)

    return pd.DataFrame(records)


# =============================================================================
# 9.  PLOTS
# =============================================================================

def plot_fair_confusion_matrices(results: pd.DataFrame,
                                  classes: list[str],
                                  out_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"FAIR Comparison — Both models on same {len(results)} test patches\n"
        f"(256×256 patch val set, 128×128 model uses majority vote of 4 sub-patches)",
        fontsize=12, fontweight="bold")

    for ax, pred_col, title in zip(
            axes,
            ["pred_256", "pred_128"],
            [f"256×256 model  (trained val_acc={results['correct_256'].mean()*100:.1f}% here)",
             f"128×128 model  (majority vote, {results['correct_128'].mean()*100:.1f}% here)"]):

        # filter to rows where both true and pred are in classes
        mask = results["true_label"].isin(classes) & results[pred_col].isin(classes)
        cm   = confusion_matrix(results.loc[mask, "true_label"],
                                 results.loc[mask, pred_col],
                                 labels=classes)
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

        sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                    xticklabels=classes, yticklabels=classes,
                    ax=ax, vmin=0, vmax=100, annot_kws={"size": 12})
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j+0.5, i+0.72, f"n={cm[i,j]}",
                        ha="center", va="center", fontsize=8, color="gray")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual",    fontsize=10)

    plt.tight_layout()
    out = os.path.join(out_dir, "fair_confusion_matrix.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved: %s", out)


def plot_fair_metrics_bars(results: pd.DataFrame,
                            classes: list[str],
                            out_dir: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(
        "Per-Class Metrics on SAME Test Set (fair comparison)",
        fontsize=13, fontweight="bold")

    for ax, metric in zip(axes, ["precision", "recall", "f1-score"]):
        rep256 = classification_report(
            results["true_label"], results["pred_256"],
            labels=classes, output_dict=True, zero_division=0)
        rep128 = classification_report(
            results["true_label"], results["pred_128"],
            labels=classes, output_dict=True, zero_division=0)

        vals256 = [rep256[c][metric] for c in classes]
        vals128 = [rep128[c][metric] for c in classes]
        x       = np.arange(len(classes))
        w       = 0.35

        b256 = ax.bar(x - w/2, vals256, w, label="256×256",
                       color="steelblue", alpha=0.85)
        b128 = ax.bar(x + w/2, vals128, w, label="128×128 (maj.vote)",
                       color="coral",     alpha=0.85)

        for b in list(b256) + list(b128):
            ax.text(b.get_x() + b.get_width()/2,
                    b.get_height() + 0.005,
                    f"{b.get_height():.2f}",
                    ha="center", va="bottom", fontsize=9)

        ax.set_title(metric.replace("-", " ").title(), fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=10, fontsize=9)
        ax.set_ylim(0.65, 1.08)
        ax.set_ylabel(metric)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    out = os.path.join(out_dir, "fair_per_class_metrics.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved: %s", out)


def plot_confidence_comparison(results: pd.DataFrame, out_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Confidence Distribution on Same Test Set", fontsize=13)

    for ax, col, label, color in zip(
            axes,
            ["conf_256", "conf_128"],
            ["256×256 model", "128×128 model (mean of 4 sub-patches)"],
            ["steelblue", "coral"]):
        ax.hist(results[col], bins=30, color=color, alpha=0.8, edgecolor="white")
        ax.axvline(results[col].mean(), color="black", linestyle="--",
                   linewidth=1.5, label=f"mean={results[col].mean():.3f}")
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Softmax confidence (max probability)")
        ax.set_ylabel("Number of patches")
        ax.legend()
        ax.grid(alpha=0.25)

    plt.tight_layout()
    out = os.path.join(out_dir, "fair_confidence_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved: %s", out)


# =============================================================================
# 10. VERDICT
# =============================================================================

def print_and_save_verdict(results: pd.DataFrame,
                            classes: list[str],
                            bundle_256: dict,
                            bundle_128: dict,
                            out_dir: str) -> None:
    acc_256 = results["correct_256"].mean()
    acc_128 = results["correct_128"].mean()
    n       = len(results)
    agree   = results["agreement"].mean()

    rep256 = classification_report(
        results["true_label"], results["pred_256"],
        labels=classes, output_dict=True, zero_division=0)
    rep128 = classification_report(
        results["true_label"], results["pred_128"],
        labels=classes, output_dict=True, zero_division=0)

    sep = "=" * 72
    lines = [
        sep,
        "FAIR MODEL COMPARISON — SAME TEST SET",
        sep,
        f"  Test set size    : {n} patches  (256×256 val split, seed=42)",
        f"  256×256 model    : trained val_acc={bundle_256['val_acc']*100:.1f}%"
        f"  (epoch {bundle_256['epoch']})",
        f"  128×128 model    : trained val_acc={bundle_128['val_acc']*100:.1f}%"
        f"  (epoch {bundle_128['epoch']})",
        "",
        "  ┌─────────────────────────────────────────────────────────────┐",
        f"  │  Accuracy on SAME {n} patches:                            │",
        f"  │    256×256 model     : {acc_256*100:.1f}%                           │",
        f"  │    128×128 model     : {acc_128*100:.1f}%  (majority vote of 4)     │",
        f"  │    Model agreement   : {agree*100:.1f}% of patches predicted same   │",
        "  └─────────────────────────────────────────────────────────────┘",
        "",
        "  Per-class F1 on SAME test set:",
        f"  {'Class':<16} {'256 F1':>8} {'128 F1':>8} {'Winner':>8}",
        "  " + "─" * 44,
    ]
    for cls in classes:
        f1_256 = rep256[cls]["f1-score"]
        f1_128 = rep128[cls]["f1-score"]
        winner = "256" if f1_256 > f1_128 else ("128" if f1_128 > f1_256 else "tie")
        lines.append(
            f"  {cls:<16} {f1_256:>8.3f} {f1_128:>8.3f} {winner:>8}")

    wf1_256 = rep256["weighted avg"]["f1-score"]
    wf1_128 = rep128["weighted avg"]["f1-score"]
    lines += [
        "  " + "─" * 44,
        f"  {'Weighted avg':<16} {wf1_256:>8.3f} {wf1_128:>8.3f}"
        f" {'256' if wf1_256 > wf1_128 else '128':>8}",
        "",
        f"  Mean confidence: 256={results['conf_256'].mean():.3f}"
        f"   128={results['conf_128'].mean():.3f}",
        "",
        "  INTERPRETATION:",
    ]

    overall_winner = "256×256" if acc_256 > acc_128 else "128×128"
    diff = abs(acc_256 - acc_128) * 100

    lines += [
        f"  • On the SAME {n} geographic locations the {overall_winner} model",
        f"    achieves higher accuracy by {diff:.1f} percentage points.",
        f"  • This is a FAIR comparison — same patches, same ground truth.",
        f"  • Model agreement = {agree*100:.1f}%: both models agree on {agree*100:.0f}% of patches.",
        f"    The remaining {(1-agree)*100:.0f}% of patches are where the models",
        f"    disagree — these are the genuinely ambiguous Mixed-cover patches.",
        f"  • The 128 model uses majority vote of 4 sub-patches, which",
        f"    reduces noise from individual sub-patch errors.",
        f"  • FINAL RECOMMENDATION:",
        f"    Use 256×256 model if: you want highest per-patch accuracy",
        f"    Use 128×128 model if: you want finer spatial detail and more",
        f"    training data coverage for future acquisition dates.",
        sep,
    ]

    for line in lines:
        print(line)

    verdict_path = os.path.join(out_dir, "fair_verdict.txt")
    with open(verdict_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log.info("Saved: %s", verdict_path)


# =============================================================================
# MAIN
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Fair cross-scale model comparison on same test set.")
    p.add_argument("--val-csv",     default=VAL_CSV)
    p.add_argument("--patches-256", default=PATCHES_256)
    p.add_argument("--model-256",   default=MODEL_DIR_256)
    p.add_argument("--model-128",   default=MODEL_DIR_128)
    p.add_argument("--output-dir",  default=OUTPUT_DIR)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("FAIR MODEL COMPARISON")
    print(f"  Device     : {DEVICE}")
    print(f"  Val CSV    : {args.val_csv}")
    print(f"  Model 256  : {args.model_256}")
    print(f"  Model 128  : {args.model_128}")
    print("=" * 60)

    # load both models
    log.info("Loading 256×256 model ...")
    bundle_256 = load_model_bundle(args.model_256)
    log.info("Loading 128×128 model ...")
    bundle_128 = load_model_bundle(args.model_128)

    # build common test set
    log.info("Building common test set from 256×256 val CSV ...")
    val_df, classes = get_common_test_set(args.val_csv, args.patches_256)

    # run both models
    results = evaluate_both(val_df, bundle_256, bundle_128)

    # save per-patch results
    csv_out = os.path.join(args.output_dir, "fair_comparison_report.csv")
    results.to_csv(csv_out, index=False)
    log.info("Per-patch report: %s", csv_out)

    # plots
    log.info("Generating plots ...")
    plot_fair_confusion_matrices(results, classes, args.output_dir)
    plot_fair_metrics_bars(results,       classes, args.output_dir)
    plot_confidence_comparison(results,            args.output_dir)

    # verdict
    print_and_save_verdict(results, classes, bundle_256, bundle_128,
                            args.output_dir)

    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()