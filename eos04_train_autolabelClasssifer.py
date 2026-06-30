#!/usr/bin/env python3
"""
SAR PATCH CLASSIFIER — UPDATED FOR EOS-04 CALIBRATED COMPOSITES
=================================================================
Trains EfficientNet-B0 on PNG patches generated from ISRO-calibrated
EOS-04 SAR composites (OPT1_sigma0, OPT2_gamma0, OPT3_rvi).

KEY DIFFERENCES from old script
---------------------------------
Old script:
  - 300 patches from a single folder (patches_png/)
  - Labels from 3 separate Excel files
  - Classes: None / Urban Dense / Hills Rural

New script:
  - Patches from multiple composite folders under patches/03JUN2026/
  - Labels from clean CSV files exported from QGIS
    (OPT1_sigma0_labels_clean.csv, OPT2_gamma0_labels_clean.csv, etc.)
  - Classes: Dense_forest / Mix / Urban
  - Patches filtered by use_flag == 1 (bad/edge patches excluded)
  - Supports training on ONE composite or ALL composites together
  - Radiometric features (mean_r, mean_g, hh_hv_rat, rvi_mean) added as
    auxiliary inputs alongside the RGB image — improves accuracy because
    these are calibrated physical values, not just visual appearance
  - Multi-input model: CNN branch (image) + MLP branch (SAR features)
    → fused for final classification

Input layout expected
---------------------
  patches/03JUN2026/
    OPT1_sigma0/
      patches/                          ← PNG files
      OPT1_sigma0_labels_clean.csv      ← from fix_exported_csv.py
    OPT2_gamma0/
      patches/
      OPT2_gamma0_labels_clean.csv
    OPT3_rvi/
      patches/
      OPT3_rvi_labels_clean.csv

Output
------
  model/EOS04/
    best_model.pth
    training_results.png
    confusion_matrix.png
    label_encoder.json
    feature_stats.json              ← mean/std of SAR features for inference

Usage
-----
  # Train on OPT1 only (recommended first run)
  python eos04_train.py --composite OPT1_sigma0

  # Train on all composites together (more data = better generalisation)
  python eos04_train.py --composite all

  # Train on OPT3 (RVI-enhanced, best for vegetation classes)
  python eos04_train.py --composite OPT3_rvi

  # Run for 30 epochs
  python eos04_train.py --composite OPT1_sigma0 --epochs 30
"""

from __future__ import annotations

import os
import json
import argparse
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

# =============================================================================
# CONFIG  — edit paths here to match your machine
# =============================================================================
BASE_DIR      = r"D:\ISRO_14"                        # root project folder
PATCHES_ROOT  = os.path.join(BASE_DIR, "patches", "03JUN2026")
MODEL_DIR     = os.path.join(BASE_DIR, "model", "EOS04")
os.makedirs(MODEL_DIR, exist_ok=True)

# Composites available — each must have a labels_clean.csv inside its folder
COMPOSITE_NAMES = ["OPT1_sigma0", "OPT2_gamma0", "OPT3_rvi"]

# Classes from your QGIS labelling
CLASS_NAMES = ["Dense_forest", "Mix", "Urban"]

# Training hyperparameters
BATCH_SIZE   = 16
EPOCHS       = 50
LR           = 1e-4
IMG_SIZE     = 224
VAL_SPLIT    = 0.2
RANDOM_SEED  = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# STEP 1 — LOAD AND MERGE LABELS FROM CSV(S)
# =============================================================================

def load_labels(composite: str, patches_root: str) -> pd.DataFrame:
    """
    Loads the labels_clean.csv for the requested composite(s).
    composite = 'OPT1_sigma0' | 'OPT2_gamma0' | 'OPT3_rvi' | 'all'

    For each row the full absolute path to the PNG is constructed so
    the Dataset can open it without knowing the composite subfolder.

    Returns DataFrame with columns:
        abs_png_path, label, label_id, composite,
        mean_r, mean_g, hh_hv_rat, rvi_mean, valid_pct
    """
    if composite == "all":
        targets = COMPOSITE_NAMES
    else:
        targets = [composite]

    dfs = []
    for name in targets:
        csv_path = os.path.join(patches_root, name,
                                f"{name}_labels_clean.csv")
        if not os.path.isfile(csv_path):
            print(f"  [WARN] CSV not found, skipping: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        print(f"  Loaded {csv_path}  →  {len(df)} rows")

        # keep only valid patches
        if "use_flag" in df.columns:
            before = len(df)
            df = df[df["use_flag"] == 1].copy()
            print(f"    use_flag filter: kept {len(df)} / {before} patches")

        # drop unlabelled rows
        df = df[df["label"].notna() & (df["label"] != "")].copy()

        # build absolute PNG path from composite subfolder + relative png_path
        composite_dir = os.path.join(patches_root, name)
        df["abs_png_path"] = df["png_path"].apply(
            lambda p: os.path.join(composite_dir, p)
        )

        # fill SAR feature columns that are N/A for this composite with 0
        # (e.g. rvi_mean = -9999 for OPT1/OPT2 — replace with 0 as neutral)
        for col in ["mean_r", "mean_g", "mean_b", "hh_hv_rat", "rvi_mean"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].replace(-9999.0, np.nan).fillna(0.0)
            else:
                df[col] = 0.0

        df["composite"] = name
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(
            f"No label CSV files found for composite='{composite}' "
            f"under {patches_root}"
        )

    merged = pd.concat(dfs, ignore_index=True)

    # Normalise label text (strip spaces)
    merged["label"] = merged["label"].str.strip()

    # Map labels to integer IDs — only Dense_forest / Mix / Urban are valid
    class_to_id = {name: i for i, name in enumerate(CLASS_NAMES)}
    merged["label_id"] = merged["label"].map(class_to_id)

    unknown = merged[merged["label_id"].isna()]["label"].unique()
    if len(unknown):
        print(f"  [WARN] Unknown labels dropped: {unknown}")
        merged = merged[merged["label_id"].notna()].copy()

    merged["label_id"] = merged["label_id"].astype(int)

    print(f"\nLabel distribution (composite={composite}):")
    for lbl, cnt in merged["label"].value_counts().items():
        print(f"  {lbl}: {cnt}")
    print(f"  TOTAL: {len(merged)}")

    return merged


# =============================================================================
# STEP 2 — SAR FEATURE NORMALISATION STATS
#           (computed on TRAIN split, applied to both train+val)
# =============================================================================

SAR_FEATURE_COLS = ["mean_r", "mean_g", "hh_hv_rat", "rvi_mean"]


def compute_feature_stats(train_df: pd.DataFrame) -> dict:
    stats = {}
    for col in SAR_FEATURE_COLS:
        stats[col] = {
            "mean": float(train_df[col].mean()),
            "std":  max(float(train_df[col].std()), 1e-6),
        }
    return stats


def normalise_features(df: pd.DataFrame, stats: dict) -> np.ndarray:
    """Returns (N, 4) float32 array of z-score normalised SAR features."""
    arr = np.zeros((len(df), len(SAR_FEATURE_COLS)), dtype=np.float32)
    for i, col in enumerate(SAR_FEATURE_COLS):
        arr[:, i] = ((df[col].values - stats[col]["mean"])
                     / stats[col]["std"]).astype(np.float32)
    return arr


# =============================================================================
# STEP 3 — PYTORCH DATASET
# =============================================================================

class EOS04PatchDataset(Dataset):
    """
    Returns (image_tensor, sar_features_tensor, label_id) for each patch.

    image_tensor    : (3, 224, 224)  float32  — RGB PNG, normalised
    sar_features    : (4,)           float32  — z-scored SAR calibration values
                      [mean_r_sigma0, mean_g_sigma0, hh_hv_ratio, rvi_mean]
    label_id        : int
    """
    def __init__(self, df: pd.DataFrame, feature_arr: np.ndarray,
                 transform=None):
        self.df           = df.reset_index(drop=True)
        self.feature_arr  = feature_arr   # pre-computed (N, 4)
        self.transform    = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        label    = int(row["label_id"])
        sar_feat = torch.tensor(self.feature_arr[idx], dtype=torch.float32)

        img = Image.open(row["abs_png_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, sar_feat, label


# =============================================================================
# STEP 4 — DATA AUGMENTATION
# =============================================================================

def get_transforms():
    """
    SAR-appropriate augmentation:
    - Horizontal + vertical flip: SAR images have no canonical orientation
    - Rotation: the satellite track is at an angle to geographic North
    - NO colour jitter: SAR composites encode physical values — hue shifts
      would corrupt the radiometric meaning of R/G/B channels
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=90),
        # NO colour jitter — SAR composite R/G/B = physical backscatter values
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_tf, val_tf


# =============================================================================
# STEP 5 — MULTI-INPUT MODEL
#   CNN branch   : EfficientNet-B0 backbone  →  (1280,) features
#   SAR branch   : small MLP on 4 calibrated features  →  (32,) features
#   Fusion head  : concat → Linear → 3 classes
#
# WHY multi-input?
#   The PNG patches are 2-98% stretched for display — the raw radiometric
#   values are partially lost in the stretch. The separate SAR feature
#   branch feeds the ACTUAL calibrated values (mean_r dB, HH/HV ratio,
#   RVI) directly into the classifier, giving it physical backscatter
#   information that the image alone cannot reliably encode.
# =============================================================================

class EOS04Classifier(nn.Module):
    def __init__(self, num_classes: int, num_sar_features: int = 4):
        super().__init__()

        # ── CNN branch (EfficientNet-B0 backbone) ──────────────────────────
        backbone = models.efficientnet_b0(weights="IMAGENET1K_V1")

        # Freeze backbone initially — unfreeze at epoch 21
        for param in backbone.parameters():
            param.requires_grad = False

        # Remove the original classifier; keep feature extractor only
        self.cnn_features = backbone.features   # outputs (B, 1280, 7, 7)
        self.pool         = backbone.avgpool     # → (B, 1280, 1, 1)
        self.cnn_dim      = backbone.classifier[1].in_features  # 1280

        # Unfreeze the last 2 feature blocks + avgpool for Phase 2
        self._phase2_params = list(backbone.features[-2:].parameters())

        # ── SAR feature branch (small MLP) ─────────────────────────────────
        self.sar_branch = nn.Sequential(
            nn.Linear(num_sar_features, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )
        self.sar_dim = 32

        # ── Fusion classifier head ──────────────────────────────────────────
        fusion_dim = self.cnn_dim + self.sar_dim   # 1280 + 32 = 1312
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, images, sar_features):
        # CNN path
        x = self.cnn_features(images)    # (B, 1280, 7, 7)
        x = self.pool(x)                 # (B, 1280, 1, 1)
        x = x.flatten(1)                 # (B, 1280)

        # SAR feature path
        s = self.sar_branch(sar_features)  # (B, 32)

        # Fusion
        fused = torch.cat([x, s], dim=1)  # (B, 1312)
        return self.classifier(fused)

    def unfreeze_for_phase2(self):
        """Called at epoch 21 to fine-tune the last 2 CNN blocks."""
        for param in self._phase2_params:
            param.requires_grad = True


# =============================================================================
# STEP 6 — TRAINING & VALIDATION LOOPS
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, sar_feat, labels in loader:
        images   = images.to(device)
        sar_feat = sar_feat.to(device)
        labels   = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, sar_feat)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, sar_feat, labels in loader:
            images   = images.to(device)
            sar_feat = sar_feat.to(device)
            labels   = labels.to(device)

            outputs    = model(images, sar_feat)
            loss       = criterion(outputs, labels)
            preds      = outputs.argmax(1)

            total_loss += loss.item() * images.size(0)
            correct    += (preds == labels).sum().item()
            total      += images.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


# =============================================================================
# STEP 7 — PLOTS
# =============================================================================

def plot_curves(history, model_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("EOS-04 SAR Patch Classifier — Training Results", fontsize=13)
    ep = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(ep, history["train_loss"], label="Train", color="steelblue")
    axes[0].plot(ep, history["val_loss"],   label="Val",   color="coral")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(ep, history["train_acc"], label="Train", color="steelblue")
    axes[1].plot(ep, history["val_acc"],   label="Val",   color="coral")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1)
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(model_dir, "training_results.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()


def plot_cm(labels, preds, class_names, model_dir):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title("Confusion Matrix (Validation Set)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    out = os.path.join(model_dir, "confusion_matrix.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--composite", default="OPT1_sigma0",
                        choices=COMPOSITE_NAMES + ["all"],
                        help="Which composite(s) to train on")
    parser.add_argument("--epochs",  type=int, default=EPOCHS)
    parser.add_argument("--batch",   type=int, default=BATCH_SIZE)
    parser.add_argument("--lr",      type=float, default=LR)
    args = parser.parse_args()

    print("=" * 60)
    print("EOS-04 SAR PATCH CLASSIFIER — TRAINING")
    print(f"Composite : {args.composite}")
    print(f"Device    : {DEVICE}")
    print("=" * 60)

    # ── Load labels ───────────────────────────────────────────────────────
    print("\n[1] Loading labels ...")
    df = load_labels(args.composite, PATCHES_ROOT)
    num_classes = len(CLASS_NAMES)
    print(f"  Classes ({num_classes}): {CLASS_NAMES}")

    # verify PNG files exist
    missing = df[~df["abs_png_path"].apply(os.path.isfile)]
    if len(missing):
        print(f"  [WARN] {len(missing)} PNG files missing — dropping them")
        df = df[df["abs_png_path"].apply(os.path.isfile)].copy()
    print(f"  Final: {len(df)} usable patches")

    if len(df) < 10:
        raise ValueError("Too few patches to train. Check your CSV paths.")

    # ── Save label encoder ────────────────────────────────────────────────
    label_enc = {name: i for i, name in enumerate(CLASS_NAMES)}
    with open(os.path.join(MODEL_DIR, "label_encoder.json"), "w") as f:
        json.dump(label_enc, f, indent=2)
    print(f"  Label encoder saved: {label_enc}")

    # ── Train / val split ─────────────────────────────────────────────────
    print(f"\n[2] Splitting dataset (80/20) ...")
    train_df, val_df = train_test_split(
        df, test_size=VAL_SPLIT, random_state=RANDOM_SEED,
        stratify=df["label_id"]
    )
    print(f"  Train: {len(train_df)}   Val: {len(val_df)}")

    # ── SAR feature normalisation ─────────────────────────────────────────
    print("\n[3] Computing SAR feature normalisation stats from train split ...")
    feat_stats = compute_feature_stats(train_df)
    with open(os.path.join(MODEL_DIR, "feature_stats.json"), "w") as f:
        json.dump(feat_stats, f, indent=2)
    for col, s in feat_stats.items():
        print(f"  {col}: mean={s['mean']:.4f}  std={s['std']:.4f}")

    train_feat = normalise_features(train_df, feat_stats)
    val_feat   = normalise_features(val_df,   feat_stats)

    # ── Datasets & loaders ────────────────────────────────────────────────
    train_tf, val_tf = get_transforms()

    train_ds = EOS04PatchDataset(train_df, train_feat, train_tf)
    val_ds   = EOS04PatchDataset(val_df,   val_feat,   val_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch,
                               shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch,
                               shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────
    print(f"\n[4] Building multi-input EfficientNet-B0 model ...")
    model = EOS04Classifier(num_classes=num_classes).to(DEVICE)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Total params    : {total:,}")
    print(f"  Trainable params: {trainable:,}  (backbone frozen until epoch 21)")

    # ── Class-weighted loss (handles imbalanced labels) ───────────────────
    counts_series = train_df["label_id"].value_counts().sort_index()
    counts_full   = np.zeros(num_classes, dtype=np.float32)
    for label_id, count in counts_series.items():
        counts_full[int(label_id)] = count

    weights_arr = np.where(counts_full > 0, 1.0 / (counts_full + 1e-6), 0.0)
    weights = torch.tensor(weights_arr, dtype=torch.float32).to(DEVICE)
    weights = weights / (weights.sum() + 1e-6) * num_classes
    criterion = nn.CrossEntropyLoss(weight=weights)
    print(f"  Class weights: { {CLASS_NAMES[i]: round(float(weights[i]), 3) for i in range(num_classes)} }")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # ── Training loop ─────────────────────────────────────────────────────
    print(f"\n[5] Training for {args.epochs} epochs ...")
    print(f"  Phase 1 (ep 1-20)  : classifier head + SAR branch only")
    print(f"  Phase 2 (ep 21+)   : + last 2 CNN blocks unfrozen")
    print("-" * 60)

    history = {k: [] for k in
               ["train_loss", "val_loss", "train_acc", "val_acc"]}
    best_val_acc  = 0.0
    best_path     = os.path.join(MODEL_DIR, "best_model.pth")
    final_preds   = []
    final_labels  = []

    for epoch in range(1, args.epochs + 1):

        # Phase 2: unfreeze last CNN blocks at epoch 21
        if epoch == 21:
            print("\n  [Phase 2] Unfreezing last 2 CNN blocks ...")
            model.unfreeze_for_phase2()
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr * 0.1, weight_decay=1e-4
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5
            )

        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE)
        vl_loss, vl_acc, v_preds, v_labels = validate(
            model, val_loader, criterion, DEVICE)

        scheduler.step(vl_loss)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        print(f"  Ep {epoch:3d}/{args.epochs} | "
              f"Train  loss={tr_loss:.4f}  acc={tr_acc:.3f} | "
              f"Val  loss={vl_loss:.4f}  acc={vl_acc:.3f}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            final_preds  = v_preds
            final_labels = v_labels
            torch.save({
                "epoch":             epoch,
                "model_state_dict":  model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc":           vl_acc,
                "class_names":       CLASS_NAMES,
                "composite":         args.composite,
            }, best_path)
            print(f"    ✓ Best model saved (val_acc={vl_acc:.3f})")

    # ── Final report ──────────────────────────────────────────────────────
    print(f"\n[6] Final evaluation ...")
    print(f"  Best validation accuracy: {best_val_acc:.4f} "
          f"({best_val_acc*100:.1f}%)")
    print("\nClassification Report:")
    print(classification_report(final_labels, final_preds,
                                target_names=CLASS_NAMES))

    print("\n[7] Saving plots ...")
    plot_curves(history, MODEL_DIR)
    plot_cm(final_labels, final_preds, CLASS_NAMES, MODEL_DIR)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"  Best model   : {best_path}")
    print(f"  Best val acc : {best_val_acc*100:.1f}%")
    print(f"  Label encoder: {os.path.join(MODEL_DIR, 'label_encoder.json')}")
    print(f"  Feature stats: {os.path.join(MODEL_DIR, 'feature_stats.json')}")
    print("=" * 60)


if __name__ == "__main__":
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    main()