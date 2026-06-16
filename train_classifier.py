"""
SAR PATCH CLASSIFIER — PYTORCH TRAINING PIPELINE
==================================================
Classifies 256x256 SAR patches into 3 classes:
  0 → None
  1 → Urban Dense
  2 → Hills Rural Sparse

Input:
  - ISRO_14/patches_png/patch_00000.png ... patch_00299.png  (300 images)
  - ISRO_14/Labels/L1-0_99.xlsx         (patches 0-99)
  - ISRO_14/Labels/L2-100_199.xlsx      (patches 100-199)
  - ISRO_14/Labels/L3-200_299.xlsx      (patches 200-299)

Output:
  - ISRO_14/model/best_model.pth        (saved best model weights)
  - ISRO_14/model/training_results.png  (loss/accuracy curves)
  - ISRO_14/model/confusion_matrix.png  (per-class accuracy)
  - ISRO_14/model/label_encoder.json    (class name ↔ index mapping)
"""

import os
import json
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

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_DIR     = r"C:\Users\shiva\OneDrive\Documents\ISRO_14"
PATCHES_DIR  = os.path.join(BASE_DIR, "patches_png")
LABELS_DIR   = os.path.join(BASE_DIR, "Labels")
MODEL_DIR    = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# Training hyperparameters
BATCH_SIZE   = 16
EPOCHS       = 50
LR           = 1e-4          # learning rate
IMG_SIZE     = 224            # resize to 224x224 (standard for pretrained models)
VAL_SPLIT    = 0.2            # 80% train, 20% val
RANDOM_SEED  = 42

# Class names — must match exactly what's in your Excel 'class' column
CLASS_NAMES  = ["None", "Urban Dense", "Hills Rural"]
NUM_CLASSES  = len(CLASS_NAMES)

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ─────────────────────────────────────────────
# STEP 1: LOAD AND MERGE LABELS FROM 3 EXCELS
# ─────────────────────────────────────────────
def load_labels():
    """
    Reads all 3 Excel files from Labels/ folder,
    merges them, and returns a clean DataFrame with
    columns: [image_name, class, label_id]
    """
    excel_files = sorted(glob.glob(os.path.join(LABELS_DIR, "*.csv")))
    file_type = "csv"
    if not excel_files:
        excel_files = sorted(glob.glob(os.path.join(LABELS_DIR, "*.xlsx")))
        file_type = "xlsx"
    if not excel_files:
        raise FileNotFoundError(f"No .csv or .xlsx files found in {LABELS_DIR}")

    print(f"\nFound {len(excel_files)} label files ({file_type}):")
    dfs = []
    for f in excel_files:
        print(f"  Loading: {os.path.basename(f)}")
        df = pd.read_csv(f) if file_type == "csv" else pd.read_excel(f)
        print(f"    Columns: {list(df.columns)}")
        print(f"    Rows: {len(df)}")
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal rows after merge: {len(merged)}")

    # Normalize column names (strip spaces, lowercase for matching)
    merged.columns = merged.columns.str.strip()

    # Find image_name and class columns
    # (handles slight column name variations)
    col_map = {}
    for col in merged.columns:
        cl = col.lower().replace(" ", "_")
        if "image" in cl and "name" in cl:
            col_map["image_name"] = col
        elif cl == "class":
            col_map["class"] = col

    if "image_name" not in col_map or "class" not in col_map:
        print(f"  All columns found: {list(merged.columns)}")
        raise ValueError(
            "Could not find 'image_name' and 'class' columns. "
            f"Columns found: {list(merged.columns)}"
        )

    merged = merged.rename(columns={
        col_map["image_name"]: "image_name",
        col_map["class"]: "class"
    })

    # Keep only needed columns
    merged = merged[["image_name", "class"]].copy()

    # Strip whitespace from values
    merged["image_name"] = merged["image_name"].str.strip()
    merged["class"]      = merged["class"].str.strip()

    # Drop rows with missing labels
    # before = len(merged)
    # merged = merged.dropna(subset=["image_name", "class"])
    # if len(merged) < before:
    #     print(f"  Dropped {before - len(merged)} rows with missing values.")
    # Fill "None" label BEFORE dropping NaN
    # pandas reads the text "None" as NaN — we need to restore it
    merged["class"] = merged["class"].fillna("None")

    # Now only drop rows where image_name itself is missing
    before = len(merged)
    merged = merged.dropna(subset=["image_name"])
    if len(merged) < before:
        print(f"  Dropped {before - len(merged)} rows with missing image_name.")

    # Map class names → integer IDs
    # Fuzzy matching to handle slight label name variations
    def normalize_class(c):
        c = str(c).strip()
        cl = c.lower()
        if "urban" in cl or "dense" in cl:
            return "Urban Dense"
        elif "hill" in cl or "rural" in cl or "sparse" in cl:
            return "Hills Rural"
        elif "none" in cl or c == "" or c == "nan":
            return "None"
        else:
            return c  # keep as-is, will error later if unknown

    merged["class"] = merged["class"].apply(normalize_class)

    class_to_id = {name: i for i, name in enumerate(CLASS_NAMES)}
    merged["label_id"] = merged["class"].map(class_to_id)

    unknown = merged[merged["label_id"].isna()]
    if len(unknown) > 0:
        print(f"\n[WARN] Unknown class values found:")
        print(unknown["class"].value_counts())
        merged = merged.dropna(subset=["label_id"])

    merged["label_id"] = merged["label_id"].astype(int)

    print(f"\nClass distribution:")
    for name, count in merged["class"].value_counts().items():
        print(f"  {name}: {count} patches")

    return merged


# ─────────────────────────────────────────────
# STEP 2: PYTORCH DATASET
# ─────────────────────────────────────────────
class SARPatchDataset(Dataset):
    """
    PyTorch Dataset for SAR patch PNG images.
    Loads images from patches_png/ and returns (image_tensor, label_id).
    """
    def __init__(self, df, patches_dir, transform=None):
        self.df          = df.reset_index(drop=True)
        self.patches_dir = patches_dir
        self.transform   = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row       = self.df.iloc[idx]
        img_name  = row["image_name"]
        label     = int(row["label_id"])

        img_path  = os.path.join(self.patches_dir, img_name)
        img       = Image.open(img_path).convert("RGB")  # (256,256,3)

        if self.transform:
            img = self.transform(img)

        return img, label


# ─────────────────────────────────────────────
# STEP 3: DATA AUGMENTATION & TRANSFORMS
# ─────────────────────────────────────────────
def get_transforms():
    """
    Training: augment to prevent overfitting on small 300-sample dataset.
    Validation: only resize+normalize (no random augmentation).
    """
    # ImageNet mean/std — used because we load a pretrained model
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),

        # Augmentation for small dataset
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),

        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_transform, val_transform


# ─────────────────────────────────────────────
# STEP 4: MODEL — EfficientNet-B0 (best for small datasets)
# ─────────────────────────────────────────────
def build_model(num_classes):
    """
    EfficientNet-B0 pretrained on ImageNet, fine-tuned for our 3 classes.

    Why EfficientNet-B0?
    - Small (~5M params) → won't overfit on 240 training samples
    - Pretrained on ImageNet → already knows edges, textures, shapes
    - Better accuracy/parameter ratio than ResNet for small datasets
    - Transfer learning: we replace only the final classifier layer

    Architecture:
    ImageNet pretrained EfficientNet-B0 backbone
        → Global Average Pooling
        → Dropout(0.3)
        → Linear(1280, 3)   [our 3 classes]
        → Softmax (implicit in CrossEntropyLoss)
    """
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")

    # Freeze all backbone layers first (optional — unfreeze later)
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier head with our 3-class head
    in_features = model.classifier[1].in_features  # 1280 for EfficientNet-B0
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes)
    )

    # Unfreeze classifier head for training
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model


# ─────────────────────────────────────────────
# STEP 5: TRAINING LOOP
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds       = outputs.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += images.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


# ─────────────────────────────────────────────
# STEP 6: PLOT RESULTS
# ─────────────────────────────────────────────
def plot_training_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Results", fontsize=14)

    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="steelblue")
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss",   color="coral")
    axes[0].set_title("Loss per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="Train Acc", color="steelblue")
    axes[1].plot(epochs, history["val_acc"],   label="Val Acc",   color="coral")
    axes[1].set_title("Accuracy per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(MODEL_DIR, "training_results.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.show()


def plot_confusion_matrix(all_labels, all_preds, class_names):
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax
    )
    ax.set_title("Confusion Matrix (Validation Set)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    out = os.path.join(MODEL_DIR, "confusion_matrix.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("="*60)
    print("SAR PATCH CLASSIFIER — TRAINING")
    print("="*60)

    # --- Load labels ---
    print("\n[1] Loading labels from Excel files...")
    df = load_labels()

    # Verify image files exist
    missing = []
    for img_name in df["image_name"]:
        if not os.path.exists(os.path.join(PATCHES_DIR, img_name)):
            missing.append(img_name)
    if missing:
        print(f"\n[WARN] {len(missing)} image files not found, e.g.: {missing[:3]}")
        df = df[~df["image_name"].isin(missing)]
    print(f"  Final dataset size: {len(df)} labeled patches")

    # Save label encoder
    label_enc = {name: i for i, name in enumerate(CLASS_NAMES)}
    with open(os.path.join(MODEL_DIR, "label_encoder.json"), "w") as f:
        json.dump(label_enc, f, indent=2)
    print(f"  Label encoder: {label_enc}")

    # --- Split train/val ---
    print(f"\n[2] Splitting dataset (80% train, 20% val)...")
    train_df, val_df = train_test_split(
        df, test_size=VAL_SPLIT, random_state=RANDOM_SEED,
        stratify=df["label_id"]   # ensure equal class ratio in both splits
    )
    print(f"  Train: {len(train_df)} patches")
    print(f"  Val:   {len(val_df)} patches")

    # --- Transforms ---
    train_transform, val_transform = get_transforms()

    # --- Datasets & DataLoaders ---
    train_dataset = SARPatchDataset(train_df, PATCHES_DIR, transform=train_transform)
    val_dataset   = SARPatchDataset(val_df,   PATCHES_DIR, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # --- Model ---
    print(f"\n[3] Building EfficientNet-B0 model...")
    model = build_model(NUM_CLASSES).to(DEVICE)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # --- Loss & Optimizer ---
    # Compute class weights to handle class imbalance
    class_counts = df["label_id"].value_counts().sort_index().values
    class_weights = torch.tensor(
        1.0 / class_counts, dtype=torch.float32
    ).to(DEVICE)
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=1e-4
    )

    # Scheduler: reduce LR when val_loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # --- Training ---
    print(f"\n[4] Training for {EPOCHS} epochs on {DEVICE}...")
    print(f"  Batch size: {BATCH_SIZE}, LR: {LR}")
    print("-"*60)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc  = 0.0
    best_model_path = os.path.join(MODEL_DIR, "best_model.pth")

    # Phase 1 (epochs 1-20): train only classifier head (backbone frozen)
    # Phase 2 (epochs 21+): unfreeze backbone for fine-tuning
    for epoch in range(1, EPOCHS + 1):

        # Unfreeze backbone at epoch 21 for fine-tuning
        if epoch == 21:
            print("\n  [Unfreezing backbone for fine-tuning from epoch 21...]")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(
                model.parameters(), lr=LR * 0.1, weight_decay=1e-4
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5
            )

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, DEVICE)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"  Epoch {epoch:3d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "class_names": CLASS_NAMES,
            }, best_model_path)
            print(f"    ✓ Best model saved (val_acc={val_acc:.3f})")

    # --- Final evaluation ---
    print(f"\n[5] Final evaluation on validation set...")
    print(f"  Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.1f}%)")

    print("\nClassification Report:")
    print(classification_report(val_labels, val_preds, target_names=CLASS_NAMES))

    # --- Plots ---
    print("\n[6] Saving plots...")
    plot_training_curves(history)
    plot_confusion_matrix(val_labels, val_preds, CLASS_NAMES)

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print(f"Best model saved to: {best_model_path}")
    print(f"Best validation accuracy: {best_val_acc*100:.1f}%")
    print("\nNext step: Run predict.py to classify new patches.")
    print("="*60)


if __name__ == "__main__":
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    main()