"""
PREDICT & EVALUATE — TEST ON 100 LABELED IMAGES
=================================================
Uses trained best_model.pth to classify 100 SAR patches
and compares against ground truth labels from CSV.

Folder structure expected:
ISRO_14/
├── model/
│   ├── best_model.pth
│   └── label_encoder.json
├── ISRO_2/
│   ├── Images/          ← 100 test PNG images
│   └── ISRO_2_CSV.csv   ← ground truth labels
"""

import os
import json
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ─────────────────────────────────────────────
# CONFIG — change BASE_DIR if needed
# ─────────────────────────────────────────────
BASE_DIR       = r"C:\Users\shiva\OneDrive\Documents\ISRO_14"
TEST_IMG_DIR   = os.path.join(BASE_DIR, "ISRO_2", "Images")
TEST_LABEL_CSV = os.path.join(BASE_DIR, "ISRO_2", "ISRO_2_CSV.csv")
MODEL_DIR      = os.path.join(BASE_DIR, "model")
MODEL_PATH     = os.path.join(MODEL_DIR, "best_model.pth")
LABEL_PATH     = os.path.join(MODEL_DIR, "label_encoder.json")
OUTPUT_CSV     = os.path.join(MODEL_DIR, "test_predictions.csv")
IMG_SIZE       = 224
N_IMAGES       = 100

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
def load_model(model_path, num_classes):
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes)
    )
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    print(f"Model loaded: {model_path}")
    print(f"Best val accuracy during training: {checkpoint['val_acc']*100:.1f}%")
    return model, checkpoint["class_names"]


# ─────────────────────────────────────────────
# TRANSFORM
# ─────────────────────────────────────────────
def get_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


# ─────────────────────────────────────────────
# PREDICT SINGLE IMAGE
# ─────────────────────────────────────────────
def predict_single(model, img_path, transform, class_names):
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred = int(probs.argmax())
    return class_names[pred], probs


# ─────────────────────────────────────────────
# LOAD GROUND TRUTH LABELS
# ─────────────────────────────────────────────
def load_ground_truth(csv_path, img_dir, n_images):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    print(f"\nCSV columns: {list(df.columns)}")

    # Find image_name column — exact match first, then fallback
    img_col = None
    for col in df.columns:
        if col.lower().strip() == "image_name":
            img_col = col
            break
    if img_col is None:
        for col in df.columns:
            if "name" in col.lower() and "unnamed" not in col.lower():
                img_col = col
                break
    if img_col is None:
        raise ValueError(
            f"Cannot find image_name column. Columns found: {list(df.columns)}"
        )

    print(f"Using image column : '{img_col}'")
    print(f"Using class column : 'class'")

    # Fix: pandas reads the text "None" as NaN — restore it
    df["class"] = df["class"].fillna("None").astype(str).str.strip()
    df[img_col] = df[img_col].fillna("").astype(str).str.strip()

    # Rename and keep only needed columns
    df = df.rename(columns={img_col: "image_name"})
    df = df[["image_name", "class"]].copy()

    # Drop rows where image_name is empty or NaN
    df = df[df["image_name"].str.len() > 0].reset_index(drop=True)

    # Normalize class names — handles slight label variations
    def normalize_class(c):
        cl = c.lower()
        if "urban" in cl or "dense" in cl:
            return "Urban Dense"
        elif "hill" in cl or "rural" in cl or "sparse" in cl:
            return "Hills Rural"
        else:
            return "None"

    df["true_class"] = df["class"].apply(normalize_class)

    # Keep only rows where image file actually exists on disk
    valid = []
    missing = []
    for _, row in df.iterrows():
        img_path = os.path.join(img_dir, row["image_name"])
        if os.path.exists(img_path):
            valid.append(row)
        else:
            missing.append(row["image_name"])

    if missing:
        print(f"\n[WARN] {len(missing)} image files not found on disk.")
        print(f"  First 3 missing: {missing[:3]}")

    df_valid = pd.DataFrame(valid).reset_index(drop=True).head(n_images)

    print(f"\nGround truth loaded: {len(df_valid)} images")
    print("Class distribution:")
    print(df_valid["true_class"].value_counts().to_string())

    return df_valid


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("SAR PATCH CLASSIFIER — TESTING ON 100 IMAGES")
    print("=" * 60)

    # Load label encoder
    with open(LABEL_PATH) as f:
        label_enc = json.load(f)
    num_classes = len(label_enc)
    class_names = list(label_enc.keys())
    print(f"\nClasses: {class_names}")

    # Load model and transform
    model, _ = load_model(MODEL_PATH, num_classes)
    transform = get_transform()

    # Load ground truth labels
    gt_df = load_ground_truth(TEST_LABEL_CSV, TEST_IMG_DIR, N_IMAGES)

    # Run predictions on all images
    print(f"\nRunning predictions on {len(gt_df)} images...")
    results = []
    for i, row in gt_df.iterrows():
        img_path = os.path.join(TEST_IMG_DIR, row["image_name"])
        pred_class, probs = predict_single(model, img_path, transform, class_names)
        results.append({
            "image_name":      row["image_name"],
            "true_class":      row["true_class"],
            "predicted_class": pred_class,
            "correct":         row["true_class"] == pred_class,
            "confidence":      float(probs.max()),
            **{f"prob_{name}": float(probs[j])
               for j, name in enumerate(class_names)},
        })
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(gt_df)} done...")

    df = pd.DataFrame(results)

    # Save full results to CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nFull results saved: {OUTPUT_CSV}")

    # Overall accuracy
    overall_acc = df["correct"].mean()
    print(f"\n{'=' * 60}")
    print(f"OVERALL ACCURACY: {overall_acc * 100:.1f}%  "
          f"({df['correct'].sum()}/{len(df)} correct)")
    print(f"{'=' * 60}")

    # Per-class accuracy
    print(f"\nPer-Class Accuracy:")
    for cls in class_names:
        subset = df[df["true_class"] == cls]
        if len(subset) == 0:
            print(f"  {cls:20s}: no samples in test set")
            continue
        acc = subset["correct"].mean()
        print(f"  {cls:20s}: {acc * 100:.1f}%  "
              f"({subset['correct'].sum()}/{len(subset)} correct)")

    print(f"\nMean Confidence: {df['confidence'].mean() * 100:.1f}%")

    # Classification report
    true_labels = df["true_class"].tolist()
    pred_labels = df["predicted_class"].tolist()
    present_classes = sorted(set(true_labels + pred_labels))
    print(f"\nClassification Report:")
    print(classification_report(
        true_labels, pred_labels,
        labels=present_classes,
        zero_division=0
    ))

    # Confusion matrix plot
    cm = confusion_matrix(true_labels, pred_labels, labels=present_classes)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=present_classes,
        yticklabels=present_classes,
        ax=ax
    )
    ax.set_title(
        f"Confusion Matrix — Test Set (n={len(df)})\n"
        f"Overall Accuracy: {overall_acc * 100:.1f}%"
    )
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    plt.tight_layout()
    cm_path = os.path.join(MODEL_DIR, "test_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    print(f"Confusion matrix saved: {cm_path}")
    plt.show()

    # Visual sample grid — green border = correct, red border = wrong
    sample = df.sample(min(9, len(df)), random_state=42)
    fig, axes = plt.subplots(3, 3, figsize=(14, 14))
    fig.suptitle(
        f"Sample Predictions  |  Green = Correct  |  Red = Wrong\n"
        f"Overall Accuracy: {overall_acc * 100:.1f}%",
        fontsize=13
    )
    for ax, (_, row) in zip(axes.flatten(), sample.iterrows()):
        img = Image.open(os.path.join(TEST_IMG_DIR, row["image_name"]))
        ax.imshow(img)
        color = "green" if row["correct"] else "red"
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(5)
        ax.set_title(
            f"{row['image_name']}\n"
            f"True:  {row['true_class']}\n"
            f"Pred:  {row['predicted_class']}  ({row['confidence'] * 100:.0f}%)",
            fontsize=8,
            color=color
        )
        ax.axis("off")

    for ax in axes.flatten()[len(sample):]:
        ax.axis("off")

    plt.tight_layout()
    grid_path = os.path.join(MODEL_DIR, "test_sample_grid.png")
    plt.savefig(grid_path, dpi=120, bbox_inches="tight")
    print(f"Sample grid saved: {grid_path}")
    plt.show()

    print(f"\n{'=' * 60}")
    print("TESTING COMPLETE!")
    print(f"Results CSV : {OUTPUT_CSV}")
    print(f"Confusion   : {cm_path}")
    print(f"Sample grid : {grid_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()