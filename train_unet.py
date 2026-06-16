"""
TRAIN U-NET FOR SAR SEMANTIC SEGMENTATION
============================================
Trains a U-Net (ResNet34 encoder, ImageNet-pretrained) to segment
SAR patches pixel-by-pixel into 3 classes:
    0 = Background / None
    1 = Urban Dense
    2 = Hills Rural

Input:
    ISRO_14/seg_to_label/*.png       <- the 50 patches you labeled
    ISRO_14/masks_seg/*_mask.png     <- matching masks (same name + _mask)

Output:
    ISRO_14/model_seg/best_unet.pth
    ISRO_14/model_seg/training_curves.png
    ISRO_14/model_seg/val_predictions.png

Usage:
    python train_unet.py
"""

import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt

try:
    import segmentation_models_pytorch as smp
except ImportError:
    print("ERROR: segmentation_models_pytorch not installed.")
    print("Run: pip install segmentation-models-pytorch")
    raise SystemExit(1)

# ─────────────────────────────────────────────
BASE_DIR  = r"C:\Users\shiva\OneDrive\Documents\ISRO_14"
IMG_DIR   = os.path.join(BASE_DIR, "seg_to_label")
MASK_DIR  = os.path.join(BASE_DIR, "masks_seg")
MODEL_DIR = os.path.join(BASE_DIR, "model_seg")
os.makedirs(MODEL_DIR, exist_ok=True)

NUM_CLASSES = 3
CLASS_NAMES = ["Background", "Urban Dense", "Hills Rural"]

BATCH_SIZE      = 4
EPOCHS_FROZEN   = 20   # decoder only, encoder frozen
EPOCHS_UNFROZEN = 40   # full network fine-tune
LR_FROZEN       = 1e-3
LR_UNFROZEN     = 1e-4
VAL_FRACTION    = 0.2
SEED            = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(SEED)
torch.manual_seed(SEED)


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class SARSegDataset(Dataset):
    def __init__(self, pairs, augment=False):
        self.pairs = pairs
        self.augment = augment
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
        mask = np.array(Image.open(mask_path), dtype=np.int64)

        if self.augment:
            img, mask = self._augment(img, mask)

        img = (img - self.mean) / self.std
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        mask = torch.from_numpy(mask.copy()).long()
        return img, mask

    def _augment(self, img, mask):
        if np.random.rand() < 0.5:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()
        if np.random.rand() < 0.5:
            img = np.flipud(img).copy()
            mask = np.flipud(mask).copy()
        k = np.random.randint(0, 4)
        if k > 0:
            img = np.rot90(img, k).copy()
            mask = np.rot90(mask, k).copy()
        return img, mask


def build_pairs():
    mask_files = sorted(glob.glob(os.path.join(MASK_DIR, "*_mask.png")))
    pairs = []
    for mp in mask_files:
        img_name = os.path.basename(mp).replace("_mask.png", ".png")
        ip = os.path.join(IMG_DIR, img_name)
        if os.path.exists(ip):
            pairs.append((ip, mp))
        else:
            print(f"  [WARN] No matching image for {mp}")
    return pairs


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
def compute_iou(pred, target, num_classes):
    pred = pred.flatten()
    target = target.flatten()
    ious = []
    for c in range(num_classes):
        pred_c = (pred == c)
        target_c = (target == c)
        intersection = (pred_c & target_c).sum()
        union = (pred_c | target_c).sum()
        ious.append(float("nan") if union == 0 else intersection / union)
    return ious


# ─────────────────────────────────────────────
# LOSS — CrossEntropy + Dice combined
# Dice handles class imbalance better than CE alone; CE gives stable
# gradients early on. Combining both is standard practice for small,
# imbalanced segmentation datasets like this one.
# ─────────────────────────────────────────────
class DiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = (probs * targets_onehot).sum(dims)
        cardinality = (probs + targets_onehot).sum(dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, class_weights, num_classes):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = DiceLoss(num_classes)

    def forward(self, logits, targets):
        return 0.5 * self.ce(logits, targets) + 0.5 * self.dice(logits, targets)


# ─────────────────────────────────────────────
# TRAIN / VAL LOOP
# ─────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_ious = []
    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for imgs, masks in loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            if is_train:
                optimizer.zero_grad()

            logits = model(imgs)
            loss = criterion(logits, masks)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)

            preds = logits.argmax(dim=1).cpu().numpy()
            targets_np = masks.cpu().numpy()
            for p, t in zip(preds, targets_np):
                all_ious.append(compute_iou(p, t, NUM_CLASSES))

    avg_loss = total_loss / len(loader.dataset)
    all_ious = np.array(all_ious, dtype=np.float32)
    iou_per_class = np.nanmean(all_ious, axis=0)
    mean_iou = np.nanmean(iou_per_class)
    return avg_loss, mean_iou, iou_per_class


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("SAR PATCH SEGMENTATION — U-NET TRAINING")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    print("\n[1] Building image/mask pairs...")
    pairs = build_pairs()
    print(f"  Found {len(pairs)} labeled pairs.")
    if len(pairs) < 10:
        print("  WARNING: very few labeled samples — results will be noisy.")

    np.random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * VAL_FRACTION))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]
    print(f"  Train: {len(train_pairs)}   Val: {len(val_pairs)}")

    train_ds = SARSegDataset(train_pairs, augment=True)
    val_ds = SARSegDataset(val_pairs, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print("\n[2] Computing class pixel weights (handles imbalance)...")
    class_pixel_counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    for _, mp in train_pairs:
        m = np.array(Image.open(mp))
        for c in range(NUM_CLASSES):
            class_pixel_counts[c] += (m == c).sum()
    class_weights = class_pixel_counts.sum() / (NUM_CLASSES * (class_pixel_counts + 1e-6))
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    print(f"  Pixel counts: {dict(zip(CLASS_NAMES, class_pixel_counts.astype(int)))}")
    print(f"  Class weights: {dict(zip(CLASS_NAMES, [round(w, 3) for w in class_weights.cpu().numpy()]))}")

    print("\n[3] Building U-Net (ResNet34 encoder, ImageNet-pretrained)...")
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES,
    ).to(DEVICE)

    criterion = CombinedLoss(class_weights, NUM_CLASSES)
    history = {"train_loss": [], "val_loss": [], "train_iou": [], "val_iou": []}
    best_val_iou = -1.0
    best_path = os.path.join(MODEL_DIR, "best_unet.pth")

    def save_if_best(val_iou, val_iou_per_class, epoch, phase):
        nonlocal best_val_iou
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_iou": val_iou,
                "val_iou_per_class": val_iou_per_class.tolist(),
                "class_names": CLASS_NAMES,
                "epoch": epoch,
                "phase": phase,
            }, best_path)

    # ── Phase 1: decoder only, encoder frozen ──
    print(f"\n[4] Phase 1: training decoder ({EPOCHS_FROZEN} epochs, encoder frozen)...")
    for p in model.encoder.parameters():
        p.requires_grad = False
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR_FROZEN)

    for epoch in range(1, EPOCHS_FROZEN + 1):
        tl, ti, _ = run_epoch(model, train_loader, criterion, optimizer)
        vl, vi, vipc = run_epoch(model, val_loader, criterion)
        history["train_loss"].append(tl); history["val_loss"].append(vl)
        history["train_iou"].append(ti); history["val_iou"].append(vi)
        print(f"  Epoch {epoch:2d}/{EPOCHS_FROZEN}  train_loss={tl:.4f} train_iou={ti:.3f}  "
              f"val_loss={vl:.4f} val_iou={vi:.3f}")
        save_if_best(vi, vipc, epoch, "frozen")

    # ── Phase 2: fine-tune full network ──
    print(f"\n[5] Phase 2: fine-tuning full network ({EPOCHS_UNFROZEN} epochs)...")
    for p in model.encoder.parameters():
        p.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_UNFROZEN)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    for epoch in range(1, EPOCHS_UNFROZEN + 1):
        tl, ti, _ = run_epoch(model, train_loader, criterion, optimizer)
        vl, vi, vipc = run_epoch(model, val_loader, criterion)
        scheduler.step(vi)
        history["train_loss"].append(tl); history["val_loss"].append(vl)
        history["train_iou"].append(ti); history["val_iou"].append(vi)
        total_epoch = EPOCHS_FROZEN + epoch
        print(f"  Epoch {total_epoch:2d}  train_loss={tl:.4f} train_iou={ti:.3f}  "
              f"val_loss={vl:.4f} val_iou={vi:.3f}")
        save_if_best(vi, vipc, total_epoch, "unfrozen")

    print(f"\nBest validation mean IoU: {best_val_iou:.3f}")

    # ── Training curves ──
    print("\n[6] Plotting training curves...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Val")
    axes[0].axvline(EPOCHS_FROZEN, color="gray", linestyle="--", label="Unfreeze point")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()

    axes[1].plot(history["train_iou"], label="Train")
    axes[1].plot(history["val_iou"], label="Val")
    axes[1].axvline(EPOCHS_FROZEN, color="gray", linestyle="--", label="Unfreeze point")
    axes[1].set_title("Mean IoU"); axes[1].set_xlabel("Epoch"); axes[1].legend()

    plt.tight_layout()
    curve_path = os.path.join(MODEL_DIR, "training_curves.png")
    plt.savefig(curve_path, dpi=120, bbox_inches="tight")
    print(f"  Saved: {curve_path}")

    # ── Validation prediction visualizations ──
    print("\n[7] Generating validation prediction visualizations...")
    checkpoint = torch.load(best_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    COLOR_MAP = {0: [0, 0, 0], 1: [255, 0, 0], 2: [0, 255, 0]}

    def mask_to_rgb(mask):
        rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for cid, color in COLOR_MAP.items():
            rgb[mask == cid] = color
        return rgb

    n_show = min(6, len(val_pairs))
    fig, axes = plt.subplots(n_show, 3, figsize=(9, 3 * n_show))
    if n_show == 1:
        axes = axes.reshape(1, -1)

    with torch.no_grad():
        for i in range(n_show):
            img, mask = val_ds[i]
            logits = model(img.unsqueeze(0).to(DEVICE))
            pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()

            orig_img = np.array(Image.open(val_pairs[i][0]).convert("RGB"))

            axes[i, 0].imshow(orig_img); axes[i, 0].set_title("Original"); axes[i, 0].axis("off")
            axes[i, 1].imshow(mask_to_rgb(mask.numpy())); axes[i, 1].set_title("Ground Truth"); axes[i, 1].axis("off")
            axes[i, 2].imshow(mask_to_rgb(pred)); axes[i, 2].set_title("Prediction"); axes[i, 2].axis("off")

    plt.tight_layout()
    pred_path = os.path.join(MODEL_DIR, "val_predictions.png")
    plt.savefig(pred_path, dpi=120, bbox_inches="tight")
    print(f"  Saved: {pred_path}")

    print("\n" + "=" * 60)
    print("DONE! Model saved to:", best_path)
    print(f"Per-class IoU at best epoch: "
          f"{dict(zip(CLASS_NAMES, [round(x, 3) for x in checkpoint['val_iou_per_class']]))}")
    print("=" * 60)


if __name__ == "__main__":
    main()