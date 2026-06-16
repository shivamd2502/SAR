# SAR Patch Analysis — ISRO Project (ISRO_14)

Deep learning pipeline for analyzing SAR (Synthetic Aperture Radar) image patches. Two models were built: an **image-level classifier** (predicts one label per 256×256 patch) and a **pixel-level segmentation model** (predicts a class for every pixel in a patch).

**Classes:** `0 = None/Background` · `1 = Urban Dense` · `2 = Hills Rural (Sparse)`

---

## Part 1 — Patch Classification

**Goal:** classify a whole SAR patch into one of the 3 terrain classes.

### Data
- 300 SAR patches (`patches_png/`, 256×256 PNG)
- Labels merged from 3 CSV exports (`L1-0_99.csv`, `L2-100_199.csv`, `L3-200_299.csv`) — 304 raw rows, cleaned to 300 valid patches
- Class distribution: None 133 · Urban Dense 104 · Hills Rural 67
- Split: 80/20 stratified → 243 train / 61 val

### Model & Training
- **EfficientNet-B0**, ImageNet-pretrained, classifier head replaced with `Dropout(0.3) → Linear(1280, 3)`
- Two-phase training:
  - Phase 1 (epochs 1–20): backbone frozen, only head trained, LR = 1e-4
  - Phase 2 (epochs 21–50): full network unfrozen, fine-tuned at LR = 1e-5
- Weighted CrossEntropyLoss (inverse class frequency) to handle imbalance
- AdamW optimizer + ReduceLROnPlateau scheduler
- Augmentation: horizontal/vertical flip, ±15° rotation, color jitter

### Results
**Best validation accuracy: 96.7%** (epoch 44)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| None | 0.96 | 1.00 | 0.98 | 27 |
| Urban Dense | 0.95 | 0.90 | 0.93 | 21 |
| Hills Rural | 0.92 | 0.92 | 0.92 | 13 |

Overall accuracy 0.95 · macro F1 0.94 · weighted F1 0.95

**Saved artifacts:** `best_model.pth`, `training_results.png`, `confusion_matrix.png`, `label_encoder.json`

---

## Part 2 — Semantic Segmentation

**Goal:** produce a per-pixel class map for each patch (urban vs. hilly vs. background regions within a single image).

### Data
- 50 manually labeled patches with pixel masks (subset of the 300, hand-annotated)
- Roughly balanced pixel counts: Background 869,831 · Urban Dense 807,923 · Hills Rural 943,686
- Split: 40 train / 10 val

### Model & Training
- **U-Net** with **ResNet34** encoder, ImageNet-pretrained (via `segmentation_models_pytorch`)
- Loss: 0.5 × weighted CrossEntropy + 0.5 × Dice loss (handles small/imbalanced dataset better than CE alone)
- Two-phase training:
  - Phase 1 (20 epochs): decoder only, encoder frozen, LR = 1e-3
  - Phase 2 (40 epochs): full network fine-tuned, LR = 1e-4, ReduceLROnPlateau on val IoU
- Augmentation: random flips + 90° rotations

### Results
**Best validation mean IoU: 0.595** (epoch 19, still in the frozen-encoder phase)

| Class | IoU |
|---|---|
| Background | 0.690 |
| Urban Dense | 0.359 |
| Hills Rural | 0.737 |

**Saved artifacts:** `best_unet.pth`, `training_curves.png`, `val_predictions.png`

### Observations
- Urban Dense is the weakest segmentation class (IoU 0.359) even though it's well-classified at the patch level (F1 0.93) — urban boundaries are harder to delineate pixel-by-pixel than to detect at the whole-patch level, and segmentation only had 50 labeled samples vs. 300 for classification.
- The best segmentation checkpoint came from the frozen-encoder phase; fine-tuning the full network didn't beat it and val IoU drifted down/oscillated afterward — a sign the 50-sample dataset (not model capacity) is the current bottleneck, with some overfitting risk during fine-tuning.

---

## Tech Stack
PyTorch · torchvision · segmentation_models_pytorch · EfficientNet-B0 / ResNet34 encoders · pandas · scikit-learn · matplotlib · seaborn
Trained locally on an NVIDIA RTX 3050 Laptop GPU.
