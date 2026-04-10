# CliniScan — Chest X-Ray Disease Detection & Classification

An end-to-end deep learning pipeline for automated detection and classification of 14 thoracic pathologies from chest X-rays, built on Kaggle using the CliniScan2 dataset.

---

## Overview

CliniScan combines two complementary models:

- **Classification** — ResNet50 fine-tuned with staged training to predict the primary pathology in an X-ray image.
- **Object Detection** — YOLOv11s trained to localize and label disease regions with bounding boxes.

The final milestone (M4) adds a **Gradio web interface** that runs both models together: given an input X-ray, it returns the predicted class, confidence scores, bounding box overlays, and a Grad-CAM heatmap highlighting the regions driving the classification.

---

## Detected Pathologies (14 Classes)

| ID | Class |
|----|-------|
| 0 | Aortic enlargement |
| 1 | Atelectasis |
| 2 | Calcification |
| 3 | Cardiomegaly |
| 4 | Consolidation |
| 5 | ILD |
| 6 | Infiltration |
| 7 | Lung Opacity |
| 8 | Nodule/Mass |
| 9 | Other lesion |
| 10 | Pleural effusion |
| 11 | Pleural thickening |
| 12 | Pneumothorax |
| 13 | Pulmonary fibrosis |

> **Note:** Class 14 ("No finding") is excluded from training.

---

## Project Structure (Milestones)

The notebook is organized into four milestones, each building on the previous:

### Milestone 1 — Baseline YOLO Detection
- Converts bounding box annotations from CSV to YOLO format (normalized `xc, yc, w, h`).
- Copies training images and creates empty label files for unannotated images.
- Writes `data.yaml` with class definitions.
- Trains `yolo11s.pt` for 25 epochs at 1024px resolution.

### Milestone 2 — Baseline ResNet50 Classification
- Single-label mapping: one dominant class per image.
- Custom `XrayDataset` PyTorch Dataset with basic `Resize + ToTensor` transforms.
- ResNet50 (pretrained) with a replaced FC head (2048 → 14).
- Trains for 5 epochs with Adam (`lr=1e-4`) and CrossEntropyLoss.
- Evaluates with Accuracy, Macro F1, and AUC (OvR).
- Saves model weights and class name JSON to `/kaggle/working/`.

### Milestone 3 — Improved Models
- **Classification improvements:**
  - Albumentations augmentation pipeline: HorizontalFlip, ShiftScaleRotate, RandomBrightnessContrast, CLAHE, GaussNoise, ImageNet normalization.
  - Two-phase staged fine-tuning:
    - Phase 1 (5 epochs): Freeze backbone, train FC head only (`lr=1e-3`).
    - Phase 2 (10 epochs): Unfreeze all layers, AdamW + CosineAnnealingLR (`lr=1e-5`).
  - Dropout regularization in the FC head.
  - Label smoothing (`0.1`) in CrossEntropyLoss.
  - Best model checkpoint saved by validation F1.
- **Detection improvements:**
  - Proper train/val split for the YOLO dataset.
  - Tuned hyperparameters: `lr0=0.005`, `momentum=0.937`, `weight_decay=5e-4`, cosine LR, mosaic augmentation.
  - Trains for 30 epochs at 640px.
- **Explainability:** Grad-CAM visualizations on validation samples.

### Milestone 4 — Final Evaluation & Gradio Demo
- Full evaluation with per-class metrics table, ROC curves (one-vs-rest for all 14 classes), and confusion matrix heatmap.
- Extended Grad-CAM visualization (8 random validation samples).
- YOLO final validation with `mAP@0.50` and `mAP@0.50:0.95`.
- **Gradio interface** (`cliniscan_predict`): accepts a PIL image, runs classification + detection inference, and returns annotated output with Grad-CAM overlay.

---

## Requirements

```
ultralytics
albumentations
timm
gradio
torch
torchvision
opencv-python
pandas
numpy
scikit-learn
matplotlib
seaborn
Pillow
tqdm
```

Install all at once (as used in the notebook):
```bash
pip install ultralytics albumentations timm gradio -q
```

---

## Dataset

- **Source:** [CliniScan2 on Kaggle](https://www.kaggle.com/datasets/alaagastien/cliniscan2)
- `train.csv` — annotations with columns: `image_id`, `class_id`, `x_min`, `y_min`, `x_max`, `y_max`
- `archive/train_meta.csv` — image dimensions (`dim0`, `dim1`) per `image_id`
- `archive/train/` — PNG chest X-ray images
- `archive/test/` — unlabeled test images

---

## Usage

The notebook is designed to run on **Kaggle** with GPU acceleration. Key paths assume the Kaggle environment:

```
/kaggle/input/datasets/alaagastien/cliniscan2/
/kaggle/working/
```

To run locally, update the path constants at the top of each milestone section:
```python
TRAIN_CSV = "/your/path/train.csv"
META_CSV  = "/your/path/train_meta.csv"
IMG_DIR   = "/your/path/train/"
WORK_DIR  = "/your/output/directory"
```

---

## Model Outputs

| File | Description |
|------|-------------|
| `cliniscan_classification_model.pth` | ResNet50 state dict (M2 baseline) |
| `best_cls.pth` | Best ResNet50 checkpoint by val F1 (M3/M4) |
| `class_names.json` | Class ID → name mapping |
| `yolo_m3/weights/best.pt` | Best YOLO detection weights (M3) |
| `data.yaml` / `yolo_dataset_m3/data.yaml` | YOLO dataset config |

---

## Evaluation Metrics

**Classification (ResNet50)**
- Accuracy
- Macro F1 Score
- AUC (One-vs-Rest, multi-class)
- Per-class precision, recall, F1 (via `classification_report`)
- Confusion matrix
- Per-class ROC curves

**Detection (YOLOv11s)**
- `mAP@0.50`
- `mAP@0.50:0.95`
- Precision & Recall

---

## Explainability

Grad-CAM is implemented as a custom `GradCAM` class targeting `model.layer4[-1]` of ResNet50. For each validation sample it produces three panels:
1. Original X-ray
2. Raw Grad-CAM heatmap
3. Heatmap overlaid on the image

This helps verify that the model attends to clinically relevant lung regions rather than image artifacts.
