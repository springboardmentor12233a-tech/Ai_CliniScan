# 🩺 CliniScan: AI-Powered Chest X-ray Analysis System

An end-to-end deep learning pipeline for **disease classification and localization in chest X-ray images** (Infosys Springboard Project).

---

## 🚀 Features

- 🧠 Disease Classification (ResNet50)
- 🎯 Object Detection (YOLOv11)
- 🔍 Explainability (Grad-CAM)
- 🌐 Flask Web App Deployment
- 🤖 AI Medical Summary Generator

---

## 📌 Project Overview

CliniScan helps detect **thoracic abnormalities** from chest X-rays.

✔ Detects abnormalities  
✔ Draws bounding boxes  
✔ Provides confidence scores  
✔ Generates AI-based medical suggestions  

---

## 📊 Dataset

- **VinBigData Chest X-ray Dataset** (VinDr-CXR)
- 14 disease categories
- Bounding box annotations

---

## ⚙️ Methodology

### 1️⃣ Data Preprocessing
- DICOM → JPEG conversion
- CSV → YOLO format conversion
- Normalization of bounding boxes
- Train/validation split

---

### 2️⃣ Detection Model (YOLOv11)
- **Model:** YOLOv11 (Ultralytics)
- **Image size:** 1024 × 1024
- **Epochs:** 30–40
- **Batch Size:** 8
- **Output:** Bounding boxes & Confidence

---

### 3️⃣ Classification Model (ResNet50)
- **Model:** ResNet50 (Pretrained on ImageNet)
- **Input Size:** 224 × 224
- **Batch Size:** 16
- **Loss Function:** CrossEntropyLoss
- **Optimizer:** Adam
- **Optimization:** Albumentations data augmentation, learning rate scheduling
- **Output:** Disease label & Probability

---

### 4️⃣ Explainability
- Grad-CAM for visual explanations
- Heatmaps highlight clinically relevant areas to improve interpretability and trust.

---

## 📸 Training Samples

### 🧪 Training Batches

![Train Batch](train_batch2.jpg)
![Train Batch](train_batch56250.jpg)
![Train Batch](train_batch56251.jpg)
![Train Batch](train_batch56252.jpg)

---

## 🔍 Validation Results

### 🟢 Ground Truth (Labels)

![Val Labels](val_batch0_labels.jpg)
![Val Labels](val_batch1_labels.jpg)
![Val Labels](val_batch2_labels.jpg)

---

### 🔵 Model Predictions

![Val Predictions](val_batch0_pred.jpg)
![Val Predictions](val_batch1_pred.jpg)
![Val Predictions](val_batch2_pred.jpg)

---

## 📈 Results & Evaluation

### 📊 Detection Performance Metrics (YOLOv11)

- **Precision:** 0.46
- **Recall:** 0.42
- **mAP@50:** 0.42  
- **mAP@50-95:** 0.39

### 📊 Classification Performance Metrics (ResNet50)

- **Accuracy:** 0.75
- **F1 Score (Macro):** 0.67
- **ROC-AUC:** 0.98

---

### 📉 Precision-Recall Curve

![PR Curve](BoxPR_curve.png)

---

### 📉 F1 Curve

![F1 Curve](BoxF1_curve.png)

---

### 📉 Precision Curve

![Precision Curve](BoxP_curve.png)

---

### 📉 Recall Curve

![Recall Curve](BoxR_curve.png)

---

### 📊 Confusion Matrix

#### Normalized
![Confusion Matrix Normalized](confusion_matrix_normalized.png)

#### Raw
![Confusion Matrix](confusion_matrix.png)

---

### 📊 Training Results

![Training Results](results.png)

---

## 🔍 Observations

✔ Good detection for:
- Cardiomegaly  
- Aortic enlargement  

❌ Weak detection for:
- Small lesions  
- Rare classes  

⚠️ Issue:
- High background predictions (false negatives)

---

## ⚠️ Challenges Faced

### ❌ Dataset Issues
- Missing labels
- Incorrect bounding boxes

✅ Fixed by:
- Proper CSV → YOLO conversion

---

### ❌ Training Errors
- “No labels found”

✅ Fixed dataset structure:

```
images/train
labels/train
```

---

### ❌ Grad-CAM Import Error

```python
ModuleNotFoundError: pytorch_grad_cam
```

✅ Fix:

```bash
pip install grad-cam
```

---

## 🧠 Strengths

✔ Good localization  
✔ Explainable predictions  
✔ End-to-end pipeline  

---

## ⚡ Limitations

❌ Needs more training  
❌ Struggles with small objects  
❌ Moderate accuracy  

---

## 🌐 Deployment

### 🔹 Flask App

- Upload X-ray
- Get prediction
- View bounding box
- AI summary

---

## 🏗️ Project Structure

```
project/
│
├── models/
├── dataset/
├── static/
├── templates/
├── app.py
├── pipeline.py
├── README.md
```

---

## 🚀 Future Improvements

- Improve mAP score
- Add more data
- Use ensemble models
- Deploy to cloud (AWS)

---

## 📌 Conclusion

This project shows how AI can:

- Assist in medical diagnosis
- Detect diseases early
- Provide explainable results

---

## 👨‍💻 Author

**Gudla Sai Ganesh**  
BTech CSE (AI & Data Engineering)

---

## 🗃️ Dataset Exploration

The following dataframe denotes a sample of the `train.csv` (from VinBigData Chest X-ray Abnormalities Detection), it contains:

- **image_id**: Denotes the filename of the image, it should be noted that it is common to see redundant records of the same image_id, which is expected; since each patient could have multiple diseases, also, an image can be reviewed by more than one radiologist
- **class_name**: Specify whether there is a disease detected or not. If the same image contains, it should be added as a new record rather than concatenating it using a delimiter
- **class_id**: Discrete representation of ID for the class_name
- **rad_id**: Identifier of the radiologist, multiple radiologists can review the same image
- **x_min**: x-axis value of the bottom-left point
- **y_min**: y-axis value of the bottom-left point
- **x_max**: x-axis value of the top-right point
- **y_max**: y-axis value of the top-right point

| image_id | class_name | class_id | rad_id | x_min | y_min | x_max | y_max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 50a418190bc3fb1ef1633bf9678929b3 | No finding | 14 | R11 | nan | nan | nan | nan |
| 21a10246a5ec7af151081d0cd6d65dc9 | No finding | 14 | R7 | nan | nan | nan | nan |
| 9a5094b2563a1ef3ff50dc5c7ff71345 | Cardiomegaly | 3 | R10 | 691 | 1375 | 1653 | 1831 |
| 051132a778e61a86eb147c7c6f564dfe | Aortic enlargement | 0 | R10 | 1264 | 743 | 1611 | 1019 |
| 063319de25ce7edb9b1c6b8881290140 | No finding | 14 | R10 | nan | nan | nan | nan |

The following dataframe denotes a sample of the `train_meta.csv` (from VinBigData Chest X-ray Resized PNG (1024x1024)), it contains:

- **image_id**: Denotes the filename of the image
- **dim0**: Denotes the height (can be confusing; dim0 should have been represented as the width)
- **dim1**: Denotes the width

| image_id | dim0 | dim1 |
| :--- | :--- | :--- |
| 4d390e07733ba06e5ff07412f09c0a92 | 3000 | 3000 |
| 289f69f6462af4933308c275d07060f0 | 3072 | 3072 |
| 68335ee73e67706aa59b8b55b54b11a4 | 2836 | 2336 |
| 7ecd6f67f649f26c05805c8359f9e528 | 2952 | 2744 |
| 2229148faa205e881cf0d932755c9e40 | 2880 | 2304 |

*Note: Coordinate values of the bounding box can contain null values when the class_name is No finding (class_id is 14); since there are no diseases to capture.*
