# 🩺 CliniScan: AI-Based Lung Abnormality Detection System

## 📌 Project Overview
CliniScan is an advanced deep learning-based system developed to detect and localize lung abnormalities from chest X-ray images.

It combines:
- YOLOv8 → Object Detection  
- DenseNet → Multi-label Classification  
- Streamlit → Web Interface  

The system helps healthcare professionals diagnose pulmonary diseases with improved accuracy and efficiency.

---

## 🎯 Project Objective
The goal is to build an AI pipeline that can:
- Detect abnormalities like:
  - Opacities  
  - Consolidations  
  - Fibrosis  
  - Masses  
- Reduce diagnostic workload  
- Improve accuracy  
- Provide interpretable outputs  

---

## 📊 Dataset Description
- Dataset: VinDr-CXR  
- ~18,000 chest X-ray images  
- Includes bounding boxes and disease labels  
- Original format: DICOM  
- Converted to: PNG/JPEG  

---

## 🚀 Project Milestones

### 🔹 Milestone 1: Data Preparation
- Convert DICOM → PNG/JPEG  
- Process CSV annotations  
- Setup environment (PyTorch, OpenCV, NumPy, pandas)

### 🔹 Milestone 2: Model Development
- YOLOv8 for detection  
- DenseNet for classification  
- Apply data augmentation  
- Evaluate using mAP, F1-score, AUC  

### 🔹 Milestone 3: Optimization
- Hyperparameter tuning  
- Transfer learning  
- Advanced augmentation  
- Grad-CAM for explainability  

### 🔹 Milestone 4: Deployment
- Model evaluation  
- Streamlit web app  
- Real-time predictions  
- Documentation  

---

## 🧠 System Architecture

### 1. Classification Module (DenseNet)
- Multi-label classification  
- Outputs probability scores  

### 2. Detection Module (YOLOv8)
- Detects abnormal regions  
- Generates bounding boxes  

### 3. Preprocessing Module
- Image conversion  
- Normalization  
- Augmentation  

### 4. Explainability Module (Grad-CAM)
- Generates heatmaps  
- Highlights important regions  

### 5. Visualization Layer
- Combines bounding boxes and heatmaps  

### 6. User Interface (Streamlit)
- Upload X-ray images  
- View predictions in real time  

---

## 📈 Evaluation & Optimization

### Detection (YOLOv8)
- Metric: mAP  

### Classification (DenseNet)
- AUC  
- F1-score  
- Precision & Recall  

### Techniques Used
- Hyperparameter tuning  
- Grad-CAM visualization  
- Real-time testing via Streamlit  

---

## ⚙️ Tech Stack
- Python  
- PyTorch  
- YOLOv8  
- DenseNet  
- NumPy, pandas  
- OpenCV, Pillow  
- matplotlib, seaborn  
- Grad-CAM  
- Streamlit  

---

## ⭐ Key Highlights
- End-to-end AI pipeline  
- Combines detection + classification + visualization  
- Interpretable AI system  
- Ready for real-world applications  
