# ============================================================
# CliniScan — Lung Abnormality Detection
# Streamlit Web App — Milestone 4
# ============================================================
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
import timm
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
import tempfile
import os

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="CliniScan — Lung Abnormality Detection",
    page_icon="🫁",
    layout="wide"
)

CLASSES = [
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Consolidation", "ILD", "Infiltration", "Lung Opacity", "Nodule/Mass",
    "Other lesion", "Pleural effusion", "Pleural thickening", "Pneumothorax",
    "Pulmonary fibrosis"
]
NUM_CLASSES = len(CLASSES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Model paths ───────────────────────────────────────────────
CLS_MODEL_PATH = "efficientnet_best_multi_label.pth"
DET_MODEL_PATH = "best-final(0.1map).pt"

# ── Load models (cached) ──────────────────────────────────────
@st.cache_resource
def load_cls_model():
    model = timm.create_model("efficientnet_b2", pretrained=False,
                               num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(CLS_MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

@st.cache_resource
def load_det_model():
    return YOLO(DET_MODEL_PATH)

# ── GradCAM ───────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, 'activations', o.detach()))
        target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, 'gradients', go[0].detach()))

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        output[0, class_idx].backward()
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1))
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, torch.sigmoid(output).detach().cpu().numpy()[0]

def overlay_heatmap(pil_img, cam, alpha=0.45):
    img_np = np.array(pil_img.resize((320, 320)))
    cam_r  = cv2.resize(cam, (320, 320))
    heatmap = cv2.applyColorMap((cam_r * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return (alpha * heatmap + (1 - alpha) * img_np).astype(np.uint8)

cls_transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

COLORS = [
    (231,76,60),(52,152,219),(46,204,113),(243,156,18),(155,89,182),
    (26,188,156),(230,126,34),(52,73,94),(233,30,99),(0,188,212),
    (139,195,74),(255,87,34),(96,125,139),(121,85,72)
]

# ── UI ────────────────────────────────────────────────────────
st.title("🫁 CliniScan — Lung Abnormality Detection")
st.markdown("**AI-powered chest X-ray analysis using Deep Learning**")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/lungs.png", width=80)
    st.header("About CliniScan")
    st.markdown("""
    **CliniScan** detects 14 lung abnormalities from chest X-rays using:
    - 🔬 EfficientNet-B2 Classifier (AUC: 0.95)
    - 🎯 YOLOv8 Detector 
    - 🔥 GradCAM Explainability
    
    **Detectable Conditions:**
    """)
    for cls in CLASSES:
        st.markdown(f"• {cls}")
    st.markdown("---")
    

# Main area
uploaded = st.file_uploader(
    "Upload a Chest X-ray (PNG or JPG)",
    type=["png", "jpg", "jpeg"]
)

conf_threshold = st.slider(
    "Detection Confidence Threshold", 
    min_value=0.01, max_value=0.5, 
    value=0.15, step=0.01
)

if uploaded is not None:
    pil_img = Image.open(uploaded).convert("RGB")

    st.markdown("---")
    st.subheader("📊 Analysis Results")

    # Load models
    with st.spinner("Loading models..."):
        cls_model = load_cls_model()
        det_model = load_det_model()
        gradcam   = GradCAM(cls_model, cls_model.blocks[-1][-1].conv_pwl)

    # ── Classification ────────────────────────────────────────
    input_tensor = cls_transform(pil_img).unsqueeze(0).to(DEVICE)
    input_tensor.requires_grad_()

    with torch.enable_grad():
        output = cls_model(input_tensor)
        probs  = torch.sigmoid(output).detach().cpu().numpy()[0]

    pred_classes = [(CLASSES[i], float(probs[i]))
                    for i in range(NUM_CLASSES) if probs[i] > 0.5]
    top_class_idx = int(np.argmax(probs))

    # ── GradCAM ───────────────────────────────────────────────
    input_tensor2 = cls_transform(pil_img).unsqueeze(0).to(DEVICE)
    input_tensor2.requires_grad_()
    with torch.enable_grad():
        cam, _ = gradcam.generate(input_tensor2, top_class_idx)
    overlay = overlay_heatmap(pil_img, cam)

    # ── Detection ─────────────────────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        pil_img.save(tmp.name)
        tmp_path = tmp.name

    det_results = det_model.predict(
        source=tmp_path, conf=conf_threshold,
        iou=0.3, save=False, verbose=False)
    os.unlink(tmp_path)

    det_img = np.array(pil_img.resize((640, 640)))
    orig_w, orig_h = pil_img.size
    detections = []

    for box in det_results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x1 = int(x1 * 640 / orig_w)
        y1 = int(y1 * 640 / orig_h)
        x2 = int(x2 * 640 / orig_w)
        y2 = int(y2 * 640 / orig_h)
        cid  = int(box.cls)
        conf = float(box.conf)
        color = COLORS[cid]
        cv2.rectangle(det_img, (x1,y1), (x2,y2), color, 2)
        cv2.putText(det_img, f"{CLASSES[cid]} {conf:.2f}",
                    (x1, max(y1-5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        detections.append((CLASSES[cid], conf))

    # ── Display results ───────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("**Original X-ray**")
        st.image(pil_img.resize((320, 320)), use_container_width="True")

    with col2:
        st.markdown("**GradCAM Heatmap**")
        cam_display = cv2.resize(cam, (320, 320))
        cam_color   = cv2.applyColorMap(
            (cam_display * 255).astype(np.uint8), cv2.COLORMAP_JET)
        st.image(cv2.cvtColor(cam_color, cv2.COLOR_BGR2RGB),
                 use_container_width="True")

    with col3:
        st.markdown("**GradCAM Overlay**")
        st.image(overlay, use_container_width="True")

    with col4:
        st.markdown("**YOLO Detection**")
        st.image(det_img, use_container_width="True")

    # ── Classification results ────────────────────────────────
    st.markdown("---")
    st.subheader("🔬 Classification Results (EfficientNet-B2)")

    if pred_classes:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Detected Conditions:**")
            for cls_name, prob in sorted(pred_classes,
                                          key=lambda x: x[1], reverse=True):
                st.progress(prob, text=f"{cls_name}: {prob:.1%}")
        with col_b:
            st.markdown("**Top Prediction:**")
            st.success(f"🎯 {CLASSES[top_class_idx]}")
            st.markdown(f"Confidence: **{probs[top_class_idx]:.1%}**")
            st.markdown(f"GradCAM focus: **{CLASSES[top_class_idx]}** region")
    else:
        st.success("✅ No abnormalities detected above threshold")

    # ── Detection results ─────────────────────────────────────
    st.markdown("---")
    st.subheader("🎯 Detection Results (YOLOv8)")

    if detections:
        st.markdown(f"**{len(detections)} region(s) detected:**")
        for cls_name, conf in sorted(detections,
                                      key=lambda x: x[1], reverse=True):
            st.markdown(f"• **{cls_name}** — confidence: {conf:.1%}")
    else:
        st.info("No regions detected above confidence threshold. "
                "Try lowering the slider.")

    # ── Model info ────────────────────────────────────────────
    st.markdown("---")
    with st.expander("ℹ️ Model Performance Summary"):
        col_x, col_y, col_z = st.columns(3)
        col_x.metric("Classifier AUC", "0.9506", "+0.0026 vs baseline")
        col_y.metric("Classifier F1",  "0.5475", "+0.0284 vs baseline")
        col_z.metric("Detection mAP50","0.1136", "+20% vs baseline")

        st.markdown("""
        **Dataset:** VinDr-CXR (15,000 chest X-rays, 14 abnormality classes)  
        **Classifier:** EfficientNet-B2 fine-tuned with Albumentations  
        **Detector:** YOLOv8m trained on 15,000 images at 640px  
        **Explainability:** GradCAM on last conv block  
        """)

st.markdown("---")



