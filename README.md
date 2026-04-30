# 🔭 AstroDetect — Astronomical Object Detection & Classification

<div align="center">

![AstroDetect Banner](https://github.com/user-attachments/assets/760e1c00-527d-4da1-8455-e0b6ec9a9a42)

[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Space-blue?style=for-the-badge)](https://huggingface.co/spaces/aniketkhandare/ASTRODETECT)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**A hybrid three-stage AI pipeline that detects and classifies celestial objects in telescope images.**

*ResNet-50 · YOLOv11s · photutils · FastAPI · Gradio*

</div>

---

## 📊 Results at a Glance

| Model | Task | Score | Benchmark |
|-------|------|-------|-----------|
| ResNet-50 | Point-source classification | **88.15% accuracy** | SDSS DR17 |
| YOLOv11s | Extended-object detection | **72.2% mAP@50** | COSMICA test set |
| vs. SOTA | COSMICA benchmark | **+10.3 pp improvement** | Prior best: 61.87% |

---

## 🌌 What It Detects

| Class | Detected By | Description |
|-------|------------|-------------|
| ⭐ **STAR** | photutils + ResNet-50 | Point-like stellar sources |
| 🌌 **GALAXY** | photutils + ResNet-50 | Extended galactic systems |
| ⚡ **QSO** | photutils + ResNet-50 | Quasi-stellar objects (quasars) |
| 🌫️ **Nebula** | YOLOv11s | Diffuse gas and dust clouds |
| 🔵 **Globular Cluster** | YOLOv11s | Dense spherical star concentrations |
| ☄️ **Comet** | YOLOv11s | Icy bodies with tails |

---

## 🏗️ Architecture

The system uses a **3-stage hybrid pipeline** designed around the physics of telescope imagery:

```
INPUT IMAGE
     │
     ├─── Stage I ──────────────────────────────────────────────────────┐
     │    photutils DAOStarFinder                                        │
     │    · σ-clipped background estimation                             │
     │    · Gaussian PSF matched-filter (FWHM = 3.0 px)                │
     │    · Detection threshold τ = η·σ_bg  (η = 4.0)                  │
     │    · Flux-weighted centroid estimation                           │
     │    → Extracts all point sources                                  │
     │                                                                   │
     ├─── Stage II ─────────────────────────────────────────────────────┤
     │    ResNet-50 Classifier                                           │
     │    · 32×32 px crops centred on each source                       │
     │    · Batch inference (single forward pass for all crops)         │
     │    · Trained on SDSS DR17 spectrophotometric data                │
     │    → Labels each source: STAR / GALAXY / QSO                    │
     │                                                                   │
     ├─── Stage III ────────────────────────────────────────────────────┤
     │    YOLOv11s Object Detector                                       │
     │    · Processes full image at 640×640 resolution                  │
     │    · Multi-scale feature pyramid (strides 8/16/32)               │
     │    · CIoU + DFL loss · NMS IoU=0.45                              │
     │    · Trained on COSMICA dataset (19,487 images)                  │
     │    → Detects: nebula / globular_cluster / comet / galaxy         │
     │                                                                   │
     └─── MERGE ────────────────────────────────────────────────────────┘
          Combine all detections → Annotated output image
```

> **Why hybrid?** Stars are 3–8 pixel PSF profiles — architecturally undetectable by YOLO (minimum grid cell = 8px). DAOStarFinder handles point sources; YOLOv11s handles extended objects. Neither can do both.

---

## 📁 Project Structure

```
CelestialDetector/
├── backend/
│   ├── main.py                 ← FastAPI REST API (3-stage pipeline)
│   └── requirements.txt
├── frontend/
│   ├── app.py                  ← Streamlit interactive UI
│   └── requirements.txt
├── model/
│   ├── resnet50_classifier.pt  ← ResNet-50 weights (98.6 MB)
│   ├── yolo11s_cosmica.pt      ← YOLOv11s weights  (19.2 MB)
│   ├── cosmica_classes.txt     ← YOLO class names
│   └── classes.txt
├── notebooks/
│   ├── COSMICA_Training.ipynb  ← YOLOv11s training (Kaggle P100)
│   ├── ResNet_Training.ipynb   ← ResNet-50 training (Colab T4)
│   └── Evaluation.ipynb        ← Confusion matrix + plots
├── huggingface/
│   └── app.py                  ← Gradio deployment
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/celestial-object-detection.git
cd celestial-object-detection
```

### 2. Install dependencies

```bash
# Backend
pip install -r backend/requirements.txt

# Frontend
pip install -r frontend/requirements.txt
```

### 3. Download model weights

Download from Google Drive and place in `model/`:

| File | Size | Source |
|------|------|--------|
| `resnet50_classifier.pt` | 98.6 MB | Google Drive / CelestialV2 |
| `yolo11s_cosmica.pt` | 19.2 MB | Kaggle output tab |
| `cosmica_classes.txt` | < 1 KB | Kaggle output tab |

### 4. Run the backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

### 5. Run the frontend

```bash
cd frontend
streamlit run app.py
```

Open `http://localhost:8501` and upload any telescope image.

---

## 🌐 Live Demo

Try it on HuggingFace Spaces — no installation needed:

**[🔭 huggingface.co/spaces/aniketkhandare/ASTRODETECT](https://huggingface.co/spaces/aniketkhandare/ASTRODETECT)**

---

## 📦 Datasets

### SDSS DR17 — Classifier Training

| Property | Value |
|----------|-------|
| Source | Sloan Digital Sky Survey Data Release 17 |
| URL | https://skyserver.sdss.org/dr17/ |
| Paper | Abdurrouf et al. (2022), ApJS 259(2), 35 |
| DOI | https://doi.org/10.3847/1538-4365/ac4414 |
| Images | 9,000 (3,000 per class) |
| Size | 64 × 64 px JPEG cutouts |
| Classes | GALAXY · STAR · QSO |
| Split | 85% train / 15% validation |

### COSMICA — Detector Training

| Property | Value |
|----------|-------|
| Source | astronomy.ru amateur astrophotography community |
| Kaggle | https://www.kaggle.com/datasets/piratinskii/astronomical-object-detection |
| Images | 19,487 annotated telescope images |
| Classes | comet · galaxy · globular_cluster · nebula |
| Split | train 15,589 / val 1,950 / test 1,948 |
| Annotations | YOLO format bounding boxes |

---

## 🧪 Training

### ResNet-50 Classifier

```
Platform  : Google Colab (NVIDIA T4 GPU)
Epochs    : 40  (best at epoch 21)
Batch     : 64
Optimizer : AdamW  lr=0.0003  wd=1e-4
Scheduler : OneCycleLR
Frozen    : ResNet-50 Layers 1–2 (ImageNet features preserved)
Trainable : Layers 3–4 + custom head
```

### YOLOv11s Detector

```
Platform  : Kaggle (Tesla P100-PCIE-16GB)
Epochs    : 50  (early stop patience=20)
Batch     : 16
Optimizer : AdamW  lr=0.001  wd=0.0005
Augments  : flipud=0.5  fliplr=0.5  degrees=180
            hsv_v=0.4   scale=0.5   mosaic=1.0
```

---

## 📈 Detailed Results

### ResNet-50 — SDSS Validation Set

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| GALAXY | 0.94 | 0.98 | **0.96** | 450 |
| STAR | 0.86 | 0.86 | **0.86** | 450 |
| QSO | 0.84 | 0.81 | **0.83** | 450 |
| **Overall** | 0.88 | 0.88 | **0.88** | 1,350 |
| **Accuracy** | — | — | **88.15%** | — |

### YOLOv11s — COSMICA Test Set

| Class | mAP@50 | mAP@50-95 | Precision | Recall |
|-------|--------|-----------|-----------|--------|
| globular_cluster | **0.930** | 0.584 | 0.887 | 0.880 |
| nebula | **0.770** | 0.493 | 0.769 | 0.709 |
| comet | **0.632** | 0.245 | 0.636 | 0.593 |
| galaxy | **0.556** | 0.282 | 0.756 | 0.465 |
| **All** | **0.722** | **0.401** | 0.762 | 0.662 |

### State-of-the-Art Comparison

| Method | mAP@50 | Params |
|--------|--------|--------|
| EfficientDet-L0 (prior) | ~42.0% | ~4M |
| YOLOv8x (prior) | ~56.0% | ~68M |
| YOLOv9-C (prior) | ~58.0% | ~69M |
| YOLOv11n — COSMICA paper best | 61.87% | 2.6M |
| **YOLOv11s — Ours** | **72.2%** | **9.4M** |

---

## 🔌 API Reference

The FastAPI backend exposes these endpoints:

```
GET  /health          — Check models are loaded
GET  /model/info      — Model sizes and class names
POST /detect          — Upload image → annotated JPEG
POST /detect/json     — Upload image → JSON detections
```

### Example: detect objects in an image

```python
import requests

with open("telescope.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/detect",
        files={"file": f},
        params={"conf": 0.25, "sigma": 4.0, "max_sources": 150}
    )

# Save annotated image
with open("result.jpg", "wb") as f:
    f.write(response.content)

# Get counts from headers
print(response.headers["X-Class-Counts"])
# → {"STAR": 475, "GALAXY": 25, "QSO": 1, "nebula": 1}
```

### Example: get JSON detections

```python
response = requests.post(
    "http://localhost:8000/detect/json",
    files={"file": open("telescope.jpg","rb")},
    params={"conf": 0.25}
)
data = response.json()
print(f"Found {data['total_detected']} objects in {data['inference_ms']}ms")
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Point-source detection | photutils · astropy |
| Classification | PyTorch · ResNet-50 |
| Object detection | Ultralytics · YOLOv11s |
| Backend API | FastAPI · uvicorn |
| Local frontend | Streamlit |
| Web deployment | Gradio · HuggingFace Spaces |
| Training (classifier) | Google Colab T4 |
| Training (detector) | Kaggle P100 |

---

## 📋 Requirements

```
Python >= 3.10
torch >= 2.0.0
torchvision >= 0.15.0
ultralytics >= 8.4.0
photutils >= 1.9.0
astropy >= 5.3.0
fastapi
uvicorn
streamlit
gradio >= 5.0.0
Pillow >= 10.0.0
numpy >= 1.24.0
```

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- **SDSS DR17** — Sloan Digital Sky Survey for the spectrophotometric training data
- **COSMICA dataset** — astronomy.ru community and Kaggle contributor piratinskii
- **DeepSpaceYolo** — Luxembourg Institute of Science and Technology (LIST) for the initial DSO dataset
- **Ultralytics** — for the YOLOv11 architecture and training framework
- **photutils / astropy** — for the PSF photometry tools

---



Made with ❤️ and a lot of telescope images

**[⭐ Star this repo](https://github.com/YOUR_USERNAME/celestial-object-detection)** if you found it useful


