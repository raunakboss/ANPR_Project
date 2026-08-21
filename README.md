# 🚘 Automatic Number Plate Recognition (ANPR)

A computer-vision based Automatic Number Plate Recognition system for Indian vehicle license plates.

The project uses a **YOLO26n detector** to localize license plates and an OCR pipeline to read the detected plate text. The training and evaluation workflow was developed in Google Colab and validated on a Tesla T4 GPU.

## ✨ Features

- License-plate detection with YOLO26n
- Indian license-plate focused dataset
- OCR using EasyOCR with multiple preprocessing variants
- Plate crop quality analysis
- Confidence-based OCR candidate selection
- Streamlit interface for image-based ANPR
- Separate LPR research dataset with identity-clean train/validation/test splits
- Reproducible evaluation metrics

## 🧠 System Architecture

```text
Input Image
    ↓
YOLO26n License Plate Detector
    ↓
Detected Plate Bounding Box
    ↓
Crop + Quality Analysis
    ↓
OCR Preprocessing Variants
    ├── Original
    ├── Sharpened
    ├── CLAHE
    ├── Adaptive Threshold
    └── OTSU
    ↓
EasyOCR
    ↓
Candidate Ranking / Confidence
    ↓
Detected Registration Number
```

## 📊 YOLO26 Detector Results

The detector was trained on an Indian license-plate dataset with:

| Split | Images |
|---|---:|
| Train | 1,156 |
| Validation | 330 |
| Test | 164 |
| Total | 1,650 |

Final held-out test performance:

| Metric | Score |
|---|---:|
| Precision | **97.52%** |
| Recall | **95.72%** |
| mAP@50 | **98.92%** |
| mAP@50-95 | **80.31%** |

Training used a **Tesla T4 (14.56 GB VRAM)** and YOLO26n with early stopping.

## 🔎 OCR Baseline

The EasyOCR benchmark was evaluated on all **164 test images**:

- OCR output on **159/164** images
- No OCR on **1/164** images
- No plate detection on **4/164** images
- OCR output rate: **97.0%**
- Mean OCR confidence: **41.8%**
- Median OCR confidence: **36.5%**

OCR confidence is treated as a diagnostic signal rather than ground-truth accuracy because the original detector dataset does not contain plate-text labels.

## 🧪 Dedicated LPR Dataset

A second dataset containing ground-truth plate text was constructed from XML annotations:

- 1,697 XML annotations discovered
- 1,693 valid plate crops
- 980 unique plate identities
- 1,322 train crops / 784 identities
- 171 validation crops / 98 identities
- 200 test crops / 98 identities
- **0 identity overlap** between train, validation, and test

A CNN-BiLSTM-CTC recognizer was investigated on this dataset. The first experimental model achieved 0% exact test accuracy and was retained as an experimental baseline rather than being presented as a production OCR model.

## 🛠️ Tech Stack

- Python
- PyTorch
- Ultralytics YOLO26
- OpenCV
- EasyOCR
- NumPy
- Pandas
- Streamlit
- Google Colab / NVIDIA Tesla T4

## 📁 Recommended Repository Structure

```text
ANPR_Project/
├── streamlit_app.py
├── requirements.txt
├── weights/
│   └── best.pt                 # trained YOLO weights (local, not committed)
├── notebooks/
│   └── train_yolo26_colab.ipynb
├── results/
│   ├── detection_metrics.md
│   └── sample_outputs/
└── README.md
```

## 🚀 Run Locally

```bash
git clone https://github.com/raunakboss/ANPR_Project.git
cd ANPR_Project
python -m venv .venv
```

Activate the environment and install dependencies:

```bash
pip install -r requirements.txt
```

Place the trained detector at:

```text
weights/best.pt
```

Then start Streamlit:

```bash
streamlit run streamlit_app.py
```

## ⚠️ Model Weights

Large model files are intentionally not committed to Git. Download/copy the trained `best.pt` file into `weights/best.pt` before running the application.

## 📌 Project Status

**Detection:** production-ready project baseline validated on a held-out test set.

**OCR:** functional EasyOCR baseline with multi-variant preprocessing; recognition accuracy still requires a stronger character-recognition model trained on sufficiently diverse Indian plate-text data.

## 👨‍💻 Author

**Raunak Saxena**

Computer Science student | Computer Vision | Machine Learning | Python
