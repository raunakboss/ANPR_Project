# 🚘 Automatic Number Plate Recognition (ANPR)

A computer-vision application for detecting Indian vehicle license plates and recovering registration text from vehicle images.

The project combines a custom-trained **YOLO26n** detector with a multi-variant **EasyOCR** pipeline, including dedicated handling for two-line plates and Indian registration-number structure.

## ✨ Features

- License-plate detection with YOLO26n
- Multi-variant OCR preprocessing
- Two-line Indian plate handling
- Removal/ignoring of the left-side `IND` logo area where possible
- OCR candidate generation and ranking
- Structure-aware correction for common OCR character confusions
- Annotated detection output
- Streamlit web interface
- Downloadable annotated result

## 🧠 System Architecture

```text
Vehicle Image
     │
     ▼
YOLO26n Plate Detection
     │
     ▼
Plate ROI Extraction
     │
     ▼
Logo / Region Handling
     │
     ▼
Image Preprocessing
     │
     ├── Original
     ├── Sharpened
     ├── CLAHE
     ├── Adaptive Threshold
     └── Otsu
     │
     ▼
EasyOCR Candidates
     │
     ▼
Two-line Plate Reconstruction
     │
     ▼
Indian-format Candidate Ranking
     │
     ▼
Final Registration Number
```

## 📊 YOLO26n Detector Results

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
| mAP@50–95 | **80.31%** |

Training used a **Tesla T4** GPU with YOLO26n and early stopping. The best checkpoint was selected from validation performance and evaluated on the held-out test set.

## 🔎 OCR Pipeline

The application uses EasyOCR with multiple preprocessing variants and candidate ranking. The pipeline additionally handles two-line plates and common OCR character confusions using Indian registration-number structure.

A representative final demo correctly recognized the two-line plate **PB10GN4497** with:

- Detector confidence: **83.2%**
- OCR confidence: **88.7%**
- 8 OCR candidates evaluated
- Best preprocessing: `two_line_combined_plate_normalized`

> OCR confidence is a diagnostic signal, not a direct measure of character-level accuracy. OCR performance varies with plate size, blur, perspective, lighting, occlusion, and image quality.

## 🧪 Dedicated LPR Dataset Experiment

A separate ground-truth plate-text dataset was constructed from XML annotations:

- 1,697 XML annotations discovered
- 1,693 valid plate crops
- 980 unique plate identities
- 1,322 train crops / 784 identities
- 171 validation crops / 98 identities
- 200 test crops / 98 identities
- **0 identity overlap** between train, validation, and test

A CNN-BiLSTM-CTC recognizer was investigated on this dataset. The initial experimental model reached 0% exact test accuracy and was retained as an experimental result rather than being presented as the production OCR component.

## 🖥️ Streamlit Demo

The application provides:

- Premium dashboard-style interface
- Drag-and-drop image upload
- One-click ANPR analysis
- Annotated detection image
- Recognized registration number
- Detector and OCR confidence
- Best preprocessing method
- Expandable OCR candidate inspection
- Downloadable annotated result
- Clear no-detection feedback

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

## 📁 Repository Structure

```text
ANPR_Project/
├── streamlit_app.py
├── train_yolo26.py
├── requirements.txt
├── results.md
├── README.md
└── weights/
    └── best.pt                 # local model weights
```

## 🚀 Run Locally

```bash
git clone https://github.com/raunakboss/ANPR_Project.git
cd ANPR_Project
python -m venv .venv
```

Activate the environment and install dependencies:

```bash
python -m pip install -r requirements.txt
```

Place the trained detector at:

```text
weights/best.pt
```

Then start the application:

```bash
streamlit run streamlit_app.py
```

## ⚠️ Model Weights

Large model files are intentionally kept out of the repository when appropriate. Place the trained `best.pt` checkpoint at `weights/best.pt` locally before running the application.

## 📌 Project Status

**Detection:** validated on a held-out test set with strong localization performance.

**OCR:** functional EasyOCR pipeline with multi-variant preprocessing, two-line reconstruction, and Indian-format candidate ranking. Character recognition remains sensitive to image quality and plate appearance.

## 👨‍💻 Author

**Raunak Saxena**

Computer Science student | Computer Vision | Machine Learning | Python

GitHub: https://github.com/raunakboss
