import cv2
import numpy as np
import streamlit as st
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import easyocr

st.set_page_config(page_title="ANPR | Indian License Plates", page_icon="🚘", layout="wide")
MODEL_PATH = Path("weights/best.pt")

st.title("🚘 Automatic Number Plate Recognition")
st.caption("YOLO26n license-plate detection + EasyOCR recognition")

@st.cache_resource
def load_models():
    detector = YOLO(str(MODEL_PATH)) if MODEL_PATH.exists() else None
    reader = easyocr.Reader(["en"], gpu=False)
    return detector, reader


def preprocess_variants(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    up = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    sharpened = cv2.filter2D(up, -1, np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(up)
    adaptive = cv2.adaptiveThreshold(up, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11)
    _, otsu = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return {"original": up, "sharpened": sharpened, "clahe": clahe, "adaptive": adaptive, "otsu": otsu}


def read_plate(reader, crop):
    candidates = []
    for variant, image in preprocess_variants(crop).items():
        for _, text, confidence in reader.readtext(
            image, detail=1, paragraph=False,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        ):
            cleaned = "".join(ch for ch in text.upper() if ch.isalnum())
            if cleaned:
                candidates.append({"text": cleaned, "confidence": float(confidence), "variant": variant})
    return sorted(candidates, key=lambda x: x["confidence"], reverse=True)


def detect(image, detector, reader):
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    result = detector.predict(frame, conf=0.25, verbose=False)[0]
    annotated = frame.copy()
    outputs = []

    if result.boxes is None:
        return annotated, outputs

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        detector_conf = float(box.conf[0])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        candidates = read_plate(reader, crop)
        best = candidates[0] if candidates else None
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"plate {detector_conf:.2f}"
        if best:
            label += f" | {best['text']} ({best['confidence']:.2f})"
        cv2.putText(annotated, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        outputs.append({
            "bbox": [x1, y1, x2, y2],
            "detector_confidence": detector_conf,
            "best_ocr": best,
            "ocr_candidates": candidates[:5],
        })

    return annotated, outputs


detector, reader = load_models()
if detector is None:
    st.warning("Place the trained YOLO weights at `weights/best.pt` to enable detection.")

uploaded = st.file_uploader("Upload a vehicle image", type=["jpg", "jpeg", "png", "webp"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Input image", use_container_width=True)

    if detector is not None and st.button("Run ANPR", type="primary"):
        with st.spinner("Detecting plates and running OCR..."):
            annotated, outputs = detect(image, detector, reader)

        st.subheader("Detection result")
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

        if not outputs:
            st.info("No license plate detected.")
        else:
            st.subheader("Recognition results")
            for i, item in enumerate(outputs, 1):
                st.markdown(f"### Plate {i}")
                best = item["best_ocr"]
                if best:
                    st.write(f"**Text:** `{best['text']}`")
                    st.write(f"**OCR confidence:** {best['confidence']:.3f}")
                    st.write(f"**Detector confidence:** {item['detector_confidence']:.3f}")
                else:
                    st.write("OCR could not read this plate.")
                if item["ocr_candidates"]:
                    st.dataframe(item["ocr_candidates"], use_container_width=True)
