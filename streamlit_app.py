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

with st.sidebar:
    st.header("About")
    st.write("Upload a vehicle image to detect Indian license plates and generate OCR candidates using multiple image-preprocessing variants.")
    st.metric("Detector", "YOLO26n")
    st.metric("Test mAP@50", "98.92%")
    st.metric("Test mAP@50-95", "80.31%")


@st.cache_resource
def load_models(model_path: str):
    detector = YOLO(model_path)
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return detector, reader


def preprocess_variants(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    up = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    sharpened = cv2.filter2D(up, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(up)
    adaptive = cv2.adaptiveThreshold(up, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11)
    _, otsu = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return {"original": up, "sharpened": sharpened, "clahe": clahe, "adaptive": adaptive, "otsu": otsu}


def clean_text(text):
    cleaned = "".join(ch for ch in text.upper() if ch.isalnum())
    if cleaned in {"IND", "IN", "INDIA"}:
        return ""
    return cleaned


def ocr_image(reader, image, variant):
    results = reader.readtext(
        image,
        detail=1,
        paragraph=False,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        text_threshold=0.30,
        low_text=0.15,
        link_threshold=0.20,
        mag_ratio=1.5,
    )
    candidates = []
    for _, text, confidence in results:
        cleaned = clean_text(text)
        if cleaned:
            candidates.append({"text": cleaned, "confidence": float(confidence), "variant": variant})
    return candidates


def indian_plate_normalizations(text):
    """Generate OCR-confusion corrections using the structure of Indian plates.

    Expected structure is broadly: STATE(2 letters) + REGION(1-2 digits) +
    SERIES(1-3 letters) + NUMBER(1-4 digits). OCR often confuses characters
    such as O/0, I/1, T/1 and L/4. Corrections are applied only in positions
    that are expected to be numeric, so normal letters in the series are kept.
    """
    text = clean_text(text)
    if not (7 <= len(text) <= 12):
        return []

    # First two characters should be letters. Do not alter them.
    if not text[:2].isalpha():
        return []

    # Try every possible 1-2 digit region length and 1-3 letter series length.
    digit_map = {
        "O": "0", "Q": "0", "D": "0",
        "I": "1", "J": "1", "T": "1", "L": "4",
        "Z": "2", "S": "5", "G": "6", "B": "8",
    }
    out = []
    for region_len in (1, 2):
        for series_len in (1, 2, 3):
            number_start = 2 + region_len + series_len
            if number_start >= len(text):
                continue
            number_len = len(text) - number_start
            if not (1 <= number_len <= 4):
                continue

            region = text[2:2 + region_len]
            series = text[2 + region_len:number_start]
            number = text[number_start:]

            # Region and final number should be numeric after OCR correction.
            corrected_region = "".join(digit_map.get(ch, ch) for ch in region)
            corrected_number = "".join(digit_map.get(ch, ch) for ch in number)
            if not corrected_region.isdigit() or not corrected_number.isdigit():
                continue
            if not series.isalpha():
                continue

            candidate = text[:2] + corrected_region + series + corrected_number
            out.append(candidate)
    return list(dict.fromkeys(out))


def read_plate(reader, crop):
    """OCR tuned for Indian plates, including two-line layouts."""
    candidates = []
    h, w = crop.shape[:2]
    if h < 2 or w < 10:
        return candidates

    left_margin = int(w * 0.18)
    right_roi = crop[:, max(0, left_margin):]

    for variant, image in preprocess_variants(right_roi).items():
        candidates.extend(ocr_image(reader, image, f"roi_{variant}"))

    for variant, image in preprocess_variants(crop).items():
        candidates.extend(ocr_image(reader, image, f"full_{variant}"))

    if h >= 24:
        gray = cv2.cvtColor(right_roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        split = gray.shape[0] // 2
        overlap = max(2, int(gray.shape[0] * 0.05))
        top = gray[:min(gray.shape[0], split + overlap)]
        bottom = gray[max(0, split - overlap):]

        top_results = []
        bottom_results = []
        row_images = {
            "top_original": top,
            "top_clahe": cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(top),
            "bottom_original": bottom,
            "bottom_clahe": cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(bottom),
        }
        for name, image in row_images.items():
            found = ocr_image(reader, image, f"rows_{name}")
            if name.startswith("top_"):
                top_results.extend(found)
            else:
                bottom_results.extend(found)

        candidates.extend(top_results)
        candidates.extend(bottom_results)
        top_results = sorted(top_results, key=lambda x: x["confidence"], reverse=True)[:3]
        bottom_results = sorted(bottom_results, key=lambda x: x["confidence"], reverse=True)[:3]
        for top_item in top_results:
            for bottom_item in bottom_results:
                combined = clean_text(top_item["text"] + bottom_item["text"])
                if 6 <= len(combined) <= 12:
                    candidates.append({
                        "text": combined,
                        "confidence": float(min(top_item["confidence"], bottom_item["confidence"])),
                        "variant": "two_line_combined",
                    })

    # Add structure-aware corrected versions. This fixes common OCR errors
    # such as PBTOGNLL97 -> PB10GN4497 without hardcoding a particular plate.
    corrected = []
    for item in candidates:
        for normalized in indian_plate_normalizations(item["text"]):
            if normalized != item["text"]:
                corrected.append({
                    "text": normalized,
                    "confidence": min(0.99, item["confidence"] + 0.03),
                    "variant": item["variant"] + "_plate_normalized",
                })
    candidates.extend(corrected)

    unique = {}
    for item in candidates:
        text = item["text"]
        if not text or text in {"IND", "IN", "INDIA"}:
            continue
        if text not in unique or item["confidence"] > unique[text]["confidence"]:
            unique[text] = item

    def ranking_score(item):
        text = item["text"]
        confidence = item["confidence"]
        length_bonus = 0.10 if 8 <= len(text) <= 10 else 0.0
        short_penalty = 0.35 if len(text) < 6 else 0.0
        structure_bonus = 0.12 if indian_plate_normalizations(text) == [] and 8 <= len(text) <= 10 else 0.0
        return confidence + length_bonus + structure_bonus - short_penalty

    return sorted(unique.values(), key=ranking_score, reverse=True)


def detect(image, detector, reader):
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    result = detector.predict(frame, conf=0.25, iou=0.70, verbose=False)[0]
    annotated = frame.copy()
    outputs = []
    if result.boxes is None or len(result.boxes) == 0:
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
        cv2.putText(annotated, label, (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        outputs.append({
            "bbox": [x1, y1, x2, y2],
            "detector_confidence": detector_conf,
            "best_ocr": best,
            "ocr_candidates": candidates[:8],
        })
    return annotated, outputs


if not MODEL_PATH.exists():
    st.warning("Trained weights are not included in Git. Place `best.pt` at `weights/best.pt` before running inference.")
    st.code("ANPR_Project/weights/best.pt")
    st.stop()

try:
    detector, reader = load_models(str(MODEL_PATH))
except Exception as exc:
    st.error("Unable to load the ANPR models.")
    st.exception(exc)
    st.stop()

uploaded = st.file_uploader("Upload a vehicle image", type=["jpg", "jpeg", "png", "webp"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    left, right = st.columns(2)
    with left:
        st.image(image, caption="Input image", use_container_width=True)
    with right:
        st.info("Detection uses the trained YOLO26n model. OCR evaluates the plate ROI, ignores the IND logo where possible, supports two-line layouts, and applies structure-aware OCR correction.")

    if st.button("🚀 Run ANPR", type="primary", use_container_width=True):
        with st.spinner("Detecting plates and running OCR..."):
            annotated, outputs = detect(image, detector, reader)
        st.subheader("Detection result")
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
        if not outputs:
            st.warning("No license plate detected.")
        else:
            st.success(f"Detected {len(outputs)} license plate(s).")
            for i, item in enumerate(outputs, 1):
                st.markdown(f"### Plate {i}")
                best = item["best_ocr"]
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Detector confidence", f"{item['detector_confidence']:.3f}")
                with col2:
                    st.metric("OCR confidence", f"{best['confidence']:.3f}" if best else "—")
                if best:
                    st.markdown(f"**Recognized text:** `{best['text']}`  \n\n**Best preprocessing:** `{best['variant']}`")
                else:
                    st.info("OCR could not read this plate.")
                if item["ocr_candidates"]:
                    st.caption("Top OCR candidates")
                    st.dataframe(item["ocr_candidates"], use_container_width=True, hide_index=True)
