import cv2
import numpy as np
import streamlit as st
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import easyocr
import re

st.set_page_config(
    page_title="ANPR | Indian License Plates",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path("weights/best.pt")

# -----------------------------------------------------------------------------
# Premium UI styling — keeps the app native Streamlit, but gives it a polished
# portfolio/demo feel without requiring external CSS frameworks or assets.
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 85% 0%, rgba(24, 119, 242, .13), transparent 28%),
            radial-gradient(circle at 10% 20%, rgba(0, 212, 255, .08), transparent 25%),
            #0b1017;
    }

    [data-testid="stHeader"] { background: rgba(11,16,23,.82); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d141d 0%, #0a0f16 100%);
        border-right: 1px solid rgba(255,255,255,.07);
    }

    .hero {
        padding: 28px 30px 24px;
        border: 1px solid rgba(90,170,255,.18);
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(20,35,53,.95), rgba(12,19,28,.88));
        box-shadow: 0 18px 55px rgba(0,0,0,.28);
        margin-bottom: 22px;
    }
    .hero-kicker {
        color: #62c8ff;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 1.6px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .hero-title {
        font-size: clamp(30px, 4vw, 48px);
        line-height: 1.05;
        font-weight: 800;
        margin: 0;
        color: #f4f8ff;
    }
    .hero-subtitle {
        color: #9eafc3;
        font-size: 15px;
        margin-top: 12px;
        max-width: 760px;
    }
    .badge {
        display: inline-block;
        padding: 5px 10px;
        margin: 4px 6px 0 0;
        border-radius: 999px;
        background: rgba(73,167,255,.10);
        border: 1px solid rgba(73,167,255,.22);
        color: #9ddcff;
        font-size: 12px;
        font-weight: 650;
    }

    .section-title {
        font-size: 21px;
        font-weight: 750;
        color: #f0f5fb;
        margin: 8px 0 12px;
    }
    .muted { color: #8fa1b5; font-size: 13px; }

    .metric-card {
        padding: 16px 18px;
        min-height: 96px;
        border-radius: 16px;
        background: rgba(18,28,40,.78);
        border: 1px solid rgba(255,255,255,.07);
    }
    .metric-label {
        color: #8fa1b5;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: .8px;
    }
    .metric-value {
        color: #f5f8fc;
        font-size: 27px;
        font-weight: 780;
        margin-top: 4px;
    }
    .metric-accent { color: #61c7ff; }

    .upload-card {
        padding: 20px;
        border-radius: 18px;
        background: rgba(17,27,39,.78);
        border: 1px solid rgba(255,255,255,.07);
        margin-bottom: 16px;
    }

    .result-card {
        padding: 22px;
        border-radius: 20px;
        background: linear-gradient(145deg, rgba(18,30,43,.94), rgba(11,18,27,.94));
        border: 1px solid rgba(93,177,255,.16);
        box-shadow: 0 16px 45px rgba(0,0,0,.20);
        margin: 8px 0 18px;
    }
    .result-label {
        color: #8fa1b5;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .plate-text {
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        font-size: clamp(28px, 4vw, 44px);
        letter-spacing: 3px;
        font-weight: 850;
        color: #ffffff;
        margin: 4px 0 10px;
    }
    .confidence-pill {
        display: inline-block;
        padding: 6px 11px;
        border-radius: 999px;
        background: rgba(0,214,143,.10);
        border: 1px solid rgba(0,214,143,.22);
        color: #65e5b4;
        font-size: 12px;
        font-weight: 700;
    }

    div[data-testid="stFileUploader"] {
        border-radius: 16px;
    }
    div.stButton > button[kind="primary"] {
        border-radius: 12px;
        height: 48px;
        font-weight: 750;
        border: 0;
        background: linear-gradient(90deg, #1687ff, #25b7ff);
        box-shadow: 0 8px 25px rgba(22,135,255,.25);
    }
    div.stButton > button[kind="primary"]:hover {
        filter: brightness(1.08);
        transform: translateY(-1px);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Model helpers
# -----------------------------------------------------------------------------
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
    """Generate OCR-confusion corrections using the broad structure of Indian plates."""
    text = clean_text(text)
    if not (7 <= len(text) <= 12) or not text[:2].isalpha():
        return []

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
            corrected_region = "".join(digit_map.get(ch, ch) for ch in region)
            corrected_number = "".join(digit_map.get(ch, ch) for ch in number)
            if corrected_region.isdigit() and corrected_number.isdigit() and series.isalpha():
                out.append(text[:2] + corrected_region + series + corrected_number)
    return list(dict.fromkeys(out))


def plate_format_score(text):
    """Score whether a candidate has a plausible Indian registration layout."""
    text = clean_text(text)
    if not (7 <= len(text) <= 12) or not text[:2].isalpha():
        return 0.0

    best = 0.0
    for region_len in (1, 2):
        for series_len in (1, 2, 3):
            number_start = 2 + region_len + series_len
            number_len = len(text) - number_start
            if not (1 <= number_len <= 4):
                continue
            region = text[2:2 + region_len]
            series = text[2 + region_len:number_start]
            number = text[number_start:]
            if region.isdigit() and series.isalpha() and number.isdigit():
                score = 1.0
                if 8 <= len(text) <= 10:
                    score += 0.25
                if region_len == 2:
                    score += 0.08
                if number_len == 4:
                    score += 0.08
                best = max(best, score)
    return best


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

        top_results, bottom_results = [], []
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
        format_score = plate_format_score(text)
        normalized_bonus = 0.08 if "normalized" in item["variant"] else 0.0
        short_penalty = 0.35 if len(text) < 6 else 0.0
        return confidence + (0.45 * format_score) + normalized_bonus - short_penalty

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


# -----------------------------------------------------------------------------
# Header / sidebar
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">Computer Vision • OCR • Indian Vehicles</div>
        <div class="hero-title">🚘 Automatic Number Plate Recognition</div>
        <div class="hero-subtitle">
            Detect Indian license plates with a trained YOLO26n model and recover
            registration text with a multi-variant EasyOCR pipeline.
        </div>
        <div>
            <span class="badge">YOLO26n</span>
            <span class="badge">EasyOCR</span>
            <span class="badge">Two-line plates</span>
            <span class="badge">OCR correction</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 🚘 ANPR Studio")
    st.caption("Indian License Plate Recognition")
    st.divider()
    st.markdown("### Model performance")
    st.metric("mAP@50", "98.92%")
    st.metric("mAP@50–95", "80.31%")
    st.caption("Measured on the held-out test set.")
    st.divider()
    st.markdown("### Pipeline")
    st.markdown("**01**  YOLO26n detection  \n**02**  Plate ROI extraction  \n**03**  Image preprocessing  \n**04**  EasyOCR candidates  \n**05**  Indian-format ranking")
    st.divider()
    st.caption("Tip: use a clear, front-facing plate image for the strongest OCR result.")

# -----------------------------------------------------------------------------
# Model availability
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Upload / run area
# -----------------------------------------------------------------------------
st.markdown('<div class="section-title">Analyze a vehicle image</div>', unsafe_allow_html=True)
st.markdown('<div class="muted">Upload JPG, JPEG, PNG or WEBP. The detector finds the plate automatically.</div>', unsafe_allow_html=True)

with st.container(border=True):
    uploaded = st.file_uploader("Vehicle image", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    left, right = st.columns([1.25, .75], gap="large")
    with left:
        st.image(image, caption=f"Input • {uploaded.name}", use_container_width=True)
    with right:
        st.markdown("### Ready to analyze")
        st.markdown("The pipeline will detect the plate, remove the logo area where possible, test multiple preprocessing variants, and rank OCR candidates by Indian plate structure.")
        st.markdown('<span class="confidence-pill">● Model ready</span>', unsafe_allow_html=True)
        st.write("")
        run = st.button("🚀  Run ANPR", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if run:
        with st.spinner("Detecting plate and running OCR variants..."):
            annotated, outputs = detect(image, detector, reader)

        st.markdown('<div class="section-title">Detection result</div>', unsafe_allow_html=True)
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

        if not outputs:
            st.error("No license plate detected. Try a clearer or closer vehicle image.")
        else:
            st.success(f"Detected {len(outputs)} license plate(s).")
            for i, item in enumerate(outputs, 1):
                best = item["best_ocr"]
                st.markdown(f'<div class="section-title">Plate {i}</div>', unsafe_allow_html=True)

                if best:
                    st.markdown(
                        f'''<div class="result-card">
                            <div class="result-label">Recognized registration</div>
                            <div class="plate-text">{best["text"]}</div>
                            <span class="confidence-pill">OCR {best["confidence"]:.1%}</span>
                        </div>''',
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("OCR could not read this plate.")

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Detector confidence", f"{item['detector_confidence']:.3f}")
                with c2:
                    st.metric("OCR confidence", f"{best['confidence']:.3f}" if best else "—")
                with c3:
                    st.metric("Candidates", len(item["ocr_candidates"]))

                if best:
                    st.caption(f"Best preprocessing: `{best['variant']}`")

                with st.expander("🔎 View OCR candidates", expanded=False):
                    if item["ocr_candidates"]:
                        st.dataframe(item["ocr_candidates"], use_container_width=True, hide_index=True)
                    else:
                        st.info("No OCR candidates were generated.")

                encoded = cv2.imencode(".jpg", annotated)[1].tobytes()
                st.download_button(
                    "⬇️ Download annotated result",
                    data=encoded,
                    file_name="anpr_result.jpg",
                    mime="image/jpeg",
                    key=f"download_{i}",
                )
else:
    st.markdown(
        """
        <div style="text-align:center; padding:70px 20px 90px; color:#8fa1b5;">
            <div style="font-size:54px; margin-bottom:12px;">📷</div>
            <div style="font-size:20px; font-weight:700; color:#dce8f5;">Upload a vehicle image to begin</div>
            <div style="font-size:14px; margin-top:7px;">Best results come from a clear, well-lit license plate.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
