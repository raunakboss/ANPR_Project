import cv2
import numpy as np
import streamlit as st
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import easyocr

st.set_page_config(
    page_title="ANPR | Indian License Plates",
    page_icon="🚘",
    layout="wide",
)

MODEL_PATH = Path("weights/best.pt")

st.title("🚘 Automatic Number Plate Recognition")
st.caption("YOLO26n license-plate detection + EasyOCR recognition")

with st.sidebar:
    st.header("About")
    st.write(
        "Upload a vehicle image to detect Indian license plates and "
        "generate OCR candidates using multiple image-preprocessing variants."
    )
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
    up = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    sharpened = cv2.filter2D(
        up,
        -1,
        np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
    )
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    ).apply(up)
    adaptive = cv2.adaptiveThreshold(
        up,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )
    _, otsu = cv2.threshold(
        up,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    return {
        "original": up,
        "sharpened": sharpened,
        "clahe": clahe,
        "adaptive": adaptive,
        "otsu": otsu,
    }


def read_plate(reader, crop):
    candidates = []
    allowlist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    for variant, image in preprocess_variants(crop).items():
        results = reader.readtext(
            image,
            detail=1,
            paragraph=False,
            allowlist=allowlist,
            text_threshold=0.35,
            low_text=0.20,
        )

        for _, text, confidence in results:
            cleaned = "".join(
                ch for ch in text.upper() if ch.isalnum()
            )
            if cleaned:
                candidates.append(
                    {
                        "text": cleaned,
                        "confidence": float(confidence),
                        "variant": variant,
                    }
                )

    # Remove duplicate OCR strings while keeping their strongest score.
    unique = {}
    for item in candidates:
        key = item["text"]
        if key not in unique or item["confidence"] > unique[key]["confidence"]:
            unique[key] = item

    return sorted(
        unique.values(),
        key=lambda x: x["confidence"],
        reverse=True,
    )


def detect(image, detector, reader):
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    result = detector.predict(
        frame,
        conf=0.25,
        iou=0.70,
        verbose=False,
    )[0]

    annotated = frame.copy()
    outputs = []

    if result.boxes is None or len(result.boxes) == 0:
        return annotated, outputs

    for box in result.boxes:
        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist(),
        )
        detector_conf = float(box.conf[0])

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        candidates = read_plate(reader, crop)
        best = candidates[0] if candidates else None

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        label = f"plate {detector_conf:.2f}"
        if best:
            label += f" | {best['text']} ({best['confidence']:.2f})"

        cv2.putText(
            annotated,
            label,
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )

        outputs.append(
            {
                "bbox": [x1, y1, x2, y2],
                "detector_confidence": detector_conf,
                "best_ocr": best,
                "ocr_candidates": candidates[:5],
            }
        )

    return annotated, outputs


if not MODEL_PATH.exists():
    st.warning(
        "Trained weights are not included in Git. Place `best.pt` at "
        "`weights/best.pt` before running inference."
    )
    st.code("ANPR_Project/weights/best.pt")
    st.stop()

try:
    detector, reader = load_models(str(MODEL_PATH))
except Exception as exc:
    st.error("Unable to load the ANPR models.")
    st.exception(exc)
    st.stop()

uploaded = st.file_uploader(
    "Upload a vehicle image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    left, right = st.columns(2)
    with left:
        st.image(
            image,
            caption="Input image",
            use_container_width=True,
        )

    with right:
        st.info(
            "Detection uses the trained YOLO26n model. "
            "OCR is evaluated across five preprocessing variants."
        )

    if st.button("🚀 Run ANPR", type="primary", use_container_width=True):
        with st.spinner("Detecting plates and running OCR..."):
            annotated, outputs = detect(
                image,
                detector,
                reader,
            )

        st.subheader("Detection result")
        st.image(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            use_container_width=True,
        )

        if not outputs:
            st.warning("No license plate detected.")
        else:
            st.success(f"Detected {len(outputs)} license plate(s).")

            for i, item in enumerate(outputs, 1):
                st.markdown(f"### Plate {i}")

                best = item["best_ocr"]
                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Detector confidence",
                        f"{item['detector_confidence']:.3f}",
                    )

                with col2:
                    st.metric(
                        "OCR confidence",
                        f"{best['confidence']:.3f}" if best else "—",
                    )

                if best:
                    st.markdown(
                        f"**Recognized text:** `{best['text']}`  "
                        f"\n\n**Best preprocessing:** `{best['variant']}`"
                    )
                else:
                    st.info("OCR could not read this plate.")

                if item["ocr_candidates"]:
                    st.caption("Top OCR candidates")
                    st.dataframe(
                        item["ocr_candidates"],
                        use_container_width=True,
                        hide_index=True,
                    )
