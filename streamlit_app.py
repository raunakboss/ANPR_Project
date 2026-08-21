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
    up = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

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


def clean_text(text):
    cleaned = "".join(
        ch for ch in text.upper() if ch.isalnum()
    )
    # The Indian flag / country marking is not part of the registration number.
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
            candidates.append(
                {
                    "text": cleaned,
                    "confidence": float(confidence),
                    "variant": variant,
                }
            )
    return candidates


def read_plate(reader, crop):
    """OCR tuned for Indian plates, including two-line layouts.

    The detector crop can contain the IND logo on the left. We therefore
    OCR both the complete crop and a right-side ROI that excludes the logo.
    For tall/two-line plates, the upper and lower text rows are also OCRed
    and combined into a single candidate.
    """
    candidates = []

    h, w = crop.shape[:2]
    if h < 2 or w < 10:
        return candidates

    # Remove the left-side IND/logo region from OCR while retaining a small
    # margin so the first registration character is not clipped.
    left_margin = int(w * 0.18)
    right_roi = crop[:, max(0, left_margin):]

    roi_variants = preprocess_variants(right_roi)

    for variant, image in roi_variants.items():
        candidates.extend(
            ocr_image(
                reader,
                image,
                f"roi_{variant}",
            )
        )

    # Also OCR the full crop. The cleaner candidates from the right ROI will
    # generally outrank logo-related text, while this helps one-line plates.
    for variant, image in preprocess_variants(crop).items():
        candidates.extend(
            ocr_image(
                reader,
                image,
                f"full_{variant}",
            )
        )

    # Two-line Indian plates: OCR top and bottom rows separately and combine.
    # This is especially useful for plates such as PB10G / N4497.
    if h >= 24:
        gray = cv2.cvtColor(right_roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(
            gray,
            None,
            fx=4,
            fy=4,
            interpolation=cv2.INTER_CUBIC,
        )

        split = gray.shape[0] // 2
        overlap = max(2, int(gray.shape[0] * 0.05))
        top = gray[:min(gray.shape[0], split + overlap)]
        bottom = gray[max(0, split - overlap):]

        top_results = []
        bottom_results = []

        for name, image in {
            "top_original": top,
            "top_clahe": cv2.createCLAHE(
                clipLimit=2.0,
                tileGridSize=(8, 8),
            ).apply(top),
            "bottom_original": bottom,
            "bottom_clahe": cv2.createCLAHE(
                clipLimit=2.0,
                tileGridSize=(8, 8),
            ).apply(bottom),
        }.items():
            found = ocr_image(
                reader,
                image,
                f"rows_{name}",
            )
            if name.startswith("top_"):
                top_results.extend(found)
            else:
                bottom_results.extend(found)

        candidates.extend(top_results)
        candidates.extend(bottom_results)

        # Combine the strongest top/bottom strings. This handles two-line
        # plates where EasyOCR sees each row independently.
        top_results = sorted(
            top_results,
            key=lambda x: x["confidence"],
            reverse=True,
        )[:3]
        bottom_results = sorted(
            bottom_results,
            key=lambda x: x["confidence"],
            reverse=True,
        )[:3]

        for top_item in top_results:
            for bottom_item in bottom_results:
                combined = clean_text(
                    top_item["text"] + bottom_item["text"]
                )
                if combined and 6 <= len(combined) <= 12:
                    candidates.append(
                        {
                            "text": combined,
                            "confidence": float(
                                min(
                                    top_item["confidence"],
                                    bottom_item["confidence"],
                                )
                            ),
                            "variant": "two_line_combined",
                        }
                    )

    # Remove duplicate OCR strings while keeping their strongest score.
    unique = {}
    for item in candidates:
        text = item["text"]
        if not text or text in {"IND", "IN", "INDIA"}:
            continue
        if text not in unique or item["confidence"] > unique[text]["confidence"]:
            unique[text] = item

    candidates = list(unique.values())

    # Prefer plausible registration-number lengths when confidence scores are
    # close. Very short strings such as IND/logo fragments should not win.
    def ranking_score(item):
        text = item["text"]
        confidence = item["confidence"]
        length_bonus = 0.10 if 8 <= len(text) <= 10 else 0.0
        short_penalty = 0.35 if len(text) < 6 else 0.0
        return confidence + length_bonus - short_penalty

    return sorted(
        candidates,
        key=ranking_score,
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
                "ocr_candidates": candidates[:8],
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
            "OCR evaluates the plate ROI, ignores the IND logo where possible, "
            "and supports two-line plate layouts."
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
