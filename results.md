# Evaluation Summary

## YOLO26n License-Plate Detector

Dataset split used for detector evaluation:

- Train: 1,156 images
- Validation: 330 images
- Test: 164 images
- One class: `indian_licence_plate`

Held-out test metrics:

| Metric | Score |
|---|---:|
| Precision | 0.9752 |
| Recall | 0.9572 |
| mAP@50 | 0.9892 |
| mAP@50-95 | 0.8031 |

The detector was trained on an NVIDIA Tesla T4 using Ultralytics YOLO26n. Training stopped early after 86 epochs, with the best checkpoint selected from validation performance.

## EasyOCR Baseline

The detector/OCR pipeline was evaluated on 164 held-out detector-test images:

| Outcome | Count |
|---|---:|
| OCR output | 159 |
| No OCR | 1 |
| No detection | 4 |
| Read errors | 0 |

OCR output rate: **97.0%**.

Mean OCR confidence: **0.418**.

Median OCR confidence: **0.365**.

These confidence values are **not equivalent to OCR accuracy** because the detector dataset did not provide ground-truth plate strings.

## Ground-Truth LPR Experiment

A separate XML-annotated dataset was converted into 1,693 plate crops with ground-truth strings.

Identity-clean split:

| Split | Crops | Unique plate identities |
|---|---:|---:|
| Train | 1,322 | 784 |
| Validation | 171 | 98 |
| Test | 200 | 98 |

There was zero overlap of `plate_text` identities between any two splits.

### CNN-BiLSTM-CTC baseline

The first dedicated recognizer achieved:

- Validation exact accuracy: 0.0%
- Test exact accuracy: 0.0%
- Test character accuracy: 18.18%

A 20-sample overfit diagnostic reached 95% exact accuracy, demonstrating that the CTC training implementation could memorize a small set and that the full-data failure was a generalization problem rather than an inability to optimize at all.

This recognizer is therefore **not presented as a production OCR model**.

## Conclusion

The strongest validated component is the YOLO26n license-plate detector. EasyOCR provides a working recognition baseline but requires further work for reliable character-level accuracy on unseen Indian plates. The repository deliberately reports this limitation instead of presenting OCR confidence as recognition accuracy.
