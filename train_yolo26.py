from pathlib import Path
from ultralytics import YOLO

# Export your YOLO-format dataset and point this path to its YAML file.
DATA_YAML = Path("anpr_dataset/data_fixed.yaml")
MODEL = "yolo26n.pt"

if not DATA_YAML.exists():
    raise FileNotFoundError(
        f"Dataset YAML not found: {DATA_YAML}. "
        "Keep datasets outside Git and update DATA_YAML to your local path."
    )

model = YOLO(MODEL)

results = model.train(
    data=str(DATA_YAML),
    epochs=100,
    imgsz=640,
    device=0,
    project="runs/detect",
    name="baseline",
    plots=True,
    patience=20,
)

print("Training complete.")
print("Best weights:", Path(results.save_dir) / "weights" / "best.pt")
