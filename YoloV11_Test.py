from ultralytics import YOLO
from pathlib import Path

SPECIES = "Falci"
TOKEN = "CMHOG" 

def evaluate_model():
    run_dir     = Path(f"{SPECIES}_{TOKEN}")
    model_path  = "Results" / run_dir  / "weights" / "best.pt"
    yaml_path   = Path("YoloConfig") / f"{SPECIES}" / f"{TOKEN}.yaml"
    output_proj = str(run_dir / "Test")  

    model = YOLO(str(model_path))
    results = model.val(
        data=str(yaml_path),
        split="test",
        project=output_proj,
        name="Testing",
        plots=True,
    )

if __name__ == '__main__':
    evaluate_model()
