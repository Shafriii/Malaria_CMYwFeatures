from ultralytics import YOLO
from pathlib import Path
import shutil
import os

def main():
    # --- CONFIG --- # change to yolov11s.pt / m / l / x as you like
    data_yaml = "YoloConfig/CMLBP.yaml"
    run_name  = "Falci_CMLBP"
    epochs    = 100
    imgsz     = 640
    batch     = 16
    seed      = 0

    # --- INIT & AUTO-DOWNLOAD ---
    cwd = Path.cwd()
    model = YOLO("yolo11n.pt")

    # --- TRAIN ---
    train_results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name=run_name,
        project=str(cwd),   # save run under current folder
        verbose=True,
        plots=True,
        pretrained=True,
        seed=seed,
    )

    # Path to best weights inside the run directory
    best_weights = Path(train_results.save_dir) / "weights" / "best.pt"

    # Copy best weights to CWD with a friendly name
    out_name = f"{run_name}_best.pt"
    out_path = cwd / out_name
    if best_weights.exists():
        shutil.copy2(best_weights, out_path)
        print(f"[INFO] Best model copied to: {out_path.resolve()}")
    else:
        print("[WARN] best.pt not found in run directory.")

    # --- VALIDATE (using best weights) ---
    best_model = YOLO(str(best_weights)) if best_weights.exists() else model
    val_results = best_model.val(data=data_yaml, imgsz=imgsz, seed=seed)
    print("=== Validation Metrics ===")
    print(val_results)

if __name__ == "__main__":
    main()