#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hardcoded JSON -> Ultralytics YOLO converter with robust path resolution.
- Reads ./training.json and ./test.json (same folder as this script)
- Creates YOLO dataset at ./F_New
- Splits training into train/val (VAL_RATIO=0.10), NO augmentation
- Keeps only: Gametocyte, Trophozoite, Schizont, Ring
"""

import os
import json
import shutil
from collections import Counter, defaultdict

from PIL import Image
from sklearn.model_selection import train_test_split
import yaml

# =========================
# HARD-CODED SETTINGS
# =========================
HERE         = os.path.abspath(os.path.dirname(__file__))  # folder of this script
TRAIN_JSON   = os.path.join(HERE, "training.json")
TEST_JSON    = os.path.join(HERE, "test.json")
OUTPUT_DIR   = os.path.join(HERE, "V_Ori")
VAL_RATIO    = 0.10
SEED         = 42

TARGET_CLASSES = ["Gametocyte", "Trophozoite", "Schizont", "Ring"]
SOURCE_TO_TARGET = {
    "gametocyte": "Gametocyte",
    "trophozoite": "Trophozoite",
    "schizont": "Schizont",
    "ring": "Ring",
    # unmapped categories (e.g., 'red blood cell', 'leukocyte', 'difficult') are ignored
}
NAME_TO_ID = {n: i for i, n in enumerate(TARGET_CLASSES)}
ID_TO_NAME = {i: n for n, i in NAME_TO_ID.items()}


# =========================
# IO helpers
# =========================
def ensure_dirs(root):
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(root, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(root, split, "labels"), exist_ok=True)


def write_dataset_yaml(root):
    data = {
        "path": os.path.abspath(root),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(TARGET_CLASSES),
        "names": TARGET_CLASSES,
    }
    with open(os.path.join(root, "dataset.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def load_json_items(pth):
    with open(pth, "r", encoding="utf-8") as f:
        return json.load(f)


# --- Robust resolver: try multiple bases + fallback recursive search
def resolve_image_path(raw_path: str):
    """
    Resolve JSON 'pathname' to an existing file.
    Tries:
      1) If absolute & exists -> use it
      2) HERE / <raw or stripped>
      3) HERE / images / <basename>
      4) parent(HERE) / <raw or stripped>
      5) parent(HERE) / images / <basename>
      6) Recursive basename search under HERE (first match)
    Returns absolute path or None.
    """
    if not raw_path:
        return None

    # Normalize separators, strip possible leading slash so joins behave
    raw = raw_path.replace("\\", "/")
    stripped = raw[1:] if raw.startswith("/") else raw
    base_name = os.path.basename(stripped)

    # 1) Absolute path as-is
    if os.path.isabs(raw) and os.path.exists(raw):
        return os.path.abspath(raw)

    # 2) HERE / stripped
    cand = os.path.join(HERE, stripped)
    if os.path.exists(cand):
        return os.path.abspath(cand)

    # 3) HERE / images / basename
    cand = os.path.join(HERE, "images", base_name)
    if os.path.exists(cand):
        return os.path.abspath(cand)

    # 4) parent(HERE) / stripped
    parent = os.path.dirname(HERE)
    cand = os.path.join(parent, stripped)
    if os.path.exists(cand):
        return os.path.abspath(cand)

    # 5) parent(HERE) / images / basename
    cand = os.path.join(parent, "images", base_name)
    if os.path.exists(cand):
        return os.path.abspath(cand)

    # 6) Recursive search by basename under HERE (first hit wins)
    for root, _, files in os.walk(HERE):
        if base_name in files:
            return os.path.abspath(os.path.join(root, base_name))

    return None


def normalize_box_xyxy_wh(xmin, ymin, xmax, ymax, W, H):
    xmin = max(0, min(float(xmin), W - 1))
    xmax = max(0, min(float(xmax), W - 1))
    ymin = max(0, min(float(ymin), H - 1))
    ymax = max(0, min(float(ymax), H - 1))
    if xmax <= xmin or ymax <= ymin:
        return None
    cx = (xmin + xmax) / 2.0 / W
    cy = (ymin + ymax) / 2.0 / H
    w  = (xmax - xmin) / W
    h  = (ymax - ymin) / H
    if w <= 0 or h <= 0:
        return None
    return cx, cy, w, h


def build_image_records(items):
    """
    Returns:
      records: dict[path] -> dict(boxes=[(xmin,ymin,xmax,ymax),...], labels=[str,...], size=(W,H))
      primary: dict[path] -> str or None (most frequent kept class)
      stats:   diagnostics dict
    """
    records, primary = {}, {}
    missing_paths = []
    kept_images = 0
    kept_boxes = 0
    ignored_boxes = 0

    for entry in items:
        imeta = entry.get("image", {})
        objs  = entry.get("objects", [])
        H     = int(imeta.get("shape", {}).get("r", 0))
        W     = int(imeta.get("shape", {}).get("c", 0))
        path  = resolve_image_path(imeta.get("pathname", ""))

        if not path or not os.path.exists(path):
            missing_paths.append(imeta.get("pathname", ""))
            continue

        boxes, labels = [], []
        for obj in objs:
            cat = str(obj.get("category", "")).strip().lower()
            tgt = SOURCE_TO_TARGET.get(cat, None)
            bb  = obj.get("bounding_box", {})
            if tgt is None:
                ignored_boxes += 1
                continue
            mn = bb.get("minimum", {})
            mx = bb.get("maximum", {})
            ymin, xmin = float(mn.get("r", 0)), float(mn.get("c", 0))
            ymax, xmax = float(mx.get("r", 0)), float(mx.get("c", 0))
            boxes.append((xmin, ymin, xmax, ymax))
            labels.append(tgt)
            kept_boxes += 1

        records[path] = {"boxes": boxes, "labels": labels, "size": (W, H)}
        kept_images += 1
        primary[path] = max(Counter(labels).items(), key=lambda x: x[1])[0] if labels else None

    stats = {
        "kept_images": kept_images,
        "kept_boxes": kept_boxes,
        "ignored_boxes": ignored_boxes,
        "missing_count": len(missing_paths),
        "missing_samples": missing_paths[:10],  # preview first 10 missing
    }
    return records, primary, stats


def safe_stratified_split(img_paths, prim_labels_dict, val_ratio, seed):
    if not img_paths:
        return [], []
    X, y = [], []
    for p in img_paths:
        pl = prim_labels_dict.get(p, None)
        X.append(p)
        y.append(pl if pl is not None else "__none__")
    try:
        tr, vl = train_test_split(X, test_size=val_ratio, random_state=seed, stratify=y)
        return tr, vl
    except Exception:
        tr, vl = train_test_split(X, test_size=val_ratio, random_state=seed, shuffle=True)
        return tr, vl


def copy_and_write_label(dst_root, split, img_path, rec):
    img_name = os.path.basename(img_path)
    dst_img  = os.path.join(dst_root, split, "images", img_name)
    if not os.path.exists(dst_img):
        shutil.copy2(img_path, dst_img)

    label_name = os.path.splitext(img_name)[0] + ".txt"
    dst_lab    = os.path.join(dst_root, split, "labels", label_name)
    W, H       = rec["size"]
    lines = []
    for (xmin, ymin, xmax, ymax), cls_name in zip(rec["boxes"], rec["labels"]):
        yolo = normalize_box_xyxy_wh(xmin, ymin, xmax, ymax, W, H)
        if yolo is None:
            continue
        cls_id = NAME_TO_ID[cls_name]
        cx, cy, w, h = yolo
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    # If you want to include negative images, uncomment the next two lines:
    # if not lines:
    #     open(dst_lab, "w", encoding="utf-8").close(); return True

    if not lines:
        return False

    with open(dst_lab, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return True


def summarize_split(dst_root, split):
    lab_dir = os.path.join(dst_root, split, "labels")
    counts = Counter()
    files = [f for f in os.listdir(lab_dir) if f.endswith(".txt")]
    for ftxt in files:
        with open(os.path.join(lab_dir, ftxt), "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    counts[ID_TO_NAME[cls_id]] += 1
    return counts, len(files)


# =========================
# Main (no args)
# =========================
def main():
    # Prepare output structure
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ensure_dirs(OUTPUT_DIR)

    # Load JSONs
    train_items = load_json_items(TRAIN_JSON)
    test_items  = load_json_items(TEST_JSON)

    # Build records (+ diagnostics)
    train_rec, train_primary, train_stats = build_image_records(train_items)
    test_rec,  _,              test_stats  = build_image_records(test_items)

    # Diagnostics to help catch path issues
    print("=== DIAGNOSTICS ===")
    print(f"TRAIN: kept_images={train_stats['kept_images']}, kept_boxes={train_stats['kept_boxes']}, "
          f"ignored_boxes={train_stats['ignored_boxes']}, missing_images={train_stats['missing_count']}")
    if train_stats['missing_count'] > 0:
        print("  e.g. missing (first 10):")
        for s in train_stats['missing_samples']:
            print("   -", s)
    print(f"TEST : kept_images={test_stats['kept_images']}, kept_boxes={test_stats['kept_boxes']}, "
          f"ignored_boxes={test_stats['ignored_boxes']}, missing_images={test_stats['missing_count']}")
    if test_stats['missing_count'] > 0:
        print("  e.g. missing (first 10):")
        for s in test_stats['missing_samples']:
            print("   -", s)

    train_imgs = list(train_rec.keys())
    test_imgs  = list(test_rec.keys())

    if len(train_imgs) == 0:
        raise SystemExit(
            "ERROR: No training images found on disk after resolving JSON paths.\n"
            "Fix path resolution (see DIAGNOSTICS above). If your images live in a different folder,\n"
            "move this script into the dataset root or adjust `resolve_image_path()` search hints."
        )

    # Split train/val (no aug)
    tr_imgs, vl_imgs = safe_stratified_split(train_imgs, train_primary, VAL_RATIO, SEED)
    if len(tr_imgs) == 0 or len(vl_imgs) == 0:
        # As a safety net, if the split still collapses, force a 90/10 plain split
        n = len(train_imgs)
        k = max(1, int(round(n * (1 - VAL_RATIO))))
        tr_imgs, vl_imgs = train_imgs[:k], train_imgs[k:]
        print(f"[WARN] Stratified split failed to produce non-empty splits. "
              f"Falling back to plain split: train={len(tr_imgs)}, val={len(vl_imgs)}")

    # Write dataset.yaml
    write_dataset_yaml(OUTPUT_DIR)

    # Copy + write labels (no augmentation anywhere)
    kept_tr = kept_vl = kept_te = 0
    for p in tr_imgs:
        kept_tr += int(copy_and_write_label(OUTPUT_DIR, "train", p, train_rec[p]))
    for p in vl_imgs:
        kept_vl += int(copy_and_write_label(OUTPUT_DIR, "val", p, train_rec[p]))
    for p in test_imgs:
        kept_te += int(copy_and_write_label(OUTPUT_DIR, "test", p, test_rec[p]))

    # Summaries
    tr_counts, tr_files = summarize_split(OUTPUT_DIR, "train")
    vl_counts, vl_files = summarize_split(OUTPUT_DIR, "val")
    te_counts, te_files = summarize_split(OUTPUT_DIR, "test")

    print("\n=== SUMMARY ===")
    print(f"Train images with labels: {tr_files} / {len(tr_imgs)}")
    for k in TARGET_CLASSES:
        print(f"  {k}: {tr_counts[k]} boxes")
    print(f"Val images with labels:   {vl_files} / {len(vl_imgs)}")
    for k in TARGET_CLASSES:
        print(f"  {k}: {vl_counts[k]} boxes")
    print(f"Test images with labels:  {te_files} / {len(test_imgs)}")
    for k in TARGET_CLASSES:
        print(f"  {k}: {te_counts[k]} boxes")
    print("\nDataset root:", os.path.abspath(OUTPUT_DIR))
    print("Done.")


if __name__ == "__main__":
    main()
