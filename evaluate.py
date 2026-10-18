"""Evaluate a YOLO waste-detection model on one dataset split.

Example (Windows):
    python evaluate.py --dataset C:\\data\\waste_detection --weights best_model.pt --split val
    python evaluate.py --dataset C:\\data\\waste_detection --weights best_model.pt --split test

The dataset folder must contain train/valid/test sub-folders (each with images/ and
labels/) and the original data.yaml (used only for the class names).

Always evaluate on 'val' or 'test', never 'train': train-set scores are not
generalization performance.

If the label files are still in the original polygon format (more than 5 numbers per
line), run once with --convert-labels. This rewrites the labels as YOLO boxes, the same
conversion the reference notebook does.
"""
import argparse
import json
import os
from pathlib import Path

import yaml

SPLIT_DIRS = {"train": "train", "val": "valid", "test": "test"}
IMG_EXT = {".jpg", ".jpeg", ".png"}
MIN_IMAGES = 100  # below this, warn that results are unreliable
MIN_INSTANCES = 10  # per-class warning threshold


def polygon_to_box_line(line):
    """'cls x1 y1 x2 y2 ...' -> 'cls xc yc w h' (same logic as the notebook)."""
    parts = line.split()
    cls, coords = parts[0], list(map(float, parts[1:]))
    xs, ys = coords[0::2], coords[1::2]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    return f"{cls} {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {x1 - x0:.6f} {y1 - y0:.6f}"


def convert_labels(label_dir):
    """Convert polygon label files in a folder to box format. Returns files changed."""
    changed = 0
    for f in Path(label_dir).glob("*.txt"):
        lines = [ln.strip() for ln in f.read_text().splitlines() if ln.strip()]
        if any(len(ln.split()) > 5 for ln in lines):
            f.write_text("\n".join(polygon_to_box_line(ln) if len(ln.split()) > 5 else ln for ln in lines) + "\n")
            changed += 1
    return changed


def count_split(dataset, split):
    """Return (n_images, per-class instance counts, n_polygon_files) for a split."""
    base = Path(dataset) / SPLIT_DIRS[split]
    images = [p for p in (base / "images").glob("*") if p.suffix.lower() in IMG_EXT]
    inst, polygons = {}, 0
    for f in (base / "labels").glob("*.txt"):
        for ln in f.read_text().splitlines():
            parts = ln.split()
            if not parts:
                continue
            inst[int(parts[0])] = inst.get(int(parts[0]), 0) + 1
            if len(parts) > 5:
                polygons += 1
    return len(images), inst, polygons


def compare(base_path, new_path):
    """Print baseline vs new results. Refuses to compare different splits or sizes."""
    a, b = json.loads(Path(base_path).read_text()), json.loads(Path(new_path).read_text())
    if (a["split"], a["images"], a["instances"]) != (b["split"], b["images"], b["instances"]):
        raise SystemExit("These runs used different splits or data. Re-evaluate both on the same split.")
    print(f"Split '{a['split']}': {a['images']} images, {a['instances']} boxes (identical for both)")
    print(f"{'Metric':<16}{'Baseline':>10}{'New':>10}{'Change':>10}")
    for k in ("precision", "recall", "mAP50", "mAP50-95"):
        print(f"{k:<16}{a[k]:>10.3f}{b[k]:>10.3f}{b[k] - a[k]:>+10.3f}")
    print(f"\n{'Class mAP50':<16}{'Baseline':>10}{'New':>10}{'Change':>10}")
    for c in a["per_class"]:
        x, y = a["per_class"][c]["mAP50"], b["per_class"].get(c, {}).get("mAP50", float("nan"))
        print(f"{c:<16}{x:>10.3f}{y:>10.3f}{y - x:>+10.3f}")
    print("\nSmall differences (about 0.01-0.02 mAP) can be noise; do not claim an improvement from them.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", help="dataset folder (contains train/valid/test and data.yaml)")
    ap.add_argument("--compare", nargs=2, metavar=("BASELINE_JSON", "NEW_JSON"),
                    help="print a side-by-side comparison of two saved result files and exit")
    ap.add_argument("--weights", default="best_model.pt")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001, help="low threshold is standard for mAP")
    ap.add_argument("--out", default="eval_results", help="folder for plots and the JSON summary")
    ap.add_argument("--name", default=None, help="label for this run (default: weights file name)")
    ap.add_argument("--convert-labels", action="store_true", help="convert polygon labels to boxes first")
    args = ap.parse_args()

    if args.compare:
        return compare(*args.compare)
    if not args.dataset:
        ap.error("--dataset is required (unless using --compare)")

    dataset = Path(args.dataset).resolve()
    if not Path(args.weights).is_file():
        raise SystemExit(f"Weights not found: {args.weights}")
    src_yaml = dataset / "data.yaml"
    if not src_yaml.is_file():
        raise SystemExit(f"data.yaml not found in {dataset}")
    names = yaml.safe_load(src_yaml.read_text())["names"]

    if args.convert_labels:
        for s in SPLIT_DIRS.values():
            n = convert_labels(dataset / s / "labels")
            print(f"converted {n} label files in {s}/")

    n_img, inst, polygons = count_split(dataset, args.split)
    if polygons:
        raise SystemExit(f"{polygons} polygon label lines found. Re-run with --convert-labels.")
    if n_img == 0:
        raise SystemExit(f"No images found for split '{args.split}' in {dataset}")

    # data.yaml with absolute paths for this machine (the original has Colab paths)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run_name = args.name or Path(args.weights).stem
    data_cfg = {
        "path": str(dataset),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": names,
    }
    local_yaml = out / "data_local.yaml"
    local_yaml.write_text(yaml.safe_dump(data_cfg))

    from ultralytics import YOLO

    model = YOLO(args.weights)
    res = model.val(
        data=str(local_yaml), split=args.split, imgsz=args.imgsz, conf=args.conf,
        plots=True, project=str(out.resolve()), name=f"{run_name}_{args.split}", exist_ok=True,
    )

    cls_names = list(names.values()) if isinstance(names, dict) else list(names)
    per_class = {}
    for i, c in enumerate(res.box.ap_class_index):
        per_class[cls_names[int(c)]] = {
            "instances": inst.get(int(c), 0),
            "precision": float(res.box.p[i]),
            "recall": float(res.box.r[i]),
            "mAP50": float(res.box.ap50[i]),
            "mAP50-95": float(res.box.ap[i]),
        }
    summary = {
        "weights": args.weights,
        "split": args.split,
        "images": n_img,
        "instances": sum(inst.values()),
        "precision": float(res.box.mp),
        "recall": float(res.box.mr),
        "mAP50": float(res.box.map50),
        "mAP50-95": float(res.box.map),
        "speed_ms_per_image": {k: round(float(v), 2) for k, v in res.speed.items()},
        "per_class": per_class,
        "plots_dir": str(res.save_dir),
    }
    (out / f"{run_name}_{args.split}.json").write_text(json.dumps(summary, indent=2))

    print(f"\n=== {args.weights} on '{args.split}' split: {n_img} images, {summary['instances']} boxes ===")
    print(f"Precision {summary['precision']:.3f} | Recall {summary['recall']:.3f} | "
          f"mAP@0.5 {summary['mAP50']:.3f} | mAP@0.5:0.95 {summary['mAP50-95']:.3f}")
    print(f"Speed (ms/image, this machine): {summary['speed_ms_per_image']}")
    print(f"{'Class':<9}{'Boxes':>6}{'P':>8}{'R':>8}{'mAP50':>8}{'mAP50-95':>10}")
    for c, m in per_class.items():
        print(f"{c:<9}{m['instances']:>6}{m['precision']:>8.3f}{m['recall']:>8.3f}{m['mAP50']:>8.3f}{m['mAP50-95']:>10.3f}")
    print(f"Confusion matrix and curves saved in: {res.save_dir}")

    if n_img < MIN_IMAGES:
        print(f"\nWARNING: only {n_img} images in this split. Treat these numbers as indicative, not reliable.")
    thin = [cls_names[k] for k in range(len(cls_names)) if inst.get(k, 0) < MIN_INSTANCES]
    if thin:
        print(f"WARNING: fewer than {MIN_INSTANCES} boxes for: {', '.join(thin)}. Their scores are not meaningful.")


if __name__ == "__main__":
    main()
