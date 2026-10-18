"""Fine-tune the baseline waste-detection model (Phase 6 experiment).

Starts from best_model.pt (the reference project's 20-epoch YOLOv8n weights) and keeps
training with a cosine learning-rate schedule. Optionally repeats training images that
contain rare classes (Glass, Waste) to reduce class imbalance.

Run a CONTROL (no oversampling) and the EXPERIMENT (with oversampling) with the same
epochs and seed, so you can tell which change helped:

    python train.py --dataset C:\\data\\waste_detection --name control
    python train.py --dataset C:\\data\\waste_detection --name oversample --oversample Glass Waste --factor 3

Training needs a GPU to be practical (free Google Colab T4 works: about 1 minute/epoch).
Then evaluate baseline and new weights with evaluate.py on the SAME split and compare:

    python evaluate.py --dataset ... --weights best_model.pt --split val --name baseline --out eval_results
    python evaluate.py --dataset ... --weights train_runs/oversample/weights/best.pt --split val --name oversample --out eval_results
    python evaluate.py --compare eval_results/baseline_val.json eval_results/oversample_val.json

Note: Ultralytics picks best.pt using the validation split, so val scores of the new model
are slightly optimistic. The test split is untouched by training, but it is tiny (45 images).
"""
import argparse
from pathlib import Path

import yaml

IMG_EXT = {".jpg", ".jpeg", ".png"}


def build_balanced_list(dataset, names, rare, factor, out_file):
    """Write a train list where images containing a rare class appear `factor` times."""
    rare_ids = {names.index(c) for c in rare}
    img_dir = Path(dataset) / "train" / "images"
    lab_dir = Path(dataset) / "train" / "labels"
    lines, n_rare = [], 0
    for img in sorted(p for p in img_dir.glob("*") if p.suffix.lower() in IMG_EXT):
        lab = lab_dir / (img.stem + ".txt")
        ids = set()
        if lab.is_file():
            ids = {int(ln.split()[0]) for ln in lab.read_text().splitlines() if ln.strip()}
        repeat = factor if ids & rare_ids else 1
        n_rare += repeat > 1
        lines += [str(img)] * repeat
    out_file.write_text("\n".join(lines) + "\n")
    return len(lines), n_rare


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--weights", default="best_model.pt", help="starting point (the baseline)")
    ap.add_argument("--name", required=True, help="run name, e.g. control or oversample")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None, help="0 for first GPU, cpu for CPU")
    ap.add_argument("--oversample", nargs="*", default=[], help="class names to repeat, e.g. Glass Waste")
    ap.add_argument("--factor", type=int, default=3, help="how many times to repeat those images")
    ap.add_argument("--project", default="train_runs")
    args = ap.parse_args()

    dataset = Path(args.dataset).resolve()
    if not Path(args.weights).is_file():
        raise SystemExit(f"Weights not found: {args.weights}")
    if not (dataset / "data.yaml").is_file():
        raise SystemExit(f"data.yaml not found in {dataset}")
    names = yaml.safe_load((dataset / "data.yaml").read_text())["names"]
    names = list(names.values()) if isinstance(names, dict) else list(names)
    bad = [c for c in args.oversample if c not in names]
    if bad:
        raise SystemExit(f"Unknown class(es) {bad}. Valid: {names}")

    # Labels must already be YOLO boxes: run evaluate.py --convert-labels once first.
    sample = next((dataset / "train" / "labels").glob("*.txt"), None)
    if sample and any(len(ln.split()) > 5 for ln in sample.read_text().splitlines()):
        raise SystemExit("Labels are still polygons. Run evaluate.py with --convert-labels once first.")

    project = Path(args.project).resolve()
    project.mkdir(parents=True, exist_ok=True)
    train_ref = "train/images"
    if args.oversample:
        lst = project / f"{args.name}_train_list.txt"
        total, n_rare = build_balanced_list(dataset, names, args.oversample, args.factor, lst)
        train_ref = str(lst)
        print(f"Oversampling {args.oversample} x{args.factor}: {n_rare} images repeated, train list now {total} entries")

    cfg = {"path": str(dataset), "train": train_ref, "val": "valid/images", "test": "test/images", "names": names}
    data_yaml = project / f"{args.name}_data.yaml"
    data_yaml.write_text(yaml.safe_dump(cfg))

    from ultralytics import YOLO

    model = YOLO(args.weights)
    model.train(
        data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        seed=args.seed, deterministic=True, device=args.device,
        cos_lr=True, lr0=0.002, optimizer="AdamW",  # gentler than a fresh run: we are fine-tuning
        project=str(project), name=args.name, exist_ok=True,
    )
    print(f"\nDone. Best weights: {project / args.name / 'weights' / 'best.pt'}")
    print("Next: evaluate these weights AND best_model.pt with evaluate.py on the same split.")


if __name__ == "__main__":
    main()
