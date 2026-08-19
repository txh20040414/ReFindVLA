#!/usr/bin/env python3
"""Export decision runs into a YOLO-style dataset with per-image JSON metadata.

Output structure:
  <output>/
    train/
      images/*.png
      labels/*.txt
      labels/*.json
    val/
      images/*.png
      labels/*.txt
      labels/*.json
    annotations.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from PIL import Image
except ImportError:
    Image = None


@dataclass
class Sample:
    image_path: Path
    metadata: Dict[str, Any]
    bbox: Optional[List[float]]
    image_size: Tuple[int, int]


def _find_latest_image(run_dir: Path, record: Dict[str, Any]) -> Optional[Path]:
    frame_name = record.get("decision_image_path") or record.get("frame_path") or record.get("image_path")
    if frame_name:
        candidate = Path(frame_name)
        if not candidate.is_absolute():
            candidate = run_dir / "frames" / candidate
        if candidate.exists():
            return candidate
    return None


def _load_sidecar(image_path: Path) -> Optional[Dict[str, Any]]:
    sidecar = image_path.with_suffix(".json")
    if not sidecar.exists():
        return None
    with sidecar.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_image_size(image_path: Path) -> Tuple[int, int]:
    if Image is None:
        return (0, 0)
    with Image.open(image_path) as img:
        return img.size


def _extract_bbox(metadata: Dict[str, Any]) -> Optional[List[float]]:
    ann = metadata.get("annotation") or metadata.get("bbox_meta") or {}
    if ann.get("visible") and ann.get("bbox"):
        return [float(v) for v in ann["bbox"]]
    return None


def _load_samples(input_dir: Path) -> List[Sample]:
    samples: List[Sample] = []
    run_dirs = sorted(p for p in input_dir.glob("recover_*") if p.is_dir())
    if (input_dir / "decision.jsonl").exists():
        run_dirs.insert(0, input_dir)
    for run_dir in run_dirs:
        jsonl = run_dir / "decision.jsonl"
        if not jsonl.exists():
            continue
        with jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                if record.get("type") not in {"decision", "control_frame"}:
                    continue
                image_path = _find_latest_image(run_dir, record)
                if image_path is None:
                    continue
                sidecar = _load_sidecar(image_path)
                metadata = sidecar if sidecar is not None else record
                bbox = _extract_bbox(metadata)
                annotation = metadata.get("annotation") or metadata.get("bbox_meta") or {}
                image_size = tuple(annotation.get("image_size") or _load_image_size(image_path))
                if len(image_size) != 2:
                    image_size = _load_image_size(image_path)
                samples.append(
                    Sample(
                        image_path=image_path,
                        metadata=metadata,
                        bbox=bbox,
                        image_size=(int(image_size[0]), int(image_size[1])),
                    )
                )
    return samples


def _yolo_line(bbox: Optional[List[float]], image_size: Tuple[int, int]) -> str:
    if bbox is None:
        return ""
    x1, y1, x2, y2 = bbox
    width, height = image_size
    if width <= 0 or height <= 0 or x2 <= x1 or y2 <= y1:
        return ""
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"


def _write_sample(sample: Sample, out_root: Path, split: str, sample_idx: int) -> Dict[str, Any]:
    images_dir = out_root / split / "images"
    labels_dir = out_root / split / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{sample.metadata.get('type', 'sample')}_{sample_idx:06d}"
    image_dst = images_dir / f"{stem}.png"
    label_txt = labels_dir / f"{stem}.txt"
    label_json = labels_dir / f"{stem}.json"

    shutil.copy2(sample.image_path, image_dst)
    label_txt.write_text(_yolo_line(sample.bbox, sample.image_size), encoding="utf-8")
    label_json.write_text(json.dumps(sample.metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    export_record = {
        "split": split,
        "image": str(image_dst),
        "label_txt": str(label_txt),
        "label_json": str(label_json),
        "source_image": str(sample.image_path),
        "bbox": sample.bbox,
        "image_size": sample.image_size,
        "metadata": sample.metadata,
    }
    return export_record


def main() -> None:
    parser = argparse.ArgumentParser(description="Export decision_runs into YOLO-style dataset folders")
    parser.add_argument("--input", required=True, type=str, help="Path to Data/decision_runs")
    parser.add_argument("--output", required=True, type=str, help="Path to export dataset")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Train split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = _load_samples(input_dir)
    if not samples:
        raise SystemExit(f"No samples found in {input_dir}")

    random.seed(args.seed)
    random.shuffle(samples)
    split_idx = int(len(samples) * args.train_ratio)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    manifest: List[Dict[str, Any]] = []
    for split, split_samples in (("train", train_samples), ("val", val_samples)):
        for idx, sample in enumerate(split_samples):
            manifest.append(_write_sample(sample, output_dir, split, idx))

    (output_dir / "annotations.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in manifest) + "\n",
        encoding="utf-8",
    )

    summary = {
        "total": len(samples),
        "train": len(train_samples),
        "val": len(val_samples),
        "output": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
