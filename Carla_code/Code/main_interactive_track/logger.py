"""Run logging and visual annotation."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

import numpy as np

from .models import DecisionPlan


class RunLogger:
    def __init__(self, root_dir: str):
        self.run_dir = os.path.abspath(os.path.join(root_dir, f"recover_{time.strftime('%Y%m%d_%H%M%S')}"))
        self.frames_dir = os.path.join(self.run_dir, "frames")
        os.makedirs(self.frames_dir, exist_ok=True)
        self.jsonl_path = os.path.join(self.run_dir, "decision.jsonl")
        self._fp = open(self.jsonl_path, "a", encoding="utf-8")
        print(f"  日志: {self.jsonl_path}")
        print(f"  图像: {self.frames_dir}")

    def write(self, row: Dict[str, Any]) -> None:
        self._fp.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._fp.flush()

    def close(self) -> None:
        self._fp.close()

    def save_frame(self, image: np.ndarray, name: str, meta: Dict[str, Any], bbox=None, image_size=None) -> str:
        import cv2

        path = os.path.join(self.frames_dir, name)
        cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        sidecar = os.path.splitext(path)[0] + ".json"
        with open(sidecar, "w", encoding="utf-8") as fp:
            json.dump(meta, fp, ensure_ascii=False, indent=2)
        if bbox is not None and image_size is not None:
            x1, y1, x2, y2 = [float(x) for x in bbox]
            w, h = float(image_size[0]), float(image_size[1])
            if x2 > x1 and y2 > y1 and w > 0 and h > 0:
                yolo = os.path.splitext(path)[0] + ".txt"
                with open(yolo, "w", encoding="utf-8") as fp:
                    fp.write(f"0 {((x1+x2)/2)/w:.6f} {((y1+y2)/2)/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}\n")
        return path


def annotate_frame(image: np.ndarray, plan: DecisionPlan, bbox_meta: Optional[Dict[str, Any]], title: str) -> np.ndarray:
    import cv2

    out = image.copy()
    bbox_meta = bbox_meta or {}
    lines = [
        title,
        f"phase={plan.phase} view={plan.view} found={plan.target_found} conf={plan.confidence:.2f}",
        f"wp=({plan.waypoint[0]:.1f},{plan.waypoint[1]:.1f},{plan.waypoint[2]:.1f}) src={plan.source}",
        str(plan.reason)[:96],
    ]
    if bbox_meta.get("bbox"):
        x1, y1, x2, y2 = bbox_meta["bbox"]
        color = (40, 220, 40) if plan.target_found else (255, 190, 30)
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    cv2.rectangle(out, (0, 0), (out.shape[1], min(out.shape[0], 96)), (0, 0, 0), -1)
    for idx, line in enumerate(lines):
        cv2.putText(out, line, (10, 22 + idx * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out
