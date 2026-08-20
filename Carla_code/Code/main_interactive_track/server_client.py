"""HTTP client for the remote Qwen2.5-VL + LoRA VLA service."""

from __future__ import annotations

import base64
from typing import Any, Dict

import numpy as np
import requests

from .config import RecoverVLAConfig


class RemoteVLAClient:
    def __init__(self, config: RecoverVLAConfig):
        self.config = config

    def check(self) -> bool:
        try:
            resp = requests.get(f"{self.config.server_url}/health", timeout=10)
            resp.raise_for_status()
            info = resp.json()
            if info.get("status") != "ok":
                print(f"  推理服务器状态异常: {info}")
                return False
            print(f"  已连接推理服务器: {info.get('model', 'unknown')}")
            print(f"  设备: {info.get('device', 'unknown')}")
            return True
        except Exception as exc:
            print(f"  无法连接推理服务器: {self.config.server_url}")
            print(f"  错误: {exc}")
            return False

    def predict(self, image_rgb: np.ndarray, user_text: str) -> Dict[str, Any]:
        import cv2

        ok, encoded = cv2.imencode(".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            return {"action": None, "response": "", "latency_ms": 0.0, "error": "image_encode_failed"}
        image_b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
        try:
            resp = requests.post(
                f"{self.config.server_url}/predict",
                json={"image_b64": image_b64, "user_text": user_text},
                timeout=self.config.remote_predict_timeout_s,
            )
            resp.raise_for_status()
            data = resp.json()
            if "latency_ms" not in data:
                data["latency_ms"] = 0.0
            return data
        except requests.exceptions.Timeout:
            return {"action": None, "response": "", "latency_ms": 0.0, "error": "remote_timeout"}
        except Exception as exc:
            return {"action": None, "response": "", "latency_ms": 0.0, "error": str(exc)}
