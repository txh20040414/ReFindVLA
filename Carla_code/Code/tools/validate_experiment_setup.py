#!/usr/bin/env python3
"""Preflight checks for the Windows CARLA-Air / remote Linux topology."""

from __future__ import annotations

import argparse
import importlib
import json
import socket
import urllib.request
from typing import Any, Dict


def _tcp(host: str, port: int, timeout: float) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Check local CARLA/AirSim and remote Qwen connectivity")
    parser.add_argument("--server-url", required=True, help="Remote model server URL, e.g. http://10.0.0.2:8000")
    parser.add_argument("--carla-host", default="127.0.0.1")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--airsim-host", default="127.0.0.1")
    parser.add_argument("--airsim-port", type=int, default=41451)
    parser.add_argument("--timeout", type=float, default=3.0)
    args = parser.parse_args()

    result: Dict[str, Any] = {
        "carla_tcp": _tcp(args.carla_host, args.carla_port, args.timeout),
        "airsim_tcp": _tcp(args.airsim_host, args.airsim_port, args.timeout),
        "server_health": None,
    }
    try:
        with urllib.request.urlopen(args.server_url.rstrip("/") + "/health", timeout=args.timeout) as response:
            result["server_health"] = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        result["server_health"] = {"error": str(exc)}

    for module_name in ("numpy", "requests", "cv2"):
        try:
            importlib.import_module(module_name)
            result[f"import_{module_name}"] = True
        except Exception as exc:
            result[f"import_{module_name}"] = str(exc)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    failed = not result["carla_tcp"] or not result["airsim_tcp"] or not isinstance(result["server_health"], dict) or result["server_health"].get("status") != "ok"
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
