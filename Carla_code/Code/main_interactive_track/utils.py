"""Small math and parsing helpers."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple


COLOR_KEYWORDS = {
    "白色": "white",
    "白": "white",
    "黑色": "black",
    "黑": "black",
    "红色": "red",
    "红": "red",
    "蓝色": "blue",
    "蓝": "blue",
    "银色": "silver",
    "银灰色": "silver",
    "银": "silver",
    "黄色": "yellow",
    "黄": "yellow",
    "绿色": "green",
    "绿": "green",
    "灰色": "gray",
    "灰": "gray",
    "橙色": "orange",
    "橙": "orange",
}

VEHICLE_KEYWORDS = {
    "轿车": "sedan",
    "小车": "sedan",
    "车": "sedan",
    "SUV": "suv",
    "suv": "suv",
    "卡车": "truck",
    "货车": "truck",
    "厢式": "truck",
    "公交车": "bus",
    "巴士": "bus",
    "摩托车": "motorcycle",
    "出租车": "taxi",
}


def clamp(value: Any, low: float, high: float) -> float:
    try:
        number = float(value)
    except Exception:
        return low
    if not math.isfinite(number):
        return low
    return max(low, min(high, number))


def distance3(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def distance2(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def normalize_angle_deg(angle: float) -> float:
    while angle > 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle


def heading_to_unit(heading_deg: float) -> Tuple[float, float]:
    rad = math.radians(heading_deg)
    return math.cos(rad), math.sin(rad)


def right_unit(heading_deg: float) -> Tuple[float, float]:
    rad = math.radians(heading_deg)
    return math.sin(rad), -math.cos(rad)


def parse_instruction(instruction: str) -> Dict[str, Optional[str]]:
    parsed: Dict[str, Optional[str]] = {"color": None, "vehicle_type": None, "raw": instruction}
    for cn, en in COLOR_KEYWORDS.items():
        if cn in instruction:
            parsed["color"] = en
            break
    for cn, en in VEHICLE_KEYWORDS.items():
        if cn in instruction:
            parsed["vehicle_type"] = en
            break
    if parsed["vehicle_type"] is None:
        parsed["vehicle_type"] = "sedan"
    return parsed


def phase_name(raw: Any) -> str:
    name = str(raw or "search").strip().lower().replace(" ", "_").replace("-", "_")
    mapping = {
        "searching": "search",
        "inspect_target": "inspect",
        "inspection": "inspect",
        "confirm_target": "confirm",
        "reacquire": "reacquire",
        "re_acquire": "reacquire",
        "re_acquire_target": "reacquire",
        "re-acquire": "reacquire",
        "follow_target": "follow",
        "tracking": "follow",
        "track": "follow",
        "return": "return_home",
        "returnhome": "return_home",
        "return_home": "return_home",
        "top_view": "inspect",
        "side_view": "inspect",
        "approach": "inspect",
        "hold": "confirm",
    }
    name = mapping.get(name, name)
    if name not in {"search", "inspect", "confirm", "reacquire", "follow", "return_home"}:
        return "search"
    return name


def view_name(raw: Any, phase: Optional[str] = None) -> str:
    if raw is None:
        if phase in {"search", "reacquire"}:
            return "top"
        if phase == "inspect":
            return "side"
        return "forward"
    name = str(raw).strip().lower().replace(" ", "_").replace("-", "_")
    mapping = {
        "top_view": "top",
        "topview": "top",
        "overhead": "top",
        "down": "top",
        "side_view": "side",
        "sideview": "side",
        "oblique": "side",
        "front": "forward",
        "follow": "forward",
    }
    name = mapping.get(name, name)
    if name not in {"top", "side", "forward"}:
        return "forward"
    return name
