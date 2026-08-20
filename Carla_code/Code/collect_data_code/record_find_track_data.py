#!/usr/bin/env python3
"""Offline recorder for find-and-track datasets.

This script only does local data collection:
- connect to CARLA-Air and AirSim
- spawn / select a target vehicle
- move the drone through search / approach / top-view phases
- save raw frames, red-box previews, JSON sidecars, and YOLO labels

Default output:
    data/find_track_data (relative to the working directory)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


DEFAULT_OUTPUT_ROOT = Path(os.environ.get("REFINDVLA_DATA_ROOT", "data/find_track_data"))
DEFAULT_INSTRUCTION = "重新搜索前面那辆橙色卡车，找到后切换成俯视"


def _bootstrap_code_path() -> None:
    code_dir = Path(__file__).resolve().parents[1]
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))


_bootstrap_code_path()

try:
    import cv2
except ImportError as exc:
    raise SystemExit("opencv-python is required for dataset recording.") from exc

try:
    import airsim
except ImportError as exc:
    raise SystemExit("airsim is required for dataset recording.") from exc

try:
    from main_interactive_track.config import CONFIG
    from main_interactive_track.prediction import forecast_target_state
    from main_interactive_track.simulator import (
        get_drone_state as _get_drone_state,
        get_target_state as _get_target_state,
        project_target_bbox_to_image as _project_target_bbox_to_image,
        spawn_background_traffic as _spawn_background_traffic,
        spawn_target_vehicle as _spawn_target_vehicle,
        airsim_ned_to_carla_location as _shared_airsim_ned_to_carla_location,
        carla_location_to_airsim_ned as _shared_carla_location_to_airsim_ned,
        _add_carla_python_paths,
    )
    from main_interactive_track.utils import parse_instruction
except ImportError as exc:
    raise SystemExit(f"Failed to import runtime helpers: {exc}") from exc


def get_drone_state(airsim_client, _airsim_module=None) -> Dict[str, Any]:
    """Return the new runtime drone state as a serializable dictionary."""
    return _get_drone_state(airsim_client).to_dict()


def get_target_vehicle_state(target_vehicle) -> Dict[str, Any]:
    """Return the new runtime target state as a serializable dictionary."""
    return _get_target_state(target_vehicle, CONFIG).to_dict()


def project_target_bbox_to_image(airsim_client, _airsim_module, target_vehicle) -> Dict[str, Any]:
    """Use the shared CARLA projection adapter for collection annotations."""
    return _project_target_bbox_to_image(airsim_client, target_vehicle, CONFIG)


def spawn_target_vehicle(world, carla_module, *, color=None, vehicle_type="sedan", spawn_location=None):
    """Keep the collector API while delegating spawning to the new runtime adapter."""
    if spawn_location is None:
        return _spawn_target_vehicle(world, carla_module, CONFIG, color, vehicle_type)

    # The runtime adapter intentionally randomizes normal episodes. For data
    # collection, honor an explicitly requested CARLA spawn point.
    blueprint_lib = world.get_blueprint_library()
    blueprint_names = {
        "sedan": ["vehicle.tesla.model3", "vehicle.audi.a2", "vehicle.bmw.grandtourer", "vehicle.mercedes.coupe"],
        "suv": ["vehicle.tesla.cybertruck", "vehicle.jeep.wrangler_rubicon", "vehicle.nissan.patrol"],
        "truck": ["vehicle.carlamotors.firetruck", "vehicle.carlamotors.carlacola"],
        "bus": ["vehicle.mercedes.sprinter"],
    }
    vehicle_bp = None
    for name in blueprint_names.get(vehicle_type, blueprint_names["sedan"]):
        try:
            vehicle_bp = blueprint_lib.find(name)
            break
        except Exception:
            continue
    if vehicle_bp is None:
        vehicles = list(blueprint_lib.filter("vehicle.*"))
        if not vehicles:
            return None
        vehicle_bp = vehicles[0]
    if color and vehicle_bp.has_attribute("color"):
        color_map = {
            "white": "255,255,255", "black": "0,0,0", "red": "255,0,0",
            "blue": "0,0,255", "silver": "192,192,192", "yellow": "255,255,0",
            "green": "0,255,0", "gray": "128,128,128", "orange": "255,165,0",
        }
        vehicle_bp.set_attribute("color", color_map.get(color, "255,255,255"))
    return world.try_spawn_actor(vehicle_bp, spawn_location)


def spawn_background_traffic(world, carla_module, *, traffic_count=8, start_index=10, allow_unsafe=False):
    """Delegate background traffic generation to the shared runtime adapter."""
    config = CONFIG
    # The shared adapter reads these values from the immutable config. Build a
    # temporary config only when the collector explicitly overrides them.
    from dataclasses import replace
    return _spawn_background_traffic(
        world,
        carla_module,
        replace(
            config,
            background_traffic=int(traffic_count),
            traffic_spawn_start_index=int(start_index),
            allow_unsafe_traffic=bool(allow_unsafe),
        ),
    )


def save_frame_sidecar(frame_path: str, metadata: Dict[str, Any], yolo_bbox=None, image_size=None) -> None:
    """Write collection sidecars without depending on the legacy tracker module."""
    sidecar_path = Path(frame_path).with_suffix(".json")
    sidecar_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if yolo_bbox is None or image_size is None:
        return
    x1, y1, x2, y2 = [float(value) for value in yolo_bbox]
    width, height = [float(value) for value in image_size]
    if width <= 0 or height <= 0 or x2 <= x1 or y2 <= y1:
        return
    yolo_path = sidecar_path.with_suffix(".txt")
    yolo_path.write_text(
        f"0 {((x1 + x2) / 2) / width:.6f} {((y1 + y2) / 2) / height:.6f} "
        f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}\n",
        encoding="utf-8",
    )


@dataclass
class FrameRecord:
    frame_index: int
    timestamp: float
    phase: str
    view: str
    instruction: str
    raw_image: str
    preview_image: str
    sidecar_json: str
    target_visible: bool
    target_bbox: Optional[List[int]]
    target_label: str
    drone_state: Dict[str, Any]
    target_state: Dict[str, Any]


def _connect_carla_local(host: str, port: int):
    """Connect to CARLA directly, without the interactive tracking launcher."""
    try:
        _carla_root = os.environ.get(
            "CARLA_AIR_ROOT",
            "",
        )
        if _carla_root:
            _add_carla_python_paths(_carla_root)
        import carla

        client = carla.Client(host, port)
        client.set_timeout(20.0)
        world = client.get_world()
        print(f"  📍 地图: {world.get_map().name}")
        return client, world, carla
    except Exception as e:
        print(f"  ❌ 无法连接 CARLA: {e}")
        return None, None, None


def _connect_airsim_local(host: str, port: int, takeoff_altitude: float):
    """Connect to AirSim and fly to a lower altitude suitable for visual collection."""
    client = airsim.MultirotorClient(ip=host, port=port)
    client.confirmConnection()
    print("  ✅ AirSim 已连接")
    client.enableApiControl(True)
    client.armDisarm(True)
    client.takeoffAsync().join()
    client.moveToZAsync(takeoff_altitude, 3).join()
    if os.environ.get("RECOVER_CAMERA_CONTROL", "0").lower() in {"1", "true", "yes", "on"}:
        _set_collection_view(client, airsim, "forward")
    else:
        print("  相机: 使用 AirSim settings.json 静态配置（未调用动态相机 API）")
    print(f"  🛫 无人机起飞 → {abs(takeoff_altitude):.1f}m")
    print("  ✅ 无人机就绪!")
    return client


def _select_spawn_location(world, spawn_index: int):
    """Pick an explicit CARLA spawn point when the operator wants map-level control."""
    if spawn_index < 0:
        return None
    try:
        spawn_points = list(world.get_map().get_spawn_points())
        if not spawn_points:
            return None
        spawn_index = max(0, min(int(spawn_index), len(spawn_points) - 1))
        sp = spawn_points[spawn_index]
        loc = sp.location
        print(
            f"  🎯 使用目标 spawn_index={spawn_index} "
            f"@ ({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f})"
        )
        return sp
    except Exception as e:
        print(f"  ⚠️ 无法按 spawn_index 选择目标点: {e}")
        return None


def _set_collection_view(airsim_client, airsim_module, view_name: str) -> None:
    """Apply a collection-friendly camera pose, independent from the online tracker defaults."""
    view_name = str(view_name).lower().strip()
    pitch_map = {
        "top": -86.0,
        "down": -78.0,
        "forward": -68.0,
        "side": -58.0,
    }
    fov_map = {
        "top": 72.0,
        "down": 78.0,
        "forward": 88.0,
        "side": 82.0,
    }
    pitch = pitch_map.get(view_name, -70.0)
    fov = fov_map.get(view_name, 85.0)
    try:
        setter = getattr(airsim_client, "simSetCameraOrientation", None)
        if setter is None:
            print("  ⚠️ 当前 AirSim 客户端没有 simSetCameraOrientation；保持静态相机")
            return
        setter("0", airsim_module.to_quaternion(math.radians(pitch), 0.0, 0.0))
        fov_setter = getattr(airsim_client, "simSetCameraFov", None)
        if fov_setter is not None:
            fov_setter("0", fov)
    except Exception as e:
        print(f"  ⚠️ 无法设置采集相机视角: {e}")


def _annotate_standard_preview(image_np: np.ndarray, bbox: Optional[List[int]], target_label: str) -> np.ndarray:
    """Create a clean preview image with a standard bbox only."""
    import cv2

    annotated = image_np.copy()
    if bbox and len(bbox) >= 4:
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.rectangle(annotated, (x1, max(0, y1 - 26)), (min(annotated.shape[1] - 1, x1 + 210), max(24, y1)), (0, 0, 0), -1)
        cv2.putText(
            annotated,
            target_label,
            (x1 + 6, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return annotated


def _airsim_ned_to_carla_location(airsim_pos: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Convert AirSim NED coordinates back to CARLA world coordinates."""
    return _shared_airsim_ned_to_carla_location(airsim_pos, CONFIG)


def _carla_location_to_airsim_ned_tuple(location) -> Tuple[float, float, float]:
    """Convert a CARLA location into AirSim NED coordinates."""
    return _shared_carla_location_to_airsim_ned(location, CONFIG)


def _get_drivable_anchor(world, carla_module, airsim_pos: Tuple[float, float, float]) -> Dict[str, Any]:
    """Return the nearest drivable road waypoint and its local road frame."""
    try:
        carla_x, carla_y, carla_z = _airsim_ned_to_carla_location(airsim_pos)
        carla_loc = carla_module.Location(carla_x, carla_y, carla_z)
        waypoint = world.get_map().get_waypoint(
            carla_loc,
            project_to_road=True,
            lane_type=carla_module.LaneType.Driving,
        )
        if waypoint is not None:
            loc = waypoint.transform.location
            rot = waypoint.transform.rotation
            return {
                "position": _carla_location_to_airsim_ned_tuple(loc),
                "heading": float(rot.yaw),
                "lane_width": float(getattr(waypoint, "lane_width", 3.5)),
            }
    except Exception:
        pass
    return {
        "position": airsim_pos,
        "heading": 0.0,
        "lane_width": 3.5,
    }


def _set_weather(world, carla_module, preset_name: str) -> None:
    """Set CARLA weather using official WeatherParameters presets."""
    presets = {
        "晴天": carla_module.WeatherParameters.ClearNoon,
        "阴天": carla_module.WeatherParameters.CloudyNoon,
        "雨天": carla_module.WeatherParameters.HardRainNoon,
        "黄昏": carla_module.WeatherParameters.ClearSunset,
        "雨天黄昏": carla_module.WeatherParameters.HardRainSunset,
        "薄雾": carla_module.WeatherParameters.SoftRainNoon,
    }
    weather = presets.get(preset_name, carla_module.WeatherParameters.ClearNoon)
    world.set_weather(weather)
    print(f"  [Weather] Set to: {preset_name}")


def _weather_name_for_episode(args: argparse.Namespace, episode_idx: int) -> str:
    sequence = ["晴天", "阴天", "雨天", "黄昏", "雨天黄昏", "薄雾"]
    mode = str(getattr(args, "weather_mode", "cycle")).lower().strip()
    if mode == "fixed":
        return str(getattr(args, "weather_name", "晴天"))
    if mode == "cycle":
        return sequence[episode_idx % len(sequence)]
    return ""


def _load_instruction_list(args: argparse.Namespace) -> List[str]:
    """Load one or more instructions for batch recording."""
    instruction_file = getattr(args, "instruction_file", "")
    if instruction_file:
        path = Path(instruction_file).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"instruction file not found: {path}")
        items = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not items:
            raise ValueError(f"instruction file is empty: {path}")
        return items
    return [str(args.instruction).strip()]


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _altitude_ceiling_for_phase(args: argparse.Namespace, phase: str) -> float:
    phase = str(phase).lower()
    if phase == "search":
        return max(20.0, float(getattr(args, "search_min_altitude", 60.0)))
    if phase == "approach":
        return max(18.0, float(getattr(args, "approach_min_altitude", 48.0)))
    if phase == "top_view":
        return max(15.0, float(getattr(args, "topview_min_altitude", 45.0)))
    return max(15.0, float(getattr(args, "hold_min_altitude", 45.0)))


def clamp_value(value, low, high):
    try:
        value = float(value)
    except Exception:
        return low
    if not math.isfinite(value):
        return low
    return max(low, min(high, value))


def _save_rgb_image(image_np: np.ndarray, path: Path) -> None:
    ok = cv2.imwrite(str(path), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"Failed to save image: {path}")


def _read_scene_image(drone_client, airsim_module) -> np.ndarray:
    responses = drone_client.simGetImages(
        [airsim_module.ImageRequest("0", airsim_module.ImageType.Scene, False, False)]
    )
    if not responses:
        raise RuntimeError("No AirSim image response.")

    resp = responses[0]
    if resp.width <= 0 or resp.height <= 0:
        raise RuntimeError("Invalid AirSim image size.")

    raw = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
    channels = raw.size // (resp.height * resp.width)
    if channels not in (3, 4):
        raise RuntimeError(f"Unexpected AirSim channel count: {channels}")
    img = raw.reshape(resp.height, resp.width, channels)
    if channels == 4:
        img = img[:, :, :3]
    return img


def _target_label_text(color: Optional[str], vehicle_type: str) -> str:
    parts: List[str] = []
    if color:
        parts.append(color)
    if vehicle_type:
        parts.append(vehicle_type)
    return " ".join(parts) if parts else "target"


def _move_to_waypoint(client, position: Tuple[float, float, float], yaw_deg: float, speed: float) -> None:
    client.moveToPositionAsync(
        float(position[0]),
        float(position[1]),
        float(position[2]),
        float(speed),
        drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
        yaw_mode=airsim.YawMode(False, float(yaw_deg)),
    ).join()


def _phase_waypoint(
    phase: str,
    target_state: Dict[str, Any],
    args: argparse.Namespace,
    idx: int,
    total: int,
    world=None,
    carla_module=None,
    visible: bool = False,
) -> Tuple[Tuple[float, float, float], float, str]:
    target_pos = target_state["position"]
    target_heading = float(target_state.get("heading", 0.0))
    heading_rad = math.radians(target_heading)
    forward_x = math.cos(heading_rad)
    forward_y = math.sin(heading_rad)
    right_x = math.sin(heading_rad)
    right_y = -math.cos(heading_rad)

    phase = phase.lower()
    road_anchor = _get_drivable_anchor(world, carla_module, target_pos) if world is not None and carla_module is not None else {"position": target_pos, "heading": target_heading, "lane_width": 3.5}
    road_pos = road_anchor["position"]
    road_heading = math.radians(float(road_anchor.get("heading", target_heading)))
    road_forward_x = math.cos(road_heading)
    road_forward_y = math.sin(road_heading)
    road_right_x = math.sin(road_heading)
    road_right_y = -math.cos(road_heading)
    lane_width = max(3.0, float(road_anchor.get("lane_width", 3.5)))
    altitude_ceiling = _altitude_ceiling_for_phase(args, phase)

    if phase == "search":
        sweep = (idx % 4) - 1.5
        along = (idx / max(total - 1, 1) - 0.5) * lane_width * 10.0
        lateral = clamp_value(sweep * lane_width * 0.18, -lane_width * 0.25, lane_width * 0.25)
        search_height = float(args.search_height)
        if visible:
            search_height = max(search_height - 10.0, 0.0)
        desired_z = road_pos[2] - search_height
        desired_z = min(desired_z, -altitude_ceiling)
        desired = (
            road_pos[0] + road_forward_x * along + road_right_x * lateral,
            road_pos[1] + road_forward_y * along + road_right_y * lateral,
            desired_z,
        )
        view = "forward"
    elif phase == "approach":
        desired_z = road_pos[2] - args.approach_height
        desired_z = min(desired_z, -altitude_ceiling)
        desired = (
            road_pos[0] - road_forward_x * args.follow_distance,
            road_pos[1] - road_forward_y * args.follow_distance,
            desired_z,
        )
        view = "forward"
    elif phase == "top_view":
        desired_z = road_pos[2] - args.top_view_height
        desired_z = min(desired_z, -altitude_ceiling)
        desired = (
            road_pos[0] + road_right_x * clamp_value(args.top_lateral_offset, -lane_width * 0.3, lane_width * 0.3),
            road_pos[1] + road_right_y * clamp_value(args.top_lateral_offset, -lane_width * 0.3, lane_width * 0.3),
            desired_z,
        )
        view = "top"
    else:
        desired_z = road_pos[2] - args.follow_height
        desired_z = min(desired_z, -altitude_ceiling)
        desired = (
            road_pos[0] - road_forward_x * args.follow_distance,
            road_pos[1] - road_forward_y * args.follow_distance,
            desired_z,
        )
        view = "forward"

    yaw = math.degrees(math.atan2(road_pos[1] - desired[1], road_pos[0] - desired[0]))
    return desired, yaw, view


def _record_phase(
    *,
    phase: str,
    phase_seconds: float,
    args: argparse.Namespace,
    run_root: Path,
    episode_root: Path,
    frame_start_idx: int,
    drone_client,
    airsim_module,
    world,
    carla_module,
    target_vehicle,
    instruction: Dict[str, Any],
    target_label: str,
    manifest: List[Dict[str, Any]],
) -> int:
    images_dir = _ensure_dir(episode_root / "images")
    preview_dir = _ensure_dir(episode_root / "preview")
    meta_dir = _ensure_dir(episode_root / "meta")

    steps = max(1, int(round(phase_seconds / args.frame_interval)))
    prev_target_state: Optional[Dict[str, Any]] = None
    kept_frames = 0
    skipped_frames = 0
    search_miss_streak = 0

    for local_idx in range(steps):
        frame_index = frame_start_idx + local_idx

        target_state = get_target_vehicle_state(target_vehicle)
        target_state["timestamp"] = time.time()

        desired_pos, yaw_deg, view_name = _phase_waypoint(
            phase=phase,
            target_state=target_state,
            args=args,
            idx=local_idx,
            total=steps,
            world=world,
            carla_module=carla_module,
        )

        _set_collection_view(drone_client, airsim_module, view_name)
        _move_to_waypoint(drone_client, desired_pos, yaw_deg, args.move_speed)
        time.sleep(args.settle_seconds)

        try:
            collision = drone_client.simGetCollisionInfo()
            if getattr(collision, "has_collided", False):
                print(f"  [WARN] collision detected in phase={phase}, recovering")
                if phase == "search":
                    recovery_floor = -args.search_min_altitude
                elif phase == "approach":
                    recovery_floor = -args.approach_min_altitude
                elif phase == "top_view":
                    recovery_floor = -args.topview_min_altitude
                else:
                    recovery_floor = -args.hold_min_altitude
                recovery_z = min(desired_pos[2], recovery_floor)
                recovery_z = min(recovery_z, -max(args.search_min_altitude + 8.0, 35.0))
                drone_client.moveToZAsync(recovery_z, 2).join()
                time.sleep(args.settle_seconds)
        except Exception:
            pass

        drone_state = get_drone_state(drone_client, airsim_module)
        image_np = _read_scene_image(drone_client, airsim_module)
        image_h, image_w = image_np.shape[:2]

        predicted_target_state = forecast_target_state(
            target_state,
            prev_target_state,
            horizon_seconds=float(args.prediction_horizon),
        )

        decision_plan = {
            "decision": phase,
            "view": view_name,
            "hold_seconds": phase_seconds,
            "reason": "offline_collection",
        }
        bbox_meta = project_target_bbox_to_image(drone_client, airsim_module, target_vehicle)
        bbox = bbox_meta.get("bbox")
        visible = bool(bbox_meta.get("visible"))
        if phase == "search":
            if visible:
                search_miss_streak = 0
            else:
                search_miss_streak += 1
                if search_miss_streak >= 6:
                    # Fast escape: raise altitude and widen search rather than circling near buildings.
                    try:
                        cur_state = get_drone_state(drone_client, airsim_module)
                        escape_z = min(float(cur_state["position"][2]), -max(args.search_min_altitude + 8.0, 35.0))
                        drone_client.moveToZAsync(escape_z, max(3.0, args.move_speed * 0.6)).join()
                        time.sleep(args.settle_seconds)
                    except Exception:
                        pass
                    search_miss_streak = 0
        quality_pass = True
        quality_reason = None
        if visible:
            if not bbox:
                quality_pass = False
                quality_reason = "visible_without_bbox"
            else:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                bbox_w = x2 - x1
                bbox_h = y2 - y1
                edge_margin = int(getattr(args, "edge_margin", 8))
                min_bbox_size = int(getattr(args, "min_visible_bbox_size", 18))
                keep_edge_frames = bool(getattr(args, "keep_edge_frames", False))
                if bbox_w < min_bbox_size or bbox_h < min_bbox_size:
                    quality_pass = False
                    quality_reason = f"small_bbox_{bbox_w}x{bbox_h}"
                elif not keep_edge_frames and (
                    x1 <= edge_margin
                    or y1 <= edge_margin
                    or x2 >= image_w - edge_margin
                    or y2 >= image_h - edge_margin
                ):
                    quality_pass = False
                    quality_reason = "edge_bbox"

        sidecar_meta = {
            "type": "frame",
            "run_root": str(run_root),
            "episode_root": str(episode_root),
            "phase": phase,
            "view": view_name,
            "instruction": instruction,
            "target_label": target_label,
            "decision_plan": decision_plan,
            "annotation": bbox_meta,
            "preview_style": "standard_bbox_only",
            "drone_state": drone_state,
            "target_state": target_state,
            "predicted_target_state": predicted_target_state,
            "source": "offline_collection",
            "frame_index": frame_index,
            "timestamp": time.time(),
            "annotation_style": "standard_bbox_only",
            "coordinate_frame": "airsim_ned",
            "world_units": "m",
            "image_units": "px",
            "bbox_format": "xyxy_px",
            "label_format": "yolo_normalized",
            "state_units": {
                "drone_state": "airsim_ned_m",
                "target_state": "airsim_ned_m",
            },
            "quality_pass": quality_pass,
            "quality_reason": quality_reason,
        }

        if not quality_pass:
            skipped_frames += 1
            print(
                f"  [{frame_index:04d}] phase={phase} view={view_name} "
                f"skip={quality_reason} visible={visible} bbox={bbox if bbox else 'None'}"
            )
            prev_target_state = target_state
            continue

        raw_name = f"{phase}_{frame_index:06d}.png"
        raw_path = images_dir / raw_name
        preview_path = preview_dir / raw_name
        meta_path = meta_dir / f"{phase}_{frame_index:06d}.json"

        _save_rgb_image(image_np, raw_path)
        preview_np = _annotate_standard_preview(image_np, bbox, target_label) if visible else image_np
        _save_rgb_image(preview_np, preview_path)

        # Primary label outputs next to the raw image.
        save_frame_sidecar(
            str(raw_path),
            sidecar_meta,
            yolo_bbox=bbox,
            image_size=(image_np.shape[1], image_np.shape[0]),
        )

        # Extra copy inside meta/ for easier browsing.
        meta_path.write_text(
            json.dumps(sidecar_meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        record = FrameRecord(
            frame_index=frame_index,
            timestamp=sidecar_meta["timestamp"],
            phase=phase,
            view=view_name,
            instruction=instruction,
            raw_image=str(raw_path.relative_to(run_root)),
            preview_image=str(preview_path.relative_to(run_root)),
            sidecar_json=str(raw_path.with_suffix(".json").relative_to(run_root)),
            target_visible=visible,
            target_bbox=bbox,
            target_label=target_label,
            drone_state=drone_state,
            target_state=target_state,
        )
        manifest.append(
            {
                "frame_index": record.frame_index,
                "timestamp": record.timestamp,
                "phase": record.phase,
                "view": record.view,
                "instruction": record.instruction,
                "raw_image": record.raw_image,
                "preview_image": record.preview_image,
                "sidecar_json": record.sidecar_json,
                "target_visible": record.target_visible,
                "target_bbox": record.target_bbox,
                "target_label": record.target_label,
                "quality_pass": quality_pass,
            }
        )

        print(
            f"  [{frame_index:04d}] phase={phase} view={view_name} "
            f"visible={visible} bbox={bbox if bbox else 'None'}"
        )
        kept_frames += 1

        prev_target_state = target_state

    print(
        f"  [phase-summary] phase={phase} kept={kept_frames} skipped={skipped_frames} total={steps}"
    )
    return frame_start_idx + steps


def _cleanup(drone_client, vehicle_actors: List[Any], traffic_actors: List[Any]) -> None:
    try:
        drone_client.hoverAsync().join()
    except Exception:
        pass
    try:
        drone_client.armDisarm(False)
    except Exception:
        pass
    try:
        drone_client.enableApiControl(False)
    except Exception:
        pass
    for actor in vehicle_actors + traffic_actors:
        try:
            actor.destroy()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Record a local find-and-track dataset run.")
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION)
    parser.add_argument("--instruction-file", type=str, default="", help="UTF-8 text file, one instruction per line")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--frame-interval", type=float, default=0.5)
    parser.add_argument("--move-speed", type=float, default=4.5)
    parser.add_argument("--settle-seconds", type=float, default=0.5)
    parser.add_argument("--search-seconds", type=float, default=12.0)
    parser.add_argument("--approach-seconds", type=float, default=12.0)
    parser.add_argument("--topview-seconds", type=float, default=12.0)
    parser.add_argument("--search-orbit-radius", type=float, default=40.0)
    parser.add_argument("--search-height", type=float, default=80.0)
    parser.add_argument("--search-min-altitude", type=float, default=70.0)
    parser.add_argument("--follow-distance", type=float, default=14.0)
    parser.add_argument("--follow-height", type=float, default=60.0)
    parser.add_argument("--approach-height", type=float, default=60.0)
    parser.add_argument("--approach-min-altitude", type=float, default=55.0)
    parser.add_argument("--top-view-height", type=float, default=52.0)
    parser.add_argument("--topview-min-altitude", type=float, default=50.0)
    parser.add_argument("--top-lateral-offset", type=float, default=8.0)
    parser.add_argument("--hold-min-altitude", type=float, default=55.0)
    parser.add_argument("--takeoff-altitude", type=float, default=-85.0)
    parser.add_argument("--min-visible-bbox-size", type=int, default=18)
    parser.add_argument("--edge-margin", type=int, default=8)
    parser.add_argument("--keep-edge-frames", action="store_true")
    parser.add_argument("--prediction-horizon", type=float, default=3.0)
    parser.add_argument("--traffic", type=int, default=8)
    parser.add_argument(
        "--allow-unsafe-traffic",
        action="store_true",
        help="Allow more than eight autopilot vehicles; only use for explicitly documented legacy batches.",
    )
    parser.add_argument("--target-spawn-index", type=int, default=-1)
    parser.add_argument("--traffic-start-index", type=int, default=10)
    parser.add_argument("--weather-mode", type=str, default="cycle", choices=["cycle", "fixed", "off"])
    parser.add_argument("--weather-name", type=str, default="晴天")
    parser.add_argument("--airsim-port", type=int, default=41451)
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--carla-host", type=str, default="127.0.0.1")
    parser.add_argument("--airsim-host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    output_root = _ensure_dir(Path(args.output_root).expanduser().resolve())
    run_root = _ensure_dir(output_root / f"run_{time.strftime('%Y%m%d_%H%M%S')}")
    instruction_list = _load_instruction_list(args)
    profile = {
        "task": "find_track_reacquire",
        "domain": "smart_traffic_inspection",
        "pain_point": "target_reacquisition_success",
        "annotation_style": "standard_bbox_only",
        "instruction": args.instruction,
        "instruction_file": args.instruction_file,
        "instruction_count": len(instruction_list),
        "episodes": max(args.episodes, len(instruction_list)),
        "frame_interval": args.frame_interval,
        "search_seconds": args.search_seconds,
        "approach_seconds": args.approach_seconds,
        "topview_seconds": args.topview_seconds,
        "move_speed": args.move_speed,
        "takeoff_altitude": args.takeoff_altitude,
        "min_visible_bbox_size": args.min_visible_bbox_size,
        "edge_margin": args.edge_margin,
        "keep_edge_frames": args.keep_edge_frames,
        "quality_filter": {
            "visible_frames_only": True,
            "min_visible_bbox_size": args.min_visible_bbox_size,
            "edge_margin": args.edge_margin,
            "keep_edge_frames": args.keep_edge_frames,
        },
        "weather_mode": args.weather_mode,
        "weather_name": args.weather_name,
        "coordinate_frame": "airsim_ned",
        "world_units": "m",
        "image_units": "px",
        "bbox_format": "xyxy_px",
        "label_format": "yolo_normalized",
        "target_spawn_index": args.target_spawn_index,
        "traffic_start_index": args.traffic_start_index,
        "allow_unsafe_traffic": args.allow_unsafe_traffic,
        "output_root": str(output_root),
        "run_root": str(run_root),
        "started_at": datetime.now().isoformat(),
    }
    (run_root / "collection_profile.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("CARLA-Air local dataset collector")
    print(f"  output_root = {output_root}")
    print(f"  run_root = {run_root}")
    print(f"  instruction_count = {len(instruction_list)}")
    print("  mode = offline collection only")

    # Intentionally do not configure TRACKING_LOG_DIR or any remote inference env vars.
    client = None
    airsim_module = None
    world = None
    carla_module = None
    drone_client = None
    target_vehicle = None
    vehicle_actors: List[Any] = []
    traffic_actors: List[Any] = []
    manifest: List[Dict[str, Any]] = []

    try:
        print("\nConnecting CARLA ...")
        client, world, carla_module = _connect_carla_local(args.carla_host, args.carla_port)
        if client is None or world is None or carla_module is None:
            raise SystemExit("Failed to connect to CARLA.")

        print("\nConnecting AirSim ...")
        drone_client = _connect_airsim_local(args.airsim_host, args.airsim_port, args.takeoff_altitude)
        airsim_module = airsim

        print("\nPreparing target and traffic ...")
        first_instruction = instruction_list[0]
        parsed = parse_instruction(first_instruction)
        target_color = parsed.get("color")
        target_vehicle_type = parsed.get("vehicle_type", "sedan")
        target_query = {
            "color": target_color,
            "vehicle_type": target_vehicle_type,
            "raw": first_instruction,
        }
        target_spawn_location = _select_spawn_location(world, args.target_spawn_index)
        target_vehicle = spawn_target_vehicle(
            world,
            carla_module,
            color=target_color,
            vehicle_type=target_vehicle_type,
            spawn_location=target_spawn_location,
        )
        if target_vehicle is None:
            raise RuntimeError("Failed to spawn target vehicle.")
        try:
            target_vehicle.set_autopilot(True)
        except Exception:
            pass
        vehicle_actors.append(target_vehicle)
        traffic_actors = spawn_background_traffic(
            world,
            carla_module,
            traffic_count=args.traffic,
            start_index=args.traffic_start_index,
            allow_unsafe=args.allow_unsafe_traffic,
        )
        target_label = _target_label_text(target_color, target_vehicle_type)

        frame_counter = 0
        total_episodes = max(args.episodes, len(instruction_list))
        if len(instruction_list) == 1 and total_episodes > 1:
            instruction_list = instruction_list * total_episodes
        elif len(instruction_list) < total_episodes:
            repeats = (total_episodes + len(instruction_list) - 1) // len(instruction_list)
            instruction_list = (instruction_list * repeats)[:total_episodes]

        for ep_idx in range(total_episodes):
            episode_instruction = instruction_list[ep_idx]
            episode_root = _ensure_dir(run_root / f"episode_{ep_idx:04d}")
            print(f"\nEpisode {ep_idx:04d} -> {episode_root}")
            print(f"  [Instruction] {episode_instruction}")
            weather_name = _weather_name_for_episode(args, ep_idx)
            if args.weather_mode != "off" and weather_name:
                _set_weather(world, carla_module, weather_name)

            episode_start = frame_counter
            episode_saved_start = len(manifest)
            episode_parsed = parse_instruction(episode_instruction)
            episode_target_color = episode_parsed.get("color", target_color)
            episode_target_vehicle_type = episode_parsed.get("vehicle_type", target_vehicle_type)
            if episode_target_color != target_color or episode_target_vehicle_type != target_vehicle_type:
                raise RuntimeError(
                    "Batch instruction file must keep the same target color/type for one run. "
                    f"Got {episode_target_color}/{episode_target_vehicle_type}, expected {target_color}/{target_vehicle_type}."
                )
            episode_target_query = {
                "color": episode_target_color,
                "vehicle_type": episode_target_vehicle_type,
                "raw": episode_instruction,
            }
            for phase_name, phase_seconds in (
                ("search", args.search_seconds),
                ("approach", args.approach_seconds),
                ("top_view", args.topview_seconds),
            ):
                frame_counter = _record_phase(
                    phase=phase_name,
                    phase_seconds=phase_seconds,
                    args=args,
                    run_root=run_root,
                    episode_root=episode_root,
                frame_start_idx=frame_counter,
                drone_client=drone_client,
                airsim_module=airsim_module,
                world=world,
                carla_module=carla_module,
                target_vehicle=target_vehicle,
                instruction=episode_target_query,
                target_label=target_label,
                manifest=manifest,
            )

            episode_summary = {
                "episode_id": ep_idx,
                "episode_root": str(episode_root),
                "instruction": episode_instruction,
                "target_label": target_label,
                "target_spawn_index": args.target_spawn_index,
                "traffic_start_index": args.traffic_start_index,
                "weather_mode": args.weather_mode,
                "weather_name": weather_name,
                "frame_start": episode_start,
                "frame_end": frame_counter - 1,
                "total_frames": frame_counter - episode_start,
                "saved_frames": len(manifest) - episode_saved_start,
                "phases": [
                    {"name": "search", "seconds": args.search_seconds},
                    {"name": "approach", "seconds": args.approach_seconds},
                    {"name": "top_view", "seconds": args.topview_seconds},
                ],
            }
            (episode_root / "episode_summary.json").write_text(
                json.dumps(episode_summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        (run_root / "annotations.jsonl").write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in manifest) + "\n",
            encoding="utf-8",
        )
        summary = {
            "run_root": str(run_root),
            "episodes": args.episodes,
            "frames": len(manifest),
            "saved_frames": len(manifest),
            "instruction": args.instruction,
            "target_label": target_label,
            "target_spawn_index": args.target_spawn_index,
            "traffic_start_index": args.traffic_start_index,
            "weather_mode": args.weather_mode,
            "takeoff_altitude": args.takeoff_altitude,
            "min_visible_bbox_size": args.min_visible_bbox_size,
            "edge_margin": args.edge_margin,
            "keep_edge_frames": args.keep_edge_frames,
            "weather_name": args.weather_name,
        }
        (run_root / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print("\nRecording complete.")
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        if drone_client is not None:
            try:
                _cleanup(drone_client, vehicle_actors, traffic_actors)
            except Exception:
                pass


if __name__ == "__main__":
    main()
