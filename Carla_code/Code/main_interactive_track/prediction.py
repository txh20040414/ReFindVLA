"""Local target anticipation helpers for hierarchical VLA tracking.

This module keeps the fast, model-agnostic part of the system separate from
the cloud VLA loop:
 - estimate a future target state from recent observations
 - estimate a reasonable prediction horizon
 - project a 3D world point to the AirSim image plane for overlays
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


def clamp_value(value: float, low: float, high: float) -> float:
    try:
        value = float(value)
    except Exception:
        return low
    if not math.isfinite(value):
        return low
    return max(low, min(high, value))


def _vec3(value: Any, fallback: Sequence[float] = (0.0, 0.0, 0.0)) -> Tuple[float, float, float]:
    if value is None:
        return float(fallback[0]), float(fallback[1]), float(fallback[2])
    try:
        x, y, z = value
        return float(x), float(y), float(z)
    except Exception:
        return float(fallback[0]), float(fallback[1]), float(fallback[2])


def _vector_norm(vec: Sequence[float]) -> float:
    return float(math.sqrt(sum(float(v) * float(v) for v in vec)))


def _heading_from_velocity(velocity: Sequence[float], fallback_heading: float = 0.0) -> float:
    vx, vy, _ = _vec3(velocity)
    speed_xy = math.sqrt(vx * vx + vy * vy)
    if speed_xy < 1e-3:
        return float(fallback_heading)
    return math.degrees(math.atan2(vy, vx))


def estimate_prediction_horizon(
    decision_interval: float = 6.0,
    model_latency_ms: Optional[float] = None,
    safety_margin_seconds: float = 1.0,
    min_horizon: float = 2.5,
    max_horizon: float = 8.0,
) -> float:
    """Estimate how far ahead the local predictor should look.

    The intuition is simple: if the model response arrives late, the target has
    already moved. A short additional horizon helps the UAV move toward the
    region where the target is likely to be when the next decision matters.
    """

    latency_seconds = float(model_latency_ms or 0.0) / 1000.0
    horizon = float(decision_interval) + latency_seconds + float(safety_margin_seconds)
    return clamp_value(horizon, min_horizon, max_horizon)


def forecast_target_state(
    current_state: Optional[Dict[str, Any]],
    previous_state: Optional[Dict[str, Any]] = None,
    horizon_seconds: float = 6.0,
    max_speed_xy: float = 20.0,
    max_acceleration_xy: float = 6.0,
) -> Optional[Dict[str, Any]]:
    """Predict a future target state using constant-velocity with acceleration hint.

    current_state / previous_state are expected to follow the structure returned
    by ``get_target_vehicle_state`` in the main tracking module.
    """

    if current_state is None:
        return None

    pos = _vec3(current_state.get("position"))
    vel = _vec3(current_state.get("velocity"))
    prev_pos = _vec3(previous_state.get("position")) if previous_state else None
    prev_vel = _vec3(previous_state.get("velocity")) if previous_state else None

    accel = (0.0, 0.0, 0.0)
    dt = None
    if previous_state is not None:
        try:
            current_ts = float(current_state.get("timestamp", time.time()))
            previous_ts = float(previous_state.get("timestamp", current_ts - 0.5))
            dt = max(current_ts - previous_ts, 1e-3)
        except Exception:
            dt = None
        if dt is not None and prev_vel is not None:
            accel = (
                (vel[0] - prev_vel[0]) / dt,
                (vel[1] - prev_vel[1]) / dt,
                (vel[2] - prev_vel[2]) / dt,
            )

    accel_xy = math.sqrt(accel[0] * accel[0] + accel[1] * accel[1])
    if accel_xy > max_acceleration_xy and accel_xy > 1e-6:
        scale = max_acceleration_xy / accel_xy
        accel = (accel[0] * scale, accel[1] * scale, accel[2] * scale)

    pred_vel = (
        vel[0] + accel[0] * horizon_seconds,
        vel[1] + accel[1] * horizon_seconds,
        vel[2] + accel[2] * horizon_seconds,
    )
    speed_xy = math.sqrt(pred_vel[0] * pred_vel[0] + pred_vel[1] * pred_vel[1])
    if speed_xy > max_speed_xy and speed_xy > 1e-6:
        scale = max_speed_xy / speed_xy
        pred_vel = (pred_vel[0] * scale, pred_vel[1] * scale, pred_vel[2])

    pred_pos = (
        pos[0] + vel[0] * horizon_seconds + 0.5 * accel[0] * horizon_seconds * horizon_seconds,
        pos[1] + vel[1] * horizon_seconds + 0.5 * accel[1] * horizon_seconds * horizon_seconds,
        pos[2] + vel[2] * horizon_seconds + 0.5 * accel[2] * horizon_seconds * horizon_seconds,
    )

    speed_xy = math.sqrt(pred_vel[0] * pred_vel[0] + pred_vel[1] * pred_vel[1])
    heading = current_state.get("heading")
    if heading is None:
        heading = _heading_from_velocity(pred_vel, fallback_heading=0.0)
    else:
        try:
            heading = float(heading)
        except Exception:
            heading = _heading_from_velocity(pred_vel, fallback_heading=0.0)

    base_speed_xy = math.sqrt(vel[0] * vel[0] + vel[1] * vel[1])
    confidence = 1.0
    if previous_state is None:
        confidence *= 0.82
    else:
        confidence *= 0.92
        if dt is not None and dt > 1.0:
            confidence *= 0.85
    confidence *= max(0.35, 1.0 - 0.05 * float(horizon_seconds))
    confidence = clamp_value(confidence, 0.2, 1.0)

    return {
        "position": pred_pos,
        "velocity": pred_vel,
        "heading": heading,
        "speed_xy": speed_xy,
        "base_speed_xy": base_speed_xy,
        "timestamp": time.time(),
        "source": "predicted",
        "confidence": confidence,
        "horizon_seconds": float(horizon_seconds),
        "acceleration": accel,
        "previous_position": prev_pos,
    }


def project_world_point_to_image(airsim_client, airsim_module, world_point, camera_name: str = "0") -> Dict[str, Any]:
    """Project a 3D point in AirSim NED coordinates into the current camera image."""

    import math as _math

    try:
        camera_info = airsim_client.simGetCameraInfo(camera_name)
    except Exception as e:
        return {"visible": False, "error": f"camera_info_failed: {e}"}

    image_w = int(round(getattr(camera_info, "width", 640) or 640))
    image_h = int(round(getattr(camera_info, "height", 480) or 480))
    try:
        fov_deg = float(getattr(camera_info, "fov", 90.0))
    except Exception:
        fov_deg = 90.0
    if not math.isfinite(fov_deg) or fov_deg <= 0:
        fov_deg = 90.0

    pose = getattr(camera_info, "pose", None)
    if pose is None:
        return {"visible": False, "error": "camera_pose_missing", "image_size": (image_w, image_h)}

    camera_pos = pose.position
    camera_rot = pose.orientation
    camera_pos = np.array(
        [
            float(getattr(camera_pos, "x_val", 0.0)),
            float(getattr(camera_pos, "y_val", 0.0)),
            float(getattr(camera_pos, "z_val", 0.0)),
        ],
        dtype=np.float64,
    )

    # AirSim quaternion is wxyz.
    qw = float(getattr(camera_rot, "w_val", 1.0))
    qx = float(getattr(camera_rot, "x_val", 0.0))
    qy = float(getattr(camera_rot, "y_val", 0.0))
    qz = float(getattr(camera_rot, "z_val", 0.0))
    # Rotation matrix from quaternion.
    R = np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )
    world_to_camera = R.T

    fx = (image_w / 2.0) / _math.tan(_math.radians(fov_deg) / 2.0)
    fy = fx
    cx = image_w / 2.0
    cy = image_h / 2.0

    world_point = np.array(_vec3(world_point), dtype=np.float64)
    cam_point = world_to_camera @ (world_point - camera_pos)
    depth = float(cam_point[0])
    if depth <= 0.15:
        return {
            "visible": False,
            "depth": depth,
            "image_size": (image_w, image_h),
            "camera_fov": fov_deg,
        }

    u = fx * (cam_point[1] / depth) + cx
    v = fy * (cam_point[2] / depth) + cy
    visible = math.isfinite(u) and math.isfinite(v) and 0 <= u < image_w and 0 <= v < image_h
    return {
        "visible": visible,
        "pixel": (float(u), float(v)),
        "depth": depth,
        "image_size": (image_w, image_h),
        "camera_fov": fov_deg,
    }


def build_prediction_summary(predicted_state: Optional[Dict[str, Any]], source: str = "predicted") -> str:
    if predicted_state is None:
        return f"{source}: none"
    pos = predicted_state.get("position", (0.0, 0.0, 0.0))
    vel = predicted_state.get("velocity", (0.0, 0.0, 0.0))
    conf = predicted_state.get("confidence", 0.0)
    horizon = predicted_state.get("horizon_seconds", 0.0)
    return (
        f"{source}: x={pos[0]:.1f} y={pos[1]:.1f} z={pos[2]:.1f} "
        f"vx={vel[0]:.1f} vy={vel[1]:.1f} vz={vel[2]:.1f} "
        f"conf={conf:.2f} horizon={horizon:.1f}s"
    )
