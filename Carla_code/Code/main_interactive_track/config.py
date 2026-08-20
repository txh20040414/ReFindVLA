"""Configuration for the RecoverVLA interactive CARLA-Air runner."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RecoverVLAConfig:
    server_url: str = os.environ.get("SERVER_URL", "http://localhost:8000")
    carla_host: str = os.environ.get("CARLA_HOST", "127.0.0.1")
    carla_port: int = int(os.environ.get("CARLA_PORT", "2000"))
    airsim_host: str = os.environ.get("AIRSIM_HOST", "127.0.0.1")
    airsim_port: int = int(os.environ.get("AIRSIM_PORT", "41451"))
    carla_air_root: str = os.environ.get(
        "CARLA_AIR_ROOT",
        "",
    )

    # Town10HD v0.1.6 measured CARLA-Air frame offsets.  The official
    # CARLA-Air guide states that these depend on the map and PlayerStart;
    # keep them configurable and record their source in every run.
    coordinate_offset_x: float = float(os.environ.get("CARLA_AIR_OFFSET_X", "172.20"))
    coordinate_offset_y: float = float(os.environ.get("CARLA_AIR_OFFSET_Y", "-183.86"))
    coordinate_offset_z: float = float(os.environ.get("CARLA_AIR_OFFSET_Z", "27.45"))
    coordinate_offset_source: str = os.environ.get(
        "CARLA_AIR_OFFSET_SOURCE",
        "CARLA-Air Town10HD v0.1.6 measured transform; verify for the active map/PlayerStart",
    )
    random_seed: int = int(os.environ.get("RECOVER_RANDOM_SEED", "0"))

    log_dir: str = os.environ.get("TRACKING_LOG_DIR", os.path.join("..", "Data", "decision_runs"))
    max_steps: int = int(os.environ.get("RECOVER_MAX_STEPS", "240"))
    control_dt: float = float(os.environ.get("RECOVER_CONTROL_DT", "0.5"))
    decision_interval: float = float(os.environ.get("RECOVER_DECISION_INTERVAL", "5.0"))
    prediction_margin_s: float = float(os.environ.get("RECOVER_PREDICTION_MARGIN", "1.0"))
    confirm_threshold: float = float(os.environ.get("RECOVER_CONFIRM_THRESHOLD", "0.72"))
    inspect_threshold: float = float(os.environ.get("RECOVER_INSPECT_THRESHOLD", "0.52"))
    memory_update_threshold: float = float(os.environ.get("RECOVER_MEMORY_UPDATE_THRESHOLD", "0.70"))
    candidate_margin: float = float(os.environ.get("RECOVER_CANDIDATE_MARGIN", "0.06"))

    takeoff_altitude_ned: float = float(os.environ.get("RECOVER_TAKEOFF_Z", "-85.0"))
    max_body_speed: float = float(os.environ.get("RECOVER_MAX_BODY_SPEED", "4.0"))
    max_vertical_speed: float = float(os.environ.get("RECOVER_MAX_VERTICAL_SPEED", "1.5"))
    max_yaw_rate: float = float(os.environ.get("RECOVER_MAX_YAW_RATE", "16.0"))
    waypoint_gain_xy: float = float(os.environ.get("RECOVER_WAYPOINT_GAIN_XY", "0.08"))
    waypoint_gain_z: float = float(os.environ.get("RECOVER_WAYPOINT_GAIN_Z", "0.16"))
    yaw_gain: float = float(os.environ.get("RECOVER_YAW_GAIN", "0.35"))

    safe_min_altitude_m: float = float(os.environ.get("RECOVER_SAFE_MIN_ALTITUDE", "35.0"))
    safe_search_altitude_m: float = float(os.environ.get("RECOVER_SAFE_SEARCH_ALTITUDE", "65.0"))
    max_search_altitude_m: float = float(os.environ.get("RECOVER_MAX_SEARCH_ALTITUDE", "90.0"))
    inspect_altitude_m: float = float(os.environ.get("RECOVER_INSPECT_ALTITUDE", "48.0"))
    follow_altitude_m: float = float(os.environ.get("RECOVER_FOLLOW_ALTITUDE", "55.0"))
    follow_distance_m: float = float(os.environ.get("RECOVER_FOLLOW_DISTANCE", "15.0"))
    side_offset_m: float = float(os.environ.get("RECOVER_SIDE_OFFSET", "14.0"))
    search_radius_m: float = float(os.environ.get("RECOVER_SEARCH_RADIUS", "45.0"))
    road_ahead_time_s: float = float(os.environ.get("RECOVER_ROAD_AHEAD_TIME", "6.0"))
    road_branch_distance_m: float = float(os.environ.get("RECOVER_ROAD_BRANCH_DISTANCE", "35.0"))
    road_branch_depth: int = int(os.environ.get("RECOVER_ROAD_BRANCH_DEPTH", "3"))
    building_clearance_m: float = float(os.environ.get("RECOVER_BUILDING_CLEARANCE", "12.0"))
    obstacle_slowdown_distance_m: float = float(os.environ.get("RECOVER_OBSTACLE_SLOWDOWN_DISTANCE", "18.0"))

    camera_pitch_deg: float = float(os.environ.get("RECOVER_CAMERA_PITCH", "-80.0"))
    camera_fov_deg: float = float(os.environ.get("RECOVER_CAMERA_FOV", "90.0"))
    # CARLA-Air Shipping builds warn that simSetCameraPose can abort the
    # process.  Static settings.json camera configuration is therefore the
    # safe default.  Dynamic orientation is an explicit opt-in only.
    camera_control_enabled: bool = _env_bool("RECOVER_CAMERA_CONTROL", False)
    image_width: int = int(os.environ.get("RECOVER_IMAGE_WIDTH", "640"))
    image_height: int = int(os.environ.get("RECOVER_IMAGE_HEIGHT", "480"))

    remote_required: bool = _env_bool("RECOVER_REMOTE_REQUIRED", True)
    use_remote_vla: bool = _env_bool("RECOVER_USE_REMOTE_VLA", True)
    prelock_enabled: bool = _env_bool("RECOVER_PRELOCK", True)
    save_control_frames: bool = _env_bool("RECOVER_SAVE_CONTROL_FRAMES", True)
    visible_frame_stride: int = int(os.environ.get("RECOVER_VISIBLE_FRAME_STRIDE", "2"))
    invisible_frame_stride: int = int(os.environ.get("RECOVER_INVISIBLE_FRAME_STRIDE", "6"))

    # CARLA-Air's FAQ recommends keeping autopilot traffic at or below eight
    # vehicles in the UAV scene.  A larger legacy batch requires an explicit
    # unsafe override and must be recorded as such.
    background_traffic: int = int(os.environ.get("RECOVER_BACKGROUND_TRAFFIC", "8"))
    allow_unsafe_traffic: bool = _env_bool("RECOVER_ALLOW_UNSAFE_TRAFFIC", False)
    traffic_spawn_start_index: int = int(os.environ.get("RECOVER_TRAFFIC_SPAWN_START", "10"))
    remote_predict_timeout_s: float = float(os.environ.get("RECOVER_REMOTE_TIMEOUT", "60"))


CONFIG = RecoverVLAConfig()
