"""CARLA-Air and AirSim adapter functions.

The rest of RecoverVLA talks to plain dict/dataclass state. Only this module
knows about the exact CARLA and AirSim Python APIs.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import RecoverVLAConfig
from .models import DroneState, TargetState
from .utils import clamp


CARLA_VEHICLE_BLUEPRINTS = {
    "sedan": ["vehicle.tesla.model3", "vehicle.audi.a2", "vehicle.bmw.grandtourer", "vehicle.mercedes.coupe"],
    "suv": ["vehicle.tesla.cybertruck", "vehicle.jeep.wrangler_rubicon", "vehicle.nissan.patrol"],
    "truck": ["vehicle.carlamotors.firetruck", "vehicle.carlamotors.carlacola"],
    "bus": ["vehicle.mercedes.sprinter"],
}


def _add_carla_python_paths(root: str) -> None:
    """Add common CARLA-Air Windows/Linux PythonAPI locations.

    CARLA distributions differ: some expose ``PythonAPI`` as a package,
    others ship a versioned egg under ``PythonAPI/carla/dist``.  Importing
    only the root directory is not sufficient for the latter layout.
    """
    if not root:
        return
    root_path = Path(root).expanduser()
    candidates = [root_path / "PythonAPI", root_path / "PythonAPI" / "carla"]
    dist_dir = root_path / "PythonAPI" / "carla" / "dist"
    if dist_dir.is_dir():
        candidates.extend(sorted(dist_dir.glob("*.egg")))
    for candidate in candidates:
        candidate_text = str(candidate)
        if candidate.exists() and candidate_text not in sys.path:
            sys.path.insert(0, candidate_text)


def connect_carla(config: RecoverVLAConfig):
    if config.carla_air_root:
        _add_carla_python_paths(config.carla_air_root)
    import carla

    if config.random_seed:
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)

    client = carla.Client(config.carla_host, config.carla_port)
    client.set_timeout(20.0)
    world = client.get_world()
    print(f"  地图: {world.get_map().name}")
    print(
        "  坐标偏移: "
        f"x={config.coordinate_offset_x:.2f}, y={config.coordinate_offset_y:.2f}, "
        f"z={config.coordinate_offset_z:.2f} ({config.coordinate_offset_source})"
    )
    return client, world, carla


def connect_airsim(config: RecoverVLAConfig):
    import airsim

    client = airsim.MultirotorClient(ip=config.airsim_host, port=config.airsim_port)
    client.confirmConnection()
    print("  AirSim 已连接")
    client.enableApiControl(True)
    client.armDisarm(True)
    client.takeoffAsync().join()
    client.moveToZAsync(config.takeoff_altitude_ned, 3).join()
    if config.camera_control_enabled:
        set_camera_view(client, airsim, "top", config)
    else:
        print("  相机: 使用 AirSim settings.json 静态配置（未调用动态相机 API）")
    print(f"  无人机起飞到 {-config.takeoff_altitude_ned:.1f}m")
    return client, airsim


def set_camera_view(airsim_client, airsim_module, view: str, config: RecoverVLAConfig) -> None:
    if not config.camera_control_enabled:
        return
    pitch_map = {"top": -90.0, "side": -58.0, "forward": -72.0}
    fov_map = {"top": 75.0, "side": 82.0, "forward": 88.0}
    pitch = pitch_map.get(view, config.camera_pitch_deg)
    fov = fov_map.get(view, config.camera_fov_deg)
    # CARLA-Air's Shipping FAQ warns that simSetCameraPose can abort the
    # native process.  Prefer the orientation-only API when available and do
    # not silently fall back to simSetCameraPose.
    try:
        orientation = airsim_module.to_quaternion(math.radians(pitch), 0.0, 0.0)
        setter = getattr(airsim_client, "simSetCameraOrientation", None)
        if setter is None:
            print("  动态相机已启用，但当前 AirSim 客户端没有 simSetCameraOrientation；保持静态相机")
            return
        setter("0", orientation)
        fov_setter = getattr(airsim_client, "simSetCameraFov", None)
        if fov_setter is not None:
            fov_setter("0", fov)
    except Exception as exc:
        print(f"  相机视角切换失败: {exc}")


def _offsets(config: Optional[RecoverVLAConfig]) -> Tuple[float, float, float]:
    if config is None:
        return (172.20, -183.86, 27.45)
    return (config.coordinate_offset_x, config.coordinate_offset_y, config.coordinate_offset_z)


def carla_location_to_airsim_ned(location, config: Optional[RecoverVLAConfig] = None) -> Tuple[float, float, float]:
    offset_x, offset_y, offset_z = _offsets(config)
    return (location.x + offset_x, location.y + offset_y, -location.z + offset_z)


def airsim_ned_to_carla_location(
    airsim_pos: Tuple[float, float, float], config: Optional[RecoverVLAConfig] = None
) -> Tuple[float, float, float]:
    offset_x, offset_y, offset_z = _offsets(config)
    x, y, z = airsim_pos
    return (x - offset_x, y - offset_y, offset_z - z)


def get_drivable_anchor(
    world, carla_module, airsim_pos: Tuple[float, float, float], config: Optional[RecoverVLAConfig] = None
) -> Dict[str, Any]:
    try:
        x, y, z = airsim_ned_to_carla_location(airsim_pos, config)
        loc = carla_module.Location(x, y, z)
        waypoint = world.get_map().get_waypoint(loc, project_to_road=True, lane_type=carla_module.LaneType.Driving)
        if waypoint is not None:
            return {
                "position": carla_location_to_airsim_ned(waypoint.transform.location, config),
                "heading": float(waypoint.transform.rotation.yaw),
                "lane_width": float(getattr(waypoint, "lane_width", 3.5)),
            }
    except Exception:
        pass
    return {"position": airsim_pos, "heading": 0.0, "lane_width": 3.5}


def spawn_target_vehicle(world, carla_module, config: RecoverVLAConfig, color: Optional[str], vehicle_type: str):
    blueprint_lib = world.get_blueprint_library()
    bp_names = CARLA_VEHICLE_BLUEPRINTS.get(vehicle_type, CARLA_VEHICLE_BLUEPRINTS["sedan"])
    vehicle_bp = None
    for name in bp_names:
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
            "white": "255,255,255",
            "black": "0,0,0",
            "red": "255,0,0",
            "blue": "0,0,255",
            "silver": "192,192,192",
            "yellow": "255,255,0",
            "green": "0,255,0",
            "gray": "128,128,128",
            "orange": "255,165,0",
        }
        vehicle_bp.set_attribute("color", color_map.get(color, "255,255,255"))

    spawn_points = list(world.get_map().get_spawn_points())
    candidates = spawn_points[config.traffic_spawn_start_index:] or spawn_points
    random.shuffle(candidates)
    for spawn_point in candidates[:30]:
        actor = world.try_spawn_actor(vehicle_bp, spawn_point)
        if actor:
            print(f"  目标车辆: {actor.type_id} @ ({spawn_point.location.x:.0f}, {spawn_point.location.y:.0f})")
            return actor
    return None


def spawn_background_traffic(world, carla_module, config: RecoverVLAConfig) -> List[Any]:
    blueprint_lib = world.get_blueprint_library()
    vehicles_bp = list(blueprint_lib.filter("vehicle.*"))
    spawn_points = list(world.get_map().get_spawn_points())
    candidates = spawn_points[config.traffic_spawn_start_index:] or spawn_points
    random.shuffle(candidates)
    requested = max(0, int(config.background_traffic))
    effective = requested
    if requested > 8 and not config.allow_unsafe_traffic:
        effective = 8
        print(
            f"  警告: 请求 {requested} 辆背景车；按 CARLA-Air UAV 安全建议限制为 {effective}。"
            " 如需复现旧批次，请显式设置 RECOVER_ALLOW_UNSAFE_TRAFFIC=1 并记录。"
        )
    actors: List[Any] = []
    for spawn_point in candidates:
        if len(actors) >= effective:
            break
        bp = random.choice(vehicles_bp)
        if bp.has_attribute("color"):
            values = bp.get_attribute("color").recommended_values
            if values:
                bp.set_attribute("color", random.choice(values))
        actor = world.try_spawn_actor(bp, spawn_point)
        if actor:
            actor.set_autopilot(True)
            actors.append(actor)
    print(f"  背景车辆: {len(actors)}")
    return actors


def get_target_state(actor, config: Optional[RecoverVLAConfig] = None) -> TargetState:
    import time

    transform = actor.get_transform()
    location = transform.location
    velocity = actor.get_velocity()
    pos = carla_location_to_airsim_ned(location, config)
    vel = (float(velocity.x), float(velocity.y), float(-velocity.z))
    speed_xy = math.sqrt(vel[0] ** 2 + vel[1] ** 2)
    return TargetState(position=pos, velocity=vel, heading=float(transform.rotation.yaw), speed_xy=speed_xy, timestamp=time.time())


def get_drone_state(airsim_client) -> DroneState:
    state = airsim_client.getMultirotorState()
    pos = state.kinematics_estimated.position
    vel = state.kinematics_estimated.linear_velocity
    q = state.kinematics_estimated.orientation
    siny = 2.0 * (q.w_val * q.z_val + q.x_val * q.y_val)
    cosy = 1.0 - 2.0 * (q.y_val * q.y_val + q.z_val * q.z_val)
    yaw = math.degrees(math.atan2(siny, cosy))
    battery = None
    try:
        battery = float(getattr(state, "battery_remaining", None))
    except Exception:
        battery = None
    return DroneState(
        position=(float(pos.x_val), float(pos.y_val), float(pos.z_val)),
        velocity=(float(vel.x_val), float(vel.y_val), float(vel.z_val)),
        heading=float(yaw),
        battery=battery,
    )


def get_drone_image(airsim_client, airsim_module) -> Optional[np.ndarray]:
    responses = airsim_client.simGetImages([airsim_module.ImageRequest("0", airsim_module.ImageType.Scene, False, False)])
    if not responses:
        return None
    response = responses[0]
    img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    if img1d.size == 0 or response.width <= 0 or response.height <= 0:
        return None
    channels = img1d.size // (response.height * response.width)
    image = img1d.reshape(response.height, response.width, channels)
    return image[:, :, :3] if channels == 4 else image


def get_collision_info(airsim_client):
    try:
        return airsim_client.simGetCollisionInfo()
    except Exception:
        return None


def _quat_to_tuple(q) -> Tuple[float, float, float, float]:
    if q is None:
        return (1.0, 0.0, 0.0, 0.0)
    return (
        float(getattr(q, "w_val", getattr(q, "w", 1.0))),
        float(getattr(q, "x_val", getattr(q, "x", 0.0))),
        float(getattr(q, "y_val", getattr(q, "y", 0.0))),
        float(getattr(q, "z_val", getattr(q, "z", 0.0))),
    )


def _rotation_matrix(q) -> np.ndarray:
    w, x, y, z = _quat_to_tuple(q)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def project_target_bbox_to_image(airsim_client, target_actor, config: RecoverVLAConfig) -> Dict[str, Any]:
    try:
        camera_info = airsim_client.simGetCameraInfo("0")
        pose = camera_info.pose
        target_transform = target_actor.get_transform()
        vertices = target_actor.bounding_box.get_world_vertices(target_transform)
    except Exception as exc:
        return {"visible": False, "bbox": None, "error": str(exc)}

    image_w = int(getattr(camera_info, "width", config.image_width) or config.image_width)
    image_h = int(getattr(camera_info, "height", config.image_height) or config.image_height)
    fov = clamp(getattr(camera_info, "fov", config.camera_fov_deg), 1.0, 170.0)
    cam_pos = np.array([pose.position.x_val, pose.position.y_val, pose.position.z_val], dtype=np.float64)
    world_to_camera = _rotation_matrix(pose.orientation).T
    fx = (image_w / 2.0) / math.tan(math.radians(fov) / 2.0)
    fy = fx
    cx, cy = image_w / 2.0, image_h / 2.0

    points = []
    visible_points = 0
    for vertex in vertices:
        wp = np.array(carla_location_to_airsim_ned(vertex, config), dtype=np.float64)
        cp = world_to_camera @ (wp - cam_pos)
        depth = float(cp[0])
        if depth <= 0.15:
            continue
        u = fx * (cp[1] / depth) + cx
        v = fy * (cp[2] / depth) + cy
        if math.isfinite(u) and math.isfinite(v):
            points.append((u, v))
            if 0 <= u < image_w and 0 <= v < image_h:
                visible_points += 1

    if len(points) < 2:
        return {"visible": False, "bbox": None, "visible_points": visible_points, "image_size": (image_w, image_h)}
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    bbox = (
        int(max(0, min(image_w - 1, math.floor(min(xs))))),
        int(max(0, min(image_h - 1, math.floor(min(ys))))),
        int(max(0, min(image_w - 1, math.ceil(max(xs))))),
        int(max(0, min(image_h - 1, math.ceil(max(ys))))),
    )
    visible = bbox[2] > bbox[0] and bbox[3] > bbox[1] and visible_points >= 1
    return {"visible": visible, "bbox": bbox if visible else None, "visible_points": visible_points, "image_size": (image_w, image_h)}
