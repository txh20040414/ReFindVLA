"""CARLA road and building context used by RecoverVLA.

This module keeps map reasoning separate from the AirSim/CARLA adapter. The
policy sees compact, serializable context: road branches, nearby building risk,
and altitude advice.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from .config import RecoverVLAConfig
from .simulator import airsim_ned_to_carla_location, carla_location_to_airsim_ned
from .utils import clamp, distance2, heading_to_unit

Vec3 = Tuple[float, float, float]


def _waypoint_dict(
    waypoint, branch_id: str, confidence: float, reason: str, config: RecoverVLAConfig
) -> Dict[str, Any]:
    loc = waypoint.transform.location
    return {
        "branch_id": branch_id,
        "position": carla_location_to_airsim_ned(loc, config),
        "heading": float(waypoint.transform.rotation.yaw),
        "lane_width": float(getattr(waypoint, "lane_width", 3.5)),
        "is_junction": bool(getattr(waypoint, "is_junction", False)),
        "confidence": clamp(confidence, 0.0, 1.0),
        "reason": reason,
    }


def road_branch_candidates(
    world,
    carla_module,
    center_ned: Vec3,
    heading_hint: float,
    config: RecoverVLAConfig,
) -> List[Dict[str, Any]]:
    """Return Top-K local road branches near the belief center.

    Uses CARLA map waypoints and waypoint.next(). At junctions this naturally
    exposes multiple branch candidates; on straight roads it still provides
    forward/nearby anchors for stable search.
    """

    if world is None or carla_module is None:
        fx, fy = heading_to_unit(heading_hint)
        fallback = (
            center_ned[0] + fx * config.road_branch_distance_m,
            center_ned[1] + fy * config.road_branch_distance_m,
            center_ned[2],
        )
        return [
            {
                "branch_id": "heading_prior",
                "position": fallback,
                "heading": heading_hint,
                "lane_width": 3.5,
                "is_junction": False,
                "confidence": 0.45,
                "reason": "no_carla_map_heading_prior",
            }
        ]

    try:
        x, y, z = airsim_ned_to_carla_location(center_ned, config)
        loc = carla_module.Location(x, y, z)
        root = world.get_map().get_waypoint(loc, project_to_road=True, lane_type=carla_module.LaneType.Driving)
    except Exception:
        root = None
    if root is None:
        return []

    branches: List[Dict[str, Any]] = []
    frontier = [(root, "road_0", 1.0, "nearest_drivable_lane")]
    seen = set()
    for depth in range(max(1, config.road_branch_depth)):
        next_frontier = []
        for waypoint, branch_id, conf, reason in frontier:
            key = (round(waypoint.transform.location.x, 1), round(waypoint.transform.location.y, 1), depth)
            if key in seen:
                continue
            seen.add(key)
            candidate = _waypoint_dict(waypoint, branch_id, conf, reason, config)
            branches.append(candidate)
            try:
                nxt = waypoint.next(config.road_branch_distance_m)
            except Exception:
                nxt = []
            for idx, wp in enumerate(nxt[:4]):
                yaw_delta = abs((float(wp.transform.rotation.yaw) - heading_hint + 180.0) % 360.0 - 180.0)
                branch_conf = conf * clamp(1.0 - yaw_delta / 180.0, 0.35, 0.95) * (0.92 ** depth)
                next_frontier.append((wp, f"{branch_id}_{idx}", branch_conf, "carla_waypoint_next_branch"))
        frontier = next_frontier
        if not frontier:
            break

    unique: Dict[str, Dict[str, Any]] = {}
    for branch in branches:
        pos = branch["position"]
        key = f"{round(pos[0], 0)}:{round(pos[1], 0)}"
        if key not in unique or branch["confidence"] > unique[key]["confidence"]:
            unique[key] = branch
    ranked = sorted(unique.values(), key=lambda item: item["confidence"], reverse=True)
    return ranked[:6]


def building_safety_context(
    world,
    carla_module,
    drone_position: Vec3,
    waypoint: Optional[Vec3],
    config: RecoverVLAConfig,
) -> Dict[str, Any]:
    """Estimate nearby building risk from CARLA level bounding boxes.

    CARLA exposes building boxes through get_level_bbs(CityObjectLabel.Buildings)
    on recent builds. If unavailable, return an explicit unknown context so the
    controller can still use conservative altitude rules.
    """

    if world is None or carla_module is None:
        return {"available": False, "risk": "unknown", "reason": "carla_world_unavailable"}
    try:
        label = carla_module.CityObjectLabel.Buildings
        boxes = world.get_level_bbs(label)
    except Exception as exc:
        return {"available": False, "risk": "unknown", "reason": f"building_bbs_unavailable:{exc}"}

    if not boxes:
        return {"available": True, "risk": "low", "nearest_distance_m": None, "recommended_altitude_m": config.safe_search_altitude_m}

    probe = waypoint or drone_position
    px, py, pz = airsim_ned_to_carla_location(probe, config)
    nearest_distance = float("inf")
    nearest_height = 0.0
    for box in boxes:
        try:
            center = box.location
            extent = box.extent
            dx = max(abs(px - center.x) - extent.x, 0.0)
            dy = max(abs(py - center.y) - extent.y, 0.0)
            d = math.sqrt(dx * dx + dy * dy)
            if d < nearest_distance:
                nearest_distance = d
                nearest_height = max(0.0, center.z + extent.z)
        except Exception:
            continue

    recommended = clamp(nearest_height + config.building_clearance_m, config.safe_search_altitude_m, config.max_search_altitude_m)
    risk = "high" if nearest_distance < config.obstacle_slowdown_distance_m else "medium" if nearest_distance < 35.0 else "low"
    return {
        "available": True,
        "risk": risk,
        "nearest_distance_m": None if not math.isfinite(nearest_distance) else nearest_distance,
        "nearest_height_m": nearest_height,
        "recommended_altitude_m": recommended,
    }


def build_environment_context(
    world,
    carla_module,
    drone_position: Vec3,
    belief_center: Vec3,
    heading_hint: float,
    planned_waypoint: Optional[Vec3],
    config: RecoverVLAConfig,
) -> Dict[str, Any]:
    branches = road_branch_candidates(world, carla_module, belief_center, heading_hint, config)
    safety = building_safety_context(world, carla_module, drone_position, planned_waypoint or belief_center, config)
    return {"road_branches": branches, "building_safety": safety}
