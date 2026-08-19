"""Low-level safety controller for executing semantic RecoverVLA waypoints."""

from __future__ import annotations

import math

from .config import RecoverVLAConfig
from .models import ControlAction, DecisionPlan, DroneState
from .utils import clamp, normalize_angle_deg


class SafetyController:
    def __init__(self, config: RecoverVLAConfig):
        self.config = config

    def action(self, drone: DroneState, plan: DecisionPlan, duration: float) -> ControlAction:
        waypoint = plan.waypoint
        current_altitude = -drone.position[2]
        desired_altitude = -waypoint[2]
        safety = (plan.search_region.safety if plan.search_region else {}) or {}
        if current_altitude < self.config.safe_min_altitude_m:
            waypoint = (drone.position[0], drone.position[1], -self.config.safe_search_altitude_m)
        elif safety.get("risk") == "high" and plan.phase in {"search", "reacquire"} and current_altitude + 8.0 < desired_altitude:
            waypoint = (drone.position[0], drone.position[1], waypoint[2])

        dx = waypoint[0] - drone.position[0]
        dy = waypoint[1] - drone.position[1]
        dz = waypoint[2] - drone.position[2]
        yaw = math.radians(drone.heading)
        forward_err = math.cos(yaw) * dx + math.sin(yaw) * dy
        right_err = -math.sin(yaw) * dx + math.cos(yaw) * dy
        dist_3d = math.sqrt(dx * dx + dy * dy + dz * dz)

        vx = clamp(forward_err * self.config.waypoint_gain_xy, -self.config.max_body_speed, self.config.max_body_speed)
        vy = clamp(right_err * self.config.waypoint_gain_xy, -self.config.max_body_speed, self.config.max_body_speed)
        vz = clamp(dz * self.config.waypoint_gain_z, -self.config.max_vertical_speed, self.config.max_vertical_speed)
        if safety.get("risk") == "high":
            vx *= 0.45
            vy *= 0.45
        if dist_3d < 4.0:
            vx *= 0.35
            vy *= 0.35
            vz *= 0.5
        yaw_rate = clamp(
            normalize_angle_deg(plan.desired_yaw - drone.heading) * self.config.yaw_gain,
            -self.config.max_yaw_rate,
            self.config.max_yaw_rate,
        )
        return ControlAction(vx=vx, vy=vy, vz=vz, yaw_rate=yaw_rate, duration=duration, distance_3d=dist_3d)
