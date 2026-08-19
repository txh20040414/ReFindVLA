"""Local belief search model for RecoverVLA.

This is the latency-robust part of the method. It extrapolates from the last
observed position, speed and direction, then creates a bounded search region.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional

from .config import RecoverVLAConfig
from .models import BeliefRegion, TargetMemory, TargetState
from .utils import clamp, heading_to_unit


def predict_target_state(
    current: Optional[TargetState],
    previous: Optional[TargetState],
    horizon_s: float,
    max_speed_xy: float = 22.0,
    max_accel_xy: float = 6.0,
) -> Optional[TargetState]:
    if current is None:
        return None
    vx, vy, vz = current.velocity
    ax = ay = az = 0.0
    if previous is not None:
        dt = max(current.timestamp - previous.timestamp, 1e-3)
        ax = (vx - previous.velocity[0]) / dt
        ay = (vy - previous.velocity[1]) / dt
        az = (vz - previous.velocity[2]) / dt
        accel_xy = math.sqrt(ax * ax + ay * ay)
        if accel_xy > max_accel_xy:
            scale = max_accel_xy / accel_xy
            ax, ay = ax * scale, ay * scale

    pvx = vx + ax * horizon_s
    pvy = vy + ay * horizon_s
    pvz = vz + az * horizon_s
    speed = math.sqrt(pvx * pvx + pvy * pvy)
    if speed > max_speed_xy:
        scale = max_speed_xy / speed
        pvx, pvy = pvx * scale, pvy * scale
        speed = max_speed_xy

    px = current.position[0] + vx * horizon_s + 0.5 * ax * horizon_s * horizon_s
    py = current.position[1] + vy * horizon_s + 0.5 * ay * horizon_s * horizon_s
    pz = current.position[2] + vz * horizon_s + 0.5 * az * horizon_s * horizon_s
    heading = math.degrees(math.atan2(pvy, pvx)) if speed > 0.1 else current.heading
    confidence = clamp(0.95 - 0.055 * horizon_s - (0.12 if previous is None else 0.0), 0.25, 0.95)
    return TargetState((px, py, pz), (pvx, pvy, pvz), heading, speed, time.time(), source="predicted", confidence=confidence)


class BeliefMap:
    def __init__(self, config: RecoverVLAConfig):
        self.config = config

    def horizon(self, last_latency_ms: Optional[float]) -> float:
        latency_s = float(last_latency_ms or 0.0) / 1000.0
        return clamp(self.config.decision_interval + latency_s + self.config.prediction_margin_s, 2.5, 9.0)

    def build(
        self,
        memory: TargetMemory,
        predicted: Optional[TargetState],
        now: Optional[float] = None,
        environment: Optional[Dict[str, Any]] = None,
    ) -> BeliefRegion:
        now = time.time() if now is None else now
        branches = list((environment or {}).get("road_branches") or [])
        safety = dict((environment or {}).get("building_safety") or {})
        if predicted is not None:
            speed_term = min(35.0, predicted.speed_xy * 2.5)
            radius = clamp(12.0 + speed_term + (1.0 - predicted.confidence) * 45.0, 15.0, 80.0)
            branches = self._weight_branches(branches, predicted)
            return BeliefRegion(
                center=predicted.position,
                radius_m=radius,
                heading=predicted.heading,
                confidence=predicted.confidence,
                reason="constant_velocity_prediction_from_target_memory",
                candidates=branches,
                safety=safety,
            )

        if memory.last_seen is not None:
            lost_time = 0.0 if memory.lost_since is None else max(0.0, now - memory.lost_since)
            fx, fy = heading_to_unit(memory.last_seen.heading)
            center = (
                memory.last_seen.position[0] + fx * min(70.0, lost_time * max(5.0, memory.last_seen.speed_xy)),
                memory.last_seen.position[1] + fy * min(70.0, lost_time * max(5.0, memory.last_seen.speed_xy)),
                memory.last_seen.position[2],
            )
            radius = clamp(25.0 + lost_time * 5.0, 25.0, 95.0)
            pseudo = TargetState(center, memory.last_seen.velocity, memory.last_seen.heading, memory.last_seen.speed_xy, now, source="last_seen_extrapolation", confidence=0.35)
            branches = self._weight_branches(branches, pseudo)
            return BeliefRegion(
                center=center,
                radius_m=radius,
                heading=memory.last_seen.heading,
                confidence=0.35,
                reason="last_seen_extrapolation",
                candidates=branches,
                safety=safety,
            )

        return BeliefRegion(
            center=(0.0, 0.0, -self.config.safe_search_altitude_m),
            radius_m=90.0,
            heading=0.0,
            confidence=0.1,
            reason="no_target_memory",
            candidates=branches,
            safety=safety,
        )

    def _weight_branches(self, branches: list, predicted: TargetState) -> list:
        if not branches:
            return []
        weighted = []
        for branch in branches:
            pos = branch.get("position", predicted.position)
            dx = float(pos[0]) - predicted.position[0]
            dy = float(pos[1]) - predicted.position[1]
            dist = math.sqrt(dx * dx + dy * dy)
            heading_delta = abs((float(branch.get("heading", predicted.heading)) - predicted.heading + 180.0) % 360.0 - 180.0)
            motion = clamp(1.0 - dist / max(25.0, self.config.search_radius_m * 2.2), 0.0, 1.0)
            direction = clamp(1.0 - heading_delta / 170.0, 0.15, 1.0)
            prior = clamp(branch.get("confidence", 0.45), 0.0, 1.0)
            item = dict(branch)
            item["prob"] = clamp(0.50 * motion + 0.30 * direction + 0.20 * prior, 0.0, 1.0)
            weighted.append(item)
        weighted.sort(key=lambda item: item["prob"], reverse=True)
        return weighted[:5]
