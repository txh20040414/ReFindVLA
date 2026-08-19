"""High-level VLA decision parsing and local fallback policy."""

from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, Optional

from .belief_map import BeliefMap
from .candidate_verification import CandidateVerifier
from .config import RecoverVLAConfig
from .models import BeliefRegion, CandidateAssessment, DecisionPlan, DroneState, TargetMemory, TargetState
from .simulator import get_drivable_anchor
from .utils import clamp, heading_to_unit, phase_name, right_unit, view_name


def build_vla_prompt(
    instruction: str,
    drone: DroneState,
    memory: TargetMemory,
    belief: BeliefRegion,
    candidate: CandidateAssessment,
    last_plan: Optional[DecisionPlan],
) -> str:
    payload = {
        "instruction": instruction,
        "uav_state": drone.to_dict(),
        "target_memory": memory.to_prompt_dict(__import__("time").time()),
        "belief_region": belief.to_dict(),
        "candidate_assessment": candidate.to_dict(),
        "last_plan": last_plan.to_dict() if last_plan else None,
    }
    return (
        "你是 RecoverVLA 的高层决策器。任务是语言意图驱动的无人机动态目标找回。"
        "不要输出底层速度，只输出一个 JSON 对象，不要 Markdown。\n"
        "可选 phase: search, inspect, confirm, reacquire, follow, return_home。\n"
        "可选 view: top, side, forward。\n"
        "如果目标不可见，要根据 target_memory 和 belief_region 主动搜索；"
        "如果有疑似目标但置信度不足，选择 inspect/confirm；"
        "只有候选确认置信度足够高才选择 reacquire/follow。\n"
        "输出格式示例: "
        '{"phase":"search","view":"top","target_found":false,"confidence":0.4,'
        '"hold_seconds":5,"reason":"沿最后方向搜索"}\n'
        f"<recover_context>{json.dumps(payload, ensure_ascii=False)}</recover_context>"
    )


def parse_model_json(result: Dict[str, Any]) -> Dict[str, Any]:
    raw = result.get("action")
    if isinstance(raw, dict):
        return raw
    response = str(result.get("response", "") or "").strip()
    if not response:
        return {}
    response = response.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        data = json.loads(response)
        return data if isinstance(data, dict) else {}
    except Exception:
        match = re.search(r"\{.*\}", response, flags=re.S)
        if match:
            try:
                data = json.loads(match.group(0))
                return data if isinstance(data, dict) else {}
            except Exception:
                return {}
    return {}


class SemanticWaypointPlanner:
    def __init__(self, config: RecoverVLAConfig):
        self.config = config

    def _altitude_z(self, altitude_m: float) -> float:
        altitude_m = clamp(altitude_m, self.config.safe_min_altitude_m, self.config.max_search_altitude_m)
        return -altitude_m

    def waypoint(
        self,
        phase: str,
        view: str,
        drone: DroneState,
        belief: BeliefRegion,
        target: Optional[TargetState],
        world=None,
        carla_module=None,
    ):
        ref_pos = target.position if target else belief.center
        ref_heading = target.heading if target else belief.heading
        road = get_drivable_anchor(world, carla_module, ref_pos) if world is not None and carla_module is not None else {
            "position": ref_pos,
            "heading": ref_heading,
            "lane_width": 3.5,
        }
        road_pos = road["position"]
        road_heading = float(road.get("heading", ref_heading))
        fx, fy = heading_to_unit(road_heading)
        rx, ry = right_unit(road_heading)
        lane_offset = min(self.config.side_offset_m, max(3.5, float(road.get("lane_width", 3.5))) * 1.3)

        if phase == "follow":
            tx, ty, tz = ref_pos
            fx2, fy2 = heading_to_unit(ref_heading)
            waypoint = (tx - fx2 * self.config.follow_distance_m, ty - fy2 * self.config.follow_distance_m, self._altitude_z(self.config.follow_altitude_m))
        elif phase in {"inspect", "confirm"}:
            side = 1.0 if math.sin(math.radians(drone.heading - road_heading)) >= 0 else -1.0
            waypoint = (road_pos[0] + rx * lane_offset * side, road_pos[1] + ry * lane_offset * side, self._altitude_z(self.config.inspect_altitude_m))
        elif phase == "return_home":
            waypoint = (0.0, 0.0, self._altitude_z(self.config.safe_search_altitude_m))
        else:
            # Search/reacquire: go slightly ahead of the belief center and use a
            # high-but-not-too-high top view to avoid confusing similar vehicles.
            ahead = min(self.config.search_radius_m, max(18.0, belief.radius_m * 0.45))
            lateral = math.sin(__import__("time").time() * 0.4) * min(18.0, belief.radius_m * 0.25)
            waypoint = (
                road_pos[0] + fx * ahead + rx * lateral,
                road_pos[1] + fy * ahead + ry * lateral,
                self._altitude_z(self.config.safe_search_altitude_m),
            )

        yaw = math.degrees(math.atan2(ref_pos[1] - drone.position[1], ref_pos[0] - drone.position[0]))
        return waypoint, yaw


class RecoverVLADecisionEngine:
    def __init__(self, config: RecoverVLAConfig):
        self.config = config
        self.belief = BeliefMap(config)
        self.verifier = CandidateVerifier(config.confirm_threshold, config.candidate_margin)
        self.waypoints = SemanticWaypointPlanner(config)

    def make_plan(
        self,
        drone: DroneState,
        memory: TargetMemory,
        observed: Optional[TargetState],
        predicted: Optional[TargetState],
        bbox_meta: Dict[str, Any],
        model_result: Optional[Dict[str, Any]],
        last_plan: Optional[DecisionPlan],
        world=None,
        carla_module=None,
        candidate_list=None,
        environment=None,
    ) -> DecisionPlan:
        belief = self.belief.build(memory, predicted, environment=environment)
        candidate = self.verifier.score(bbox_meta, observed, memory, belief, candidate_list)
        raw = parse_model_json(model_result or {})

        has_candidate = candidate.bbox is not None
        local_phase = "follow" if candidate.target_found else ("inspect" if candidate.confidence >= self.config.inspect_threshold and has_candidate else "search")
        if memory.lost_since is not None:
            local_phase = "reacquire" if candidate.target_found else ("inspect" if has_candidate and candidate.confidence >= self.config.inspect_threshold else "search")

        phase = phase_name(raw.get("phase") or raw.get("decision") or local_phase)
        confidence = clamp(raw.get("confidence", candidate.confidence), 0.0, 1.0)
        target_found = bool(raw.get("target_found", candidate.target_found)) and candidate.confidence >= self.config.memory_update_threshold
        if phase in {"reacquire", "follow"} and not target_found:
            phase = "inspect" if candidate.confidence >= self.config.inspect_threshold and has_candidate else "search"
        if phase == "search" and target_found:
            phase = "reacquire"
        view = view_name(raw.get("view"), phase)
        if phase in {"search", "reacquire"}:
            view = "top"
        elif phase in {"inspect", "confirm"} and view == "forward":
            view = "side"

        hold = clamp(raw.get("hold_seconds", self.config.decision_interval), 3.0, 10.0)
        waypoint, yaw = self.waypoints.waypoint(phase, view, drone, belief, observed or predicted, world, carla_module)
        safety = dict(belief.safety or {})
        if safety.get("available") and safety.get("risk") in {"high", "medium"} and phase in {"search", "reacquire"}:
            altitude = clamp(safety.get("recommended_altitude_m", self.config.safe_search_altitude_m), self.config.safe_search_altitude_m, self.config.max_search_altitude_m)
            waypoint = (waypoint[0], waypoint[1], -altitude)
        reason = str(raw.get("reason") or candidate.reason or belief.reason)
        return DecisionPlan(
            phase=phase,
            view=view,
            target_found=target_found,
            confidence=confidence,
            waypoint=waypoint,
            desired_yaw=yaw,
            hold_seconds=hold,
            reason=reason,
            search_region=belief,
            candidate=candidate,
            raw_model=raw,
            source="remote_vla" if raw else "local_fallback",
        )
