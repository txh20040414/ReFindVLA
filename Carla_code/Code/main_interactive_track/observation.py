"""Vehicle observation and memory gating for RecoverVLA."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from .config import RecoverVLAConfig
from .models import CandidateAssessment, TargetMemory, TargetState
from .simulator import get_target_state, project_target_bbox_to_image


def observe_vehicle_candidates(
    world,
    airsim_client,
    target_actor,
    config: RecoverVLAConfig,
    max_candidates: int = 12,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Project visible CARLA vehicles into the UAV camera.

    The target actor id is retained only for logging/evaluation. Scoring code
    must not use is_target as an input feature.
    """

    candidates: List[Dict[str, Any]] = []
    target_bbox: Dict[str, Any] = {"visible": False, "bbox": None}
    target_id = None
    if target_actor is not None:
        try:
            target_id = int(target_actor.id)
            target_bbox = project_target_bbox_to_image(airsim_client, target_actor, config)
        except Exception:
            target_bbox = {"visible": False, "bbox": None}

    try:
        actors = list(world.get_actors().filter("vehicle.*"))
    except Exception:
        actors = [target_actor] if target_actor is not None else []

    for actor in actors:
        if actor is None:
            continue
        try:
            bbox_meta = project_target_bbox_to_image(airsim_client, actor, config)
            if not bbox_meta.get("visible") or bbox_meta.get("bbox") is None:
                continue
            state = get_target_state(actor)
            bbox = tuple(bbox_meta["bbox"])
            area = max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])
            candidates.append(
                {
                    "candidate_id": int(actor.id),
                    "type_id": str(getattr(actor, "type_id", "vehicle.unknown")),
                    "bbox": bbox,
                    "bbox_area": area,
                    "visible_points": int(bbox_meta.get("visible_points") or 0),
                    "state": state,
                    "is_target": target_id is not None and int(actor.id) == target_id,
                }
            )
        except Exception:
            continue

    candidates.sort(key=lambda item: (item["bbox_area"], item["visible_points"]), reverse=True)
    return target_bbox, candidates[:max_candidates]


def visible_target_observation(target_actor, target_bbox: Dict[str, Any]) -> Optional[TargetState]:
    """Return target state only when the target is visually observable."""

    if target_actor is None or not target_bbox.get("visible"):
        return None
    try:
        observed = get_target_state(target_actor)
        observed.source = "visible_oracle_projection"
        return observed
    except Exception:
        return None


def update_target_memory(
    memory: TargetMemory,
    observed: Optional[TargetState],
    assessment: Optional[CandidateAssessment],
    now: Optional[float] = None,
    update_threshold: float = 0.70,
) -> bool:
    """Update memory only after visible, high-confidence confirmation."""

    now = time.time() if now is None else now
    confirmed = (
        observed is not None
        and assessment is not None
        and assessment.confidence >= update_threshold
        and (assessment.target_found or assessment.bbox is not None)
    )
    if confirmed:
        memory.previous_seen = memory.last_seen
        memory.last_seen = observed
        memory.lost_since = None
        memory.observations.append(observed)
        if len(memory.observations) > 64:
            memory.observations = memory.observations[-64:]
        return True

    if memory.last_seen is not None and memory.lost_since is None:
        memory.lost_since = now
    return False
