"""Candidate verification for reducing false re-acquisition."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from .models import BeliefRegion, CandidateAssessment, TargetMemory, TargetState
from .utils import clamp, distance2


class CandidateVerifier:
    """Score a visible candidate using paper-friendly interpretable terms."""

    def __init__(self, confirm_threshold: float = 0.72, candidate_margin: float = 0.06):
        self.confirm_threshold = confirm_threshold
        self.candidate_margin = candidate_margin

    def score(
        self,
        bbox_meta: Optional[Dict[str, Any]],
        observed: Optional[TargetState],
        memory: TargetMemory,
        belief: Optional[BeliefRegion],
        candidates: Optional[List[Dict[str, Any]]] = None,
    ) -> CandidateAssessment:
        if candidates:
            return self.score_candidates(candidates, memory, belief)

        bbox_meta = bbox_meta or {}
        bbox = bbox_meta.get("bbox")
        visible = bool(bbox_meta.get("visible")) and bbox is not None
        visual = 1.0 if visible else 0.0

        language = 0.55
        if memory.color:
            language += 0.18
        if memory.vehicle_type:
            language += 0.18
        language = clamp(language, 0.0, 1.0)

        motion = 0.5
        if observed is not None and memory.last_seen is not None:
            heading_delta = abs((observed.heading - memory.last_seen.heading + 180.0) % 360.0 - 180.0)
            speed_delta = abs(observed.speed_xy - memory.last_seen.speed_xy)
            motion = clamp(1.0 - heading_delta / 90.0 - speed_delta / 18.0, 0.0, 1.0)

        spatial = 0.45
        if observed is not None and belief is not None:
            d = distance2(observed.position, belief.center)
            spatial = clamp(1.0 - d / max(1.0, belief.radius_m), 0.0, 1.0)

        reid = 0.5
        score = 0.35 * visual + 0.25 * language + 0.20 * motion + 0.10 * spatial + 0.10 * reid
        target_found = score >= self.confirm_threshold and visible
        reason = (
            f"score={score:.2f}; visual={visual:.2f}, language={language:.2f}, "
            f"motion={motion:.2f}, spatial={spatial:.2f}, reid={reid:.2f}"
        )
        return CandidateAssessment(
            target_found=target_found,
            confidence=clamp(score, 0.0, 1.0),
            score_terms={"visual": visual, "language": language, "motion": motion, "spatial": spatial, "reid": reid},
            bbox=tuple(bbox) if bbox is not None else None,
            reason=reason,
        )

    def score_candidates(
        self,
        candidates: List[Dict[str, Any]],
        memory: TargetMemory,
        belief: Optional[BeliefRegion],
    ) -> CandidateAssessment:
        scored = [self._score_one(candidate, memory, belief) for candidate in candidates]
        scored.sort(key=lambda item: item["score"], reverse=True)
        if not scored:
            return CandidateAssessment(False, 0.0, {"visual": 0.0, "language": 0.0, "motion": 0.0, "spatial": 0.0, "reid": 0.0}, reason="no_visible_vehicle_candidates")

        best = scored[0]
        second = scored[1]["score"] if len(scored) > 1 else 0.0
        margin = best["score"] - second
        target_found = best["score"] >= self.confirm_threshold and margin >= self.candidate_margin
        compact = [
            {
                "candidate_id": item["candidate_id"],
                "type_id": item["candidate_type"],
                "bbox": item["bbox"],
                "confidence": round(item["score"], 3),
                "score_terms": item["terms"],
                "is_target_eval_only": item.get("is_target_eval_only"),
            }
            for item in scored[:6]
        ]
        reason = (
            f"best={best['score']:.2f}, margin={margin:.2f}; "
            f"visual={best['terms']['visual']:.2f}, language={best['terms']['language']:.2f}, "
            f"motion={best['terms']['motion']:.2f}, spatial={best['terms']['spatial']:.2f}, "
            f"reid={best['terms']['reid']:.2f}"
        )
        return CandidateAssessment(
            target_found=target_found,
            confidence=clamp(best["score"], 0.0, 1.0),
            score_terms=best["terms"],
            bbox=best["bbox"],
            candidate_id=best["candidate_id"],
            candidate_type=best["candidate_type"],
            candidates=compact,
            reason=reason,
        )

    def _score_one(self, candidate: Dict[str, Any], memory: TargetMemory, belief: Optional[BeliefRegion]) -> Dict[str, Any]:
        bbox = tuple(candidate.get("bbox")) if candidate.get("bbox") is not None else None
        area = float(candidate.get("bbox_area") or 0.0)
        visual = clamp(0.35 + min(area, 18000.0) / 18000.0 * 0.65, 0.0, 1.0) if bbox else 0.0

        type_id = str(candidate.get("type_id") or "")
        language = 0.42
        if memory.vehicle_type:
            type_match = self._vehicle_type_match(type_id, memory.vehicle_type)
            language += 0.36 if type_match else -0.12
        if memory.color:
            # CARLA actor color is not always readable after spawn, so keep this
            # as a weak prior unless an external detector/ReID later provides it.
            language += 0.08
        language = clamp(language, 0.0, 1.0)

        state = candidate.get("state")
        motion = 0.45
        if isinstance(state, TargetState) and memory.last_seen is not None:
            heading_delta = abs((state.heading - memory.last_seen.heading + 180.0) % 360.0 - 180.0)
            speed_delta = abs(state.speed_xy - memory.last_seen.speed_xy)
            motion = clamp(1.0 - heading_delta / 110.0 - speed_delta / 22.0, 0.0, 1.0)

        spatial = 0.40
        if isinstance(state, TargetState) and belief is not None:
            d = distance2(state.position, belief.center)
            spatial = clamp(1.0 - d / max(1.0, belief.radius_m), 0.0, 1.0)

        reid = 0.50
        score = 0.35 * visual + 0.25 * language + 0.20 * motion + 0.10 * spatial + 0.10 * reid
        return {
            "candidate_id": int(candidate.get("candidate_id", -1)),
            "candidate_type": type_id,
            "bbox": bbox,
            "score": clamp(score, 0.0, 1.0),
            "terms": {"visual": visual, "language": language, "motion": motion, "spatial": spatial, "reid": reid},
            "is_target_eval_only": bool(candidate.get("is_target")),
        }

    @staticmethod
    def _vehicle_type_match(type_id: str, vehicle_type: str) -> bool:
        type_id = type_id.lower()
        if vehicle_type == "truck":
            return any(token in type_id for token in ("truck", "carlacola", "firetruck", "sprinter", "cybertruck"))
        if vehicle_type == "bus":
            return "bus" in type_id or "sprinter" in type_id
        if vehicle_type == "suv":
            return any(token in type_id for token in ("jeep", "patrol", "cybertruck", "suv"))
        if vehicle_type == "sedan":
            return any(token in type_id for token in ("tesla", "audi", "bmw", "mercedes", "lincoln", "dodge"))
        return True
