"""Shared data structures for RecoverVLA."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

Vec3 = Tuple[float, float, float]
BBox = Tuple[int, int, int, int]


@dataclass
class DroneState:
    position: Vec3
    velocity: Vec3
    heading: float
    battery: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TargetState:
    position: Vec3
    velocity: Vec3
    heading: float
    speed_xy: float
    timestamp: float
    source: str = "observed"
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TargetMemory:
    raw_instruction: str
    color: Optional[str] = None
    vehicle_type: Optional[str] = None
    appearance: str = ""
    last_seen: Optional[TargetState] = None
    previous_seen: Optional[TargetState] = None
    lost_since: Optional[float] = None
    observations: List[TargetState] = field(default_factory=list)

    def to_prompt_dict(self, now: float) -> Dict[str, Any]:
        lost_time = None if self.lost_since is None else max(0.0, now - self.lost_since)
        return {
            "target_color": self.color,
            "target_type": self.vehicle_type,
            "appearance": self.appearance,
            "last_seen": self.last_seen.to_dict() if self.last_seen else None,
            "lost_time_seconds": lost_time,
        }


@dataclass
class BeliefRegion:
    center: Vec3
    radius_m: float
    heading: float
    confidence: float
    reason: str
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    safety: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateAssessment:
    target_found: bool
    confidence: float
    score_terms: Dict[str, float]
    bbox: Optional[BBox] = None
    candidate_id: Optional[int] = None
    candidate_type: Optional[str] = None
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DecisionPlan:
    phase: str
    view: str
    target_found: bool
    confidence: float
    waypoint: Vec3
    desired_yaw: float
    hold_seconds: float
    reason: str
    search_region: Optional[BeliefRegion] = None
    candidate: Optional[CandidateAssessment] = None
    raw_model: Dict[str, Any] = field(default_factory=dict)
    source: str = "local"

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data


@dataclass
class ControlAction:
    vx: float
    vy: float
    vz: float
    yaw_rate: float
    duration: float
    distance_3d: float
    clipped: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
