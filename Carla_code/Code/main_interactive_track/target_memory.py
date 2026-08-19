"""Target memory for dynamic target re-acquisition."""

from __future__ import annotations

import time
from typing import Optional

from .models import TargetMemory, TargetState


class TargetMemoryBank:
    def __init__(self, memory: TargetMemory):
        self.memory = memory

    def observe(self, state: Optional[TargetState], now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        if state is None:
            if self.memory.last_seen is not None and self.memory.lost_since is None:
                self.memory.lost_since = now
            return
        self.memory.previous_seen = self.memory.last_seen
        self.memory.last_seen = state
        self.memory.lost_since = None
        self.memory.observations.append(state)
        if len(self.memory.observations) > 40:
            self.memory.observations = self.memory.observations[-40:]

    @property
    def is_lost(self) -> bool:
        return self.memory.lost_since is not None

    def lost_time(self, now: Optional[float] = None) -> float:
        now = time.time() if now is None else now
        if self.memory.lost_since is None:
            return 0.0
        return max(0.0, now - self.memory.lost_since)

    def prompt_dict(self, now: Optional[float] = None):
        return self.memory.to_prompt_dict(time.time() if now is None else now)
