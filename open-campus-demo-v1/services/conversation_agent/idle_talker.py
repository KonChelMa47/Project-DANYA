"""無人時の短い独り言・長い独り言のタイミング管理。"""

from __future__ import annotations

import random

import config
from agent_state import AgentState


class IdleTalker:
    def ensure_schedule(self, now: float, state: AgentState) -> None:
        if state.next_idle_short_time <= 0:
            state.next_idle_short_time = now + random.uniform(
                config.idle_short_min_sec, config.idle_short_max_sec
            )
        if state.next_idle_long_time <= 0:
            state.next_idle_long_time = now + config.idle_long_interval_sec

    def due_kind(self, now: float, people_count: int, state: AgentState) -> str | None:
        if people_count > 0:
            state.next_idle_short_time = 0.0
            state.next_idle_long_time = 0.0
            return None
        self.ensure_schedule(now, state)
        if now >= state.next_idle_long_time:
            return "long"
        if now >= state.next_idle_short_time:
            return "short"
        return None

    def mark_spoken(self, now: float, state: AgentState, kind: str) -> None:
        if kind == "long":
            state.next_idle_long_time = now + config.idle_long_interval_sec
            state.next_idle_short_time = now + random.uniform(
                config.idle_short_min_sec, config.idle_short_max_sec
            )
            return
        state.next_idle_short_time = now + random.uniform(
            config.idle_short_min_sec, config.idle_short_max_sec
        )
