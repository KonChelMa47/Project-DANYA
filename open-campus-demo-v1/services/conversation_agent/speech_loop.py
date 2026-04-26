"""発話ループ制御。"""

from __future__ import annotations

import config
from agent_state import AgentState


class SpeechLoop:
    @staticmethod
    def estimate_busy_sec(speech: str) -> float:
        sec = (len(speech) * config.speech_char_sec + config.speech_fixed_overhead_sec) * config.speech_wait_scale
        sec = max(config.min_speech_busy_sec, sec)
        if config.speech_wait_max_sec > 0:
            sec = min(config.speech_wait_max_sec, sec)
        return sec

    def can_emit(self, now: float, people_count: int, state: AgentState, debug: bool = False) -> tuple[bool, str, float]:
        interval = config.active_speech_interval_sec if people_count > 0 else config.idle_speech_interval_sec
        can, reason = state.can_speak(now, interval)
        if debug and not can:
            print(f"[speech_loop] skip: {reason}")
        return can, reason, interval

    def on_spoken(
        self,
        now: float,
        speech: str,
        topic: str,
        target_visitor_id: str | None,
        state: AgentState,
        *,
        busy_sec: float | None = None,
    ) -> None:
        busy = self.estimate_busy_sec(speech) if busy_sec is None else busy_sec
        state.set_speech_busy(now, busy)
        state.update_recents(speech, topic, config.max_recent_speeches, config.max_recent_topics)
        state.remember_visitor_topic(target_visitor_id, topic)

