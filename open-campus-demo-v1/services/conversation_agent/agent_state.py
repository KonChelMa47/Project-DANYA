"""エージェント内部状態管理。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from schemas import GlobalSceneInfo


@dataclass
class AgentState:
    known_visitor_ids: set[str] = field(default_factory=set)
    last_global_vlm_time: float = 0.0
    last_speech_time: float = 0.0
    speech_busy_until: float = 0.0
    speech_count: int = 0
    # 本番ループの 通常→特別→通常→日常 サイクルで直前と同じ特別種を避ける
    last_cycle_special_kind: Optional[str] = None

    recent_speeches: list[str] = field(default_factory=list)
    recent_topics: list[str] = field(default_factory=list)
    visitor_topic_memory: dict[str, list[str]] = field(default_factory=dict)
    visitor_story_steps: dict[str, int] = field(default_factory=dict)
    audience_story_steps: dict[str, int] = field(default_factory=dict)
    visitor_last_seen: dict[str, float] = field(default_factory=dict)
    visitor_last_spoken: dict[str, float] = field(default_factory=dict)
    visitor_audience_memory: dict[str, str] = field(default_factory=dict)
    visitor_opening_memory: dict[str, list[str]] = field(default_factory=dict)
    visitor_profiles: dict[str, dict] = field(default_factory=dict)
    already_explained_topics: set[str] = field(default_factory=set)
    recent_openings: list[str] = field(default_factory=list)
    recent_emotion_flows: list[str] = field(default_factory=list)

    no_people_since: float = 0.0
    next_idle_short_time: float = 0.0
    next_idle_long_time: float = 0.0

    current_mode: str = "idle"
    current_strategy: str = ""
    last_processed_event_id: Optional[str] = None
    last_processed_timestamp: float = 0.0
    last_global_scene_info: GlobalSceneInfo = field(default_factory=GlobalSceneInfo)
    last_camera_scene_info: dict = field(default_factory=dict)
    dynamic_findings: list[str] = field(default_factory=list)

    def detect_new_visitors(self, active_visitor_ids: set[str]) -> set[str]:
        new_ids = active_visitor_ids - self.known_visitor_ids
        if new_ids:
            self.known_visitor_ids.update(new_ids)
        return new_ids

    def can_speak(self, now: float, interval_sec: float) -> tuple[bool, str]:
        if now < self.speech_busy_until:
            return False, "speech_busy_until待機中"
        if (now - self.last_speech_time) < interval_sec:
            return False, "発話間隔待機中"
        return True, "発話可能"

    def set_speech_busy(self, now: float, busy_sec: float) -> None:
        self.last_speech_time = now
        self.speech_busy_until = now + max(0.0, busy_sec)

    def is_duplicate_speech(self, speech: str) -> bool:
        return bool(self.recent_speeches and self.recent_speeches[-1] == speech)

    def remember_visitor_topic(self, visitor_id: Optional[str], topic: str) -> None:
        if visitor_id:
            topics = self.visitor_topic_memory.setdefault(visitor_id, [])
            topics.append(topic)
            if len(topics) > 12:
                del topics[:-12]

    def update_recents(self, speech: str, topic: str, max_speeches: int, max_topics: int) -> None:
        self.speech_count += 1
        self.recent_speeches.append(speech)
        self.recent_topics.append(topic)
        if len(self.recent_speeches) > max_speeches:
            self.recent_speeches.pop(0)
        if len(self.recent_topics) > max_topics:
            self.recent_topics.pop(0)

    def remember_opening(self, visitor_id: Optional[str], opening: str, max_items: int = 12) -> None:
        if not opening:
            return
        self.recent_openings.append(opening)
        if len(self.recent_openings) > max_items:
            del self.recent_openings[:-max_items]
        if visitor_id:
            openings = self.visitor_opening_memory.setdefault(visitor_id, [])
            openings.append(opening)
            if len(openings) > max_items:
                del openings[:-max_items]

    def remember_emotion_flow(self, flow: list[str], max_items: int = 10) -> None:
        if not flow:
            return
        key = ">".join(flow)
        self.recent_emotion_flows.append(key)
        if len(self.recent_emotion_flows) > max_items:
            del self.recent_emotion_flows[:-max_items]

    def recent_visitor_topics(self, visitor_id: Optional[str]) -> list[str]:
        if not visitor_id:
            return []
        return self.visitor_topic_memory.get(visitor_id, [])[-6:]

    def story_step_for(self, visitor_id: Optional[str], audience_type: str) -> int:
        if visitor_id and visitor_id in self.visitor_story_steps:
            return self.visitor_story_steps[visitor_id]
        return self.audience_story_steps.get(audience_type, 0)

    def remember_story_step(self, visitor_id: Optional[str], audience_type: str, step: int) -> None:
        next_step = max(0, step + 1)
        if visitor_id:
            self.visitor_story_steps[visitor_id] = next_step
        self.audience_story_steps[audience_type] = next_step

    def update_idle_presence(self, now: float, people_count: int) -> None:
        if people_count <= 0:
            if self.no_people_since <= 0:
                self.no_people_since = now
        else:
            self.no_people_since = 0.0

    def to_debug_dict(self) -> dict:
        return {
            "known_visitor_ids": sorted(self.known_visitor_ids),
            "last_global_vlm_time": self.last_global_vlm_time,
            "last_speech_time": self.last_speech_time,
            "speech_busy_until": self.speech_busy_until,
            "recent_speeches_count": len(self.recent_speeches),
            "speech_count": self.speech_count,
            "last_cycle_special_kind": self.last_cycle_special_kind,
            "recent_topics": self.recent_topics[-5:],
            "audience_story_steps": dict(self.audience_story_steps),
            "recent_openings": self.recent_openings[-5:],
            "recent_emotion_flows": self.recent_emotion_flows[-3:],
            "current_mode": self.current_mode,
            "current_strategy": self.current_strategy,
            "last_processed_event_id": self.last_processed_event_id,
            "last_processed_timestamp": self.last_processed_timestamp,
        }

