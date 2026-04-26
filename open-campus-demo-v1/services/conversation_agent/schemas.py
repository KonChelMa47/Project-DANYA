"""Pydanticスキーマ定義。壊れた入力にも極力耐える。"""

from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, field_validator

EmotionTag = Literal[
    "<happy_high>", "<happy_normal>", "<angry_high>", "<angry_normal>",
    "<sad_high>", "<sad_normal>", "<fear_high>", "<fear_normal>",
    "<surprised_high>", "<surprised_normal>",
]
EmotionName = Literal[
    "happy_high", "happy_normal", "angry_high", "angry_normal",
    "sad_high", "sad_normal", "fear_high", "fear_normal",
    "surprised_high", "surprised_normal",
]
ModeType = Literal["idle", "hook", "intro", "deepen", "returning", "quiz", "crowd", "closing"]
AudienceType = Literal[
    "high_school_student", "parent_or_adult", "child", "group", "general", "unknown"
]

EMOTION_NAMES = {
    "happy_high", "happy_normal", "angry_high", "angry_normal",
    "sad_high", "sad_normal", "fear_high", "fear_normal",
    "surprised_high", "surprised_normal",
}


def emotion_name_to_tag(emotion: str) -> str:
    name = emotion if emotion in EMOTION_NAMES else "happy_normal"
    return f"<{name}>"


def emotion_tag_to_name(tag: str) -> str:
    cleaned = str(tag or "").strip().strip("<>")
    return cleaned if cleaned in EMOTION_NAMES else "happy_normal"


def _safe_float(v: Any, default: float = 0.0) -> float:
    if v in (None, "", "unknown", "UNKNOWN"):
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    if v in (None, "", "unknown", "UNKNOWN"):
        return default
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


class VLMInfo(BaseModel):
    gender: str = "unknown"
    age_estimate: str = "unknown"
    clothing_description: str = "unknown"
    expression: str = "unknown"
    pose_description: str = "unknown"
    accessories: list[str] = Field(default_factory=list)
    carried_items: list[str] = Field(default_factory=list)
    expression_confidence: float = 0.0

    @field_validator("expression_confidence", mode="before")
    @classmethod
    def _v_expr(cls, v: Any) -> float:
        return _safe_float(v, 0.0)


class VisitorInfo(BaseModel):
    visitor_id: str = "unknown"
    dwell_time_sec: float = 0.0
    bbox_width_px: int = 0
    returning: bool = False
    visit_count: int = 1
    same_person_confidence: float = 0.0
    vlm: VLMInfo = Field(default_factory=VLMInfo)

    @field_validator("dwell_time_sec", mode="before")
    @classmethod
    def _v_dwell(cls, v: Any) -> float:
        return _safe_float(v, 0.0)

    @field_validator("bbox_width_px", mode="before")
    @classmethod
    def _v_bbox(cls, v: Any) -> int:
        return _safe_int(v, 0)

    @field_validator("visit_count", mode="before")
    @classmethod
    def _v_visit_count(cls, v: Any) -> int:
        return max(1, _safe_int(v, 1))

    @field_validator("same_person_confidence", mode="before")
    @classmethod
    def _v_same_person_confidence(cls, v: Any) -> float:
        return _safe_float(v, 0.0)

    @classmethod
    def from_person_info(cls, payload: dict) -> "VisitorInfo":
        return cls(
            visitor_id=str(payload.get("visitor_id", "unknown")),
            dwell_time_sec=payload.get("dwell_time_sec", 0.0),
            bbox_width_px=payload.get("bbox_width_px", 0),
            returning=bool(payload.get("was_lost_and_returned", False)),
            visit_count=payload.get("visit_count", 1),
            same_person_confidence=payload.get("same_person_confidence", 0.0),
            vlm=VLMInfo(
                gender=payload.get("gender", "unknown"),
                age_estimate=payload.get("age_estimate", "unknown"),
                clothing_description=payload.get("clothing_description", "unknown"),
                expression=payload.get("expression", "unknown"),
                pose_description=payload.get("pose_description", "unknown"),
                accessories=payload.get("accessories", []),
                carried_items=payload.get("carried_items", []),
                expression_confidence=payload.get("expression_confidence", 0.0),
            ),
        )


class PersonInfoEvent(BaseModel):
    event_type: Literal["person_info_updated"] = "person_info_updated"
    timestamp: float = 0.0
    event_id: Optional[str] = None
    person_info: dict = Field(default_factory=dict)

    @field_validator("timestamp", mode="before")
    @classmethod
    def _v_ts(cls, v: Any) -> float:
        return _safe_float(v, 0.0)


class EnvironmentInfoEvent(BaseModel):
    event_type: Literal["environment_info_updated"] = "environment_info_updated"
    timestamp: float = 0.0
    event_id: Optional[str] = None
    environment_info: dict = Field(default_factory=dict)
    new_visitor_ids: list[str] = Field(default_factory=list)
    returning_visitor_ids: list[str] = Field(default_factory=list)

    @field_validator("timestamp", mode="before")
    @classmethod
    def _v_ts(cls, v: Any) -> float:
        return _safe_float(v, 0.0)


class CameraSceneEvent(BaseModel):
    event_type: Literal["camera_scene_updated"] = "camera_scene_updated"
    timestamp: float = 0.0
    event_id: Optional[str] = None
    scene_info: dict = Field(default_factory=dict)

    @field_validator("timestamp", mode="before")
    @classmethod
    def _v_ts(cls, v: Any) -> float:
        return _safe_float(v, 0.0)


class GlobalSceneInfo(BaseModel):
    scene_summary: str = "unknown"
    people_flow: Literal["increasing", "stable", "decreasing", "unknown"] = "unknown"
    attention_target: Literal["danya", "nanamaru", "display", "passing", "unknown"] = "unknown"
    crowd_state: Literal["none", "single", "small_group", "crowd", "unknown"] = "unknown"
    movement_state: Literal["passing", "stopped", "approaching", "leaving", "mixed", "unknown"] = "unknown"
    engagement_estimate: float = 0.0
    confusion_estimate: float = 0.0
    leaving_risk: float = 0.0
    notable_event: str = ""
    recommended_interaction: Literal["hook", "intro", "deepen", "returning", "quiz", "crowd", "closing", "idle"] = "idle"

    @field_validator("engagement_estimate", "confusion_estimate", "leaving_risk", mode="before")
    @classmethod
    def _vf(cls, v: Any) -> float:
        return _safe_float(v, 0.0)


class SituationInfo(BaseModel):
    mode: ModeType
    reason: str = ""
    target_visitor_id: Optional[str] = None
    people_count: int = 0


class StrategyPlan(BaseModel):
    mode: ModeType
    target_visitor_id: Optional[str] = None
    topic: str = "展示案内"
    strategy_summary: str = ""
    avoid_topics: list[str] = Field(default_factory=list)
    recommended_emotion: EmotionTag = "<happy_normal>"
    speech_intent: str = "案内"
    priority: float = 0.5

    @field_validator("priority", mode="before")
    @classmethod
    def _vp(cls, v: Any) -> float:
        return _safe_float(v, 0.5)


class SpeechSegment(BaseModel):
    emotion: EmotionName = "happy_normal"
    text: str = ""


class TopicPlan(BaseModel):
    mode: ModeType
    target_visitor_id: Optional[str] = None
    people_count: int = 0
    audience_type: AudienceType = "general"
    audience_label: str = "来場者"
    primary_topic: str = "ダーニャ自身"
    topics: list[str] = Field(default_factory=list)
    knowledge_points: list[str] = Field(default_factory=list)
    story_phase: str = ""
    story_arc: list[str] = Field(default_factory=list)
    story_step: int = 0
    previous_story_topic: str = ""
    next_story_topic: str = ""
    intent: str = "展示MCとして自分から話しかける"
    visual_hook: str = ""
    vlm_observations: list[str] = Field(default_factory=list)
    vlm_humor: str = ""
    vlm_confidence_note: str = ""
    use_visual_detail: bool = True
    scene_note: str = ""
    depth_level: int = 1
    is_returning: bool = False
    avoid_topics: list[str] = Field(default_factory=list)
    avoid_openings: list[str] = Field(default_factory=list)


class EmotionPlan(BaseModel):
    flow: list[EmotionName] = Field(default_factory=lambda: ["surprised_high", "happy_high", "sad_normal", "happy_normal"])
    style_note: str = "感情デモとして、楽しく大げさに切り替える"


class SpeechOutput(BaseModel):
    mode: ModeType
    target_visitor_id: Optional[str] = None
    emotion_tag: EmotionTag = "<happy_normal>"
    speech: str = ""
    segments: list[SpeechSegment] = Field(default_factory=list)
    topic: str
    strategy_summary: str
    priority: float = 0.5

    @classmethod
    def from_segments(
        cls,
        *,
        mode: ModeType,
        target_visitor_id: Optional[str],
        segments: list[SpeechSegment],
        topic: str,
        strategy_summary: str,
        priority: float = 0.5,
    ) -> "SpeechOutput":
        safe_segments = segments or [SpeechSegment(emotion="happy_normal", text="ダーニャ、ここにおるよ。")]
        speech = "\n".join(
            f"{emotion_name_to_tag(seg.emotion)}{seg.text.strip()}"
            for seg in safe_segments
            if seg.text.strip()
        )
        first_tag = emotion_name_to_tag(safe_segments[0].emotion)
        return cls(
            mode=mode,
            target_visitor_id=target_visitor_id,
            emotion_tag=first_tag,
            speech=speech,
            segments=safe_segments,
            topic=topic,
            strategy_summary=strategy_summary,
            priority=priority,
        )


class EvaluationResult(BaseModel):
    result: Literal["success", "neutral", "failure"] = "neutral"
    engagement_delta: float = 0.0
    reason: str = ""


class StrategyLogEvent(BaseModel):
    timestamp_iso: str
    strategy: StrategyPlan
    speech: SpeechOutput
    situation: SituationInfo
    scene: GlobalSceneInfo
    evaluation: EvaluationResult

