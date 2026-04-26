"""ルールベース戦略計画。後でLLM差し替え可能。"""

from __future__ import annotations

import random

import config
from agent_state import AgentState
from llm_client import generate_strategy_with_llm
from schemas import SituationInfo, StrategyPlan, VisitorInfo


class StrategyPlanner:
    DEFAULT_TOPICS = {
        "idle": ["呼び込み", "ダーニャ自己紹介"],
        "hook": ["興味喚起", "展示の仕掛け予告"],
        "intro": ["展示概要", "追跡と発話の連動"],
        "deepen": ["仕組みの深掘り", "自己改善ループ"],
        "returning": ["前回と違う見方", "続きの話題"],
        "quiz": ["二択クイズ", "仕組み当てクイズ"],
        "crowd": ["全体向け要点", "短時間まとめ"],
        "closing": ["短い締め", "次の見どころ予告"],
    }
    EMOTION = {
        "idle": "<happy_normal>",
        "hook": "<surprised_normal>",
        "intro": "<happy_normal>",
        "deepen": "<surprised_high>",
        "returning": "<happy_high>",
        "quiz": "<fear_normal>",
        "crowd": "<happy_high>",
        "closing": "<sad_normal>",
    }

    def _pick_topic(self, mode: str, state: AgentState) -> str:
        candidates = self.DEFAULT_TOPICS.get(mode, ["展示案内"])
        for c in candidates:
            if c not in state.recent_topics[-3:]:
                return c
        return random.choice(candidates)

    def plan(self, situation: SituationInfo, visitors: list[VisitorInfo], rag_results: list[dict], state: AgentState) -> StrategyPlan:
        topic = self._pick_topic(situation.mode, state)
        if rag_results:
            topic = f"{topic}（RAG参照）"
        if situation.mode == "returning" and situation.target_visitor_id:
            last_topics = state.visitor_topic_memory.get(situation.target_visitor_id, [])
            if topic in last_topics[-3:]:
                topic = "前回と別角度の紹介"
        summary = f"{situation.reason} / topic={topic}"
        speech_intent = {
            "idle": "呼び込み",
            "closing": "短く引き止める",
            "quiz": "反応を引き出す",
            "crowd": "全体向け要点共有",
        }.get(situation.mode, "展示案内")
        base = StrategyPlan(
            mode=situation.mode,
            target_visitor_id=situation.target_visitor_id,
            topic=topic,
            strategy_summary=summary,
            avoid_topics=list(set(state.recent_topics[-3:])),
            recommended_emotion=self.EMOTION.get(situation.mode, "<happy_normal>"),
            speech_intent=speech_intent,
            priority=0.8 if situation.mode in ("closing", "deepen") else 0.6,
        )
        if not config.use_llm:
            return base

        target = visitors[0] if visitors else None
        context = {
            "mode": situation.mode,
            "target_visitor_id": situation.target_visitor_id,
            "people_count": situation.people_count,
            "target_dwell_sec": target.dwell_time_sec if target else 0.0,
            "returning": target.returning if target else False,
            "expression": target.vlm.expression if target else "unknown",
            "age_estimate": target.vlm.age_estimate if target else "unknown",
            "clothing_description": target.vlm.clothing_description if target else "unknown",
            "scene_summary": state.last_global_scene_info.scene_summary,
            "attention_target": state.last_global_scene_info.attention_target,
            "movement_state": state.last_global_scene_info.movement_state,
            "leaving_risk": state.last_global_scene_info.leaving_risk,
            "rag_results": rag_results[:5],
            "recent_topics": state.recent_topics[-5:],
            "recent_speeches": state.recent_speeches[-3:],
            "visitor_topic_memory": state.visitor_topic_memory,
            "character": "金沢弁タメ口、高校生優先、引き止め優先",
            "forbidden": "金沢観光や歴史文化の一般紹介はしない",
        }
        llm_data = generate_strategy_with_llm(context)
        if not llm_data:
            return base
        try:
            return StrategyPlan(**{**base.model_dump(), **llm_data})
        except Exception:
            return base

