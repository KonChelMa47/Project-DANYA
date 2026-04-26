"""状況分析: 8モード選択。"""

from __future__ import annotations

import config
from agent_state import AgentState
from schemas import GlobalSceneInfo, SituationInfo, VisitorInfo


class SituationAnalyzer:
    def analyze(
        self,
        visitors: list[VisitorInfo],
        people_count: int,
        global_scene: GlobalSceneInfo,
        state: AgentState,
        new_visitor_ids: set[str],
    ) -> SituationInfo:
        if people_count <= 0:
            return SituationInfo(mode="idle", reason="人がいない", target_visitor_id=None, people_count=0)
        if not visitors:
            return SituationInfo(mode="hook", reason="人物は見えるが画像解析詳細待ち", target_visitor_id=None, people_count=people_count)

        target = max(visitors, key=lambda v: v.dwell_time_sec)

        if global_scene.leaving_risk >= config.leaving_risk_threshold:
            return SituationInfo(mode="closing", reason="leaving_risk高", target_visitor_id=target.visitor_id, people_count=people_count)
        if target.returning:
            return SituationInfo(mode="returning", reason="returning visitor", target_visitor_id=target.visitor_id, people_count=people_count)
        if people_count >= config.people_count_crowd_threshold or global_scene.crowd_state == "crowd":
            return SituationInfo(mode="crowd", reason="人数が多い", target_visitor_id=target.visitor_id, people_count=people_count)
        if target.dwell_time_sec >= config.deepen_dwell_sec:
            return SituationInfo(mode="deepen", reason="30秒以上滞在", target_visitor_id=target.visitor_id, people_count=people_count)
        if target.dwell_time_sec >= config.intro_dwell_sec:
            return SituationInfo(mode="intro", reason="10秒以上滞在", target_visitor_id=target.visitor_id, people_count=people_count)
        if new_visitor_ids:
            return SituationInfo(mode="hook", reason="新規visitor検出", target_visitor_id=target.visitor_id, people_count=people_count)
        if len(state.recent_topics) >= 3 and len(set(state.recent_topics[-3:])) == 1:
            return SituationInfo(mode="quiz", reason="話題が単調", target_visitor_id=target.visitor_id, people_count=people_count)
        fallback = "intro" if target.dwell_time_sec >= 5 else "hook"
        return SituationInfo(mode=fallback, reason="通常遷移", target_visitor_id=target.visitor_id, people_count=people_count)

