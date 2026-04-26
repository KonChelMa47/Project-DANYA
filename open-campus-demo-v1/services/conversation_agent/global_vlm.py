"""global VLMインターフェース。現在はモック実装。"""

from __future__ import annotations

import os
import random
import time

import config
from agent_state import AgentState
from schemas import GlobalSceneInfo


def capture_global_frame():
    """将来の全体画像取得口。現在は未実装。"""
    return None


def mock_global_vlm(people_count: int, longest_dwell_sec: float, has_new_visitor: bool, returning: bool) -> GlobalSceneInfo:
    crowd_state = "none"
    if people_count == 1:
        crowd_state = "single"
    elif 2 <= people_count <= 3:
        crowd_state = "small_group"
    elif people_count >= 4:
        crowd_state = "crowd"

    if people_count == 0:
        return GlobalSceneInfo(
            scene_summary="来場者は見えない",
            people_flow="decreasing",
            attention_target="unknown",
            crowd_state="none",
            movement_state="passing",
            engagement_estimate=0.0,
            confusion_estimate=0.0,
            leaving_risk=0.1,
            notable_event="無人状態",
            recommended_interaction="idle",
        )

    recommended = "hook" if has_new_visitor else "intro"
    if people_count >= 4:
        recommended = "crowd"
    if longest_dwell_sec >= 30:
        recommended = "deepen"
    if returning:
        recommended = "returning"

    leaving = 0.25 if people_count > 0 else 0.1
    if longest_dwell_sec < 6 and not has_new_visitor:
        leaving = 0.75
    return GlobalSceneInfo(
        scene_summary=f"来場者{people_count}名、滞在最長{longest_dwell_sec:.1f}秒",
        people_flow="stable",
        attention_target="danya",
        crowd_state=crowd_state,
        movement_state="stopped" if longest_dwell_sec >= 3 else "passing",
        engagement_estimate=min(1.0, 0.2 + longest_dwell_sec / 40.0),
        confusion_estimate=0.2,
        leaving_risk=leaving,
        notable_event="新規来場あり" if has_new_visitor else "滞在継続",
        recommended_interaction=recommended,
    )


class GlobalVLMService:
    def __init__(self, use_openai: bool = False, timeout_sec: int = 30, debug: bool = False) -> None:
        self.use_openai = use_openai
        self.timeout_sec = timeout_sec
        self.debug = debug

    def should_run(self, now: float, has_new_visitor: bool, state: AgentState) -> bool:
        if has_new_visitor:
            return True
        return (now - state.last_global_vlm_time) >= self.timeout_sec

    def run(self, state: AgentState, people_count: int = 0, longest_dwell_sec: float = 0.0, has_new_visitor: bool = False, returning: bool = False) -> GlobalSceneInfo:
        if self.use_openai and os.getenv("OPENAI_API_KEY"):
            if self.debug:
                print("[global_vlm] OPENAI要求だがMVPのためモック実行")
            scene = mock_global_vlm(people_count, longest_dwell_sec, has_new_visitor, returning)
        else:
            scene = mock_global_vlm(people_count, longest_dwell_sec, has_new_visitor, returning)
        state.last_global_vlm_time = time.time()
        state.last_global_scene_info = scene
        return scene

