#!/usr/bin/env python3
"""LLM発話のスモークテスト: 各モードで3回ずつ生成し、本文を標準出力に出す。

使い方:
  cd open-campus-demo-v1/services/conversation_agent
  python3 scripts/test_speech_generation_smoke.py

要 OPENAI_API_KEY。DANYA_FORCE_TEMPLATE=1 のときはスキップする。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# conversation_agent をパスに載せる
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(_ROOT.parent / "visitor_tracker" / ".env")
load_dotenv(_ROOT / ".env")


def main() -> int:
    if os.getenv("DANYA_FORCE_TEMPLATE", "").lower() in {"1", "true", "yes"}:
        print("DANYA_FORCE_TEMPLATE が有効なためスキップ")
        return 0
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY が無いためスキップ")
        return 0

    import config
    from agent_state import AgentState
    from behavior_composer import BehaviorComposer
    from emotion_planner import EmotionPlanner
    from schemas import EmotionPlan, TopicPlan
    from special_talker import DAILY_CHITCHAT_TOPICS, SPECIAL_KINDS, SpecialTalker

    if not config.use_llm:
        print("config.use_llm が False のためスキップ")
        return 0

    state = AgentState()
    composer = BehaviorComposer()
    planner = EmotionPlanner()
    special = SpecialTalker()

    tp_intro = TopicPlan(
        mode="intro",
        target_visitor_id="visitor_test",
        people_count=1,
        audience_type="high_school_student",
        audience_label="そこのメガネの学生さん",
        primary_topic="プロジェクトデザイン",
        topics=["プロジェクトデザイン", "KIT", "展示"],
        knowledge_points=[
            "KITのプロジェクトデザインは1年生から課題を見つけて手を動かす流れがある。",
            "ダーニャは情報理工学部の学生が短期間で作った会話アバター。",
        ],
        use_visual_detail=True,
        vlm_observations=["メガネが見えている", "展示の前で止まっている"],
        vlm_humor="近づきすぎて画面が圧力になっとる。",
    )
    tp_crowd = TopicPlan(
        mode="crowd",
        people_count=3,
        audience_type="group",
        audience_label="みなさん",
        primary_topic="情報理工学部",
        topics=["情報理工学部", "ナナマル"],
        knowledge_points=[
            "情報理工学部には情報工学科・知能情報システム学科・ロボティクス学科がある。",
            "ナナマルはヒューマノイドロボットの展示だ。",
        ],
        use_visual_detail=False,
    )
    tp_return = TopicPlan(
        mode="returning",
        target_visitor_id="visitor_test",
        people_count=1,
        audience_type="general",
        audience_label="目の前の来場者さん",
        primary_topic="人物追跡",
        topics=["人物追跡", "LLM"],
        knowledge_points=["人物追跡で滞在に合わせて話題の深さを変えられる。", "LLMは感情タグ付きの短文にしている。"],
        is_returning=True,
        use_visual_detail=False,
    )

    scenarios: list[tuple[str, TopicPlan, bool]] = [
        ("normal_long_intro", tp_intro, True),
        ("normal_short_intro", tp_intro, False),
        ("normal_crowd_long", tp_crowd, True),
        ("normal_returning", tp_return, True),
    ]

    runs = 3
    print("=== BehaviorComposer（通常発話）===\n")
    for name, tp, long_form in scenarios:
        print(f"--- {name} (各{runs}回) ---")
        for i in range(1, runs + 1):
            ep = planner.plan(tp, state, long_form=long_form)
            speech = composer.compose(tp, ep, [], state, long_form=long_form)
            print(f"[{i}] {speech.strategy_summary}\n{speech.speech}\n")

    print("\n=== SpecialTalker（特別・日常）===\n")
    base_tp = tp_intro.model_copy(update={"mode": "intro", "people_count": 2})
    for kind in SPECIAL_KINDS:
        print(f"--- special:{kind} ---")
        for i in range(1, runs + 1):
            sp = special.compose(kind, base_tp, people_count=2)
            print(f"[{i}] {sp.topic}\n{sp.speech}\n")

    daily_sample = list(DAILY_CHITCHAT_TOPICS)[:5]
    for dk in daily_sample:
        print(f"--- daily:{dk} ---")
        for i in range(1, runs + 1):
            sp = special.compose("daily_chitchat", base_tp, people_count=1, daily_key=dk)
            print(f"[{i}] {sp.topic}\n{sp.speech}\n")

    print("=== generate_dedupe_append_segment ===\n")
    from llm_client import generate_dedupe_append_segment

    for i in range(1, runs + 1):
        d = generate_dedupe_append_segment(
            {
                "recent_speech": "<happy_normal>テストの繰り返し文です。",
                "reason": "似すぎ回避",
            }
        )
        print(f"[{i}] {d}\n")

    print("完了")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
