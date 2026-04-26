"""感情フローを作る。"""

from __future__ import annotations

import random

from agent_state import AgentState
from schemas import EmotionPlan, EmotionName, TopicPlan


LONG_FLOWS: list[list[EmotionName]] = [
    ["happy_normal", "surprised_high", "fear_normal", "happy_high"],
    ["happy_normal", "angry_normal", "surprised_high", "happy_normal"],
    ["happy_normal", "fear_high", "sad_normal", "happy_high"],
    ["surprised_high", "happy_normal", "angry_high", "happy_normal"],
    ["happy_normal", "sad_high", "surprised_normal", "happy_high"],
    ["surprised_normal", "happy_normal", "fear_high", "happy_normal"],
    ["fear_normal", "happy_normal", "surprised_high", "happy_high"],
    ["happy_normal", "angry_high", "fear_high", "happy_normal"],
]

SHORT_FLOWS: list[list[EmotionName]] = [
    ["sad_normal", "happy_normal"],
    ["angry_normal", "happy_normal"],
    ["fear_normal", "happy_normal"],
    ["happy_normal"],
    ["surprised_normal", "happy_normal"],
]


class EmotionPlanner:
    _EXTREME = (
        "金沢弁語尾を多めに。happy_normalを土台に多めに置き、間にsurprised/fear/sad/angryで山を作る。"
        "ピークはhappy_highやsurprised_highで振り切る。happy_normal一色の平坦さだけは避ける。"
    )

    def plan(self, topic_plan: TopicPlan, state: AgentState, *, long_form: bool) -> EmotionPlan:
        flows = LONG_FLOWS if long_form else SHORT_FLOWS
        candidates = flows.copy()
        random.shuffle(candidates)
        for flow in candidates:
            key = ">".join(flow)
            if key not in state.recent_emotion_flows[-3:]:
                return EmotionPlan(flow=flow, style_note=self._style_note(topic_plan, long_form))
        return EmotionPlan(flow=candidates[0], style_note=self._style_note(topic_plan, long_form))

    def _style_note(self, topic_plan: TopicPlan, long_form: bool) -> str:
        if not long_form:
            return (
                self._EXTREME
                + "短い独り言。1から2文。『え、うわ、ちょっと待って』の勢いに、少し生意気なツッコミを混ぜる"
            )
        if topic_plan.audience_type == "child":
            return (
                self._EXTREME
                + "4セグメント以内。感情の落差を大きく。子供にも分かる言葉で、えらそうすぎない小さな自慢とツッコミを入れる"
            )
        if topic_plan.audience_type == "parent_or_adult":
            return (
                self._EXTREME
                + "4セグメント以内。驚き、緊張、軽い嫉妬を切り替え、技術と教育的価値を少し生意気な展示MC口調で話す"
            )
        if topic_plan.audience_type == "high_school_student":
            return (
                self._EXTREME
                + "4セグメント以内。『え、まじで』『うわ、待って』くらいの勢いで、少し先輩ぶりながら進路やAIへの興味を引き出す"
            )
        if topic_plan.audience_type == "group":
            return (
                self._EXTREME
                + "4セグメント以内。会場MCとして全体へ、少し生意気な呼び込みとツッコミで感情を大きく振る"
            )
        return (
            self._EXTREME
            + "4セグメント以内。驚き・怖がり・嫉妬・明るさを、少し生意気な言葉の入り方で激しく切り替える"
        )
