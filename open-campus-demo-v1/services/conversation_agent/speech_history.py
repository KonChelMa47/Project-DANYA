"""発話履歴の更新と重複検査。"""

from __future__ import annotations

from agent_state import AgentState
from schemas import SpeechOutput


class SpeechHistory:
    @staticmethod
    def opening_of(output: SpeechOutput) -> str:
        if output.segments:
            return output.segments[0].text.strip()[:28]
        return output.speech.strip()[:28]

    def is_too_similar(self, output: SpeechOutput, state: AgentState) -> bool:
        opening = self.opening_of(output)
        if opening and opening in state.recent_openings[-5:]:
            return True
        normalized = output.speech.replace("\n", "")
        return normalized in [s.replace("\n", "") for s in state.recent_speeches[-5:]]

    def remember(self, output: SpeechOutput, state: AgentState) -> None:
        state.remember_opening(output.target_visitor_id, self.opening_of(output))
        state.remember_emotion_flow([seg.emotion for seg in output.segments])
