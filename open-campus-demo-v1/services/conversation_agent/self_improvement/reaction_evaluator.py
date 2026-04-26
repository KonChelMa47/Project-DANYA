"""簡易反応評価。"""

from __future__ import annotations

from typing import Optional

from schemas import EvaluationResult, GlobalSceneInfo, VisitorInfo


class ReactionEvaluator:
    def evaluate(
        self,
        before_visitor: Optional[VisitorInfo],
        after_visitor: Optional[VisitorInfo],
        before_people_count: int,
        after_people_count: int,
        before_scene: Optional[GlobalSceneInfo],
        after_scene: Optional[GlobalSceneInfo],
    ) -> EvaluationResult:
        score = 0.0
        reasons = []
        if before_visitor and after_visitor:
            if after_visitor.dwell_time_sec > before_visitor.dwell_time_sec:
                score += 0.3
                reasons.append("滞在時間増")
            if after_visitor.bbox_width_px > before_visitor.bbox_width_px:
                score += 0.1
                reasons.append("距離接近推定")
            if after_visitor.returning:
                score += 0.2
                reasons.append("戻り訪問")
        if after_people_count > before_people_count:
            score += 0.2
            reasons.append("人数増")
        elif after_people_count < before_people_count:
            score -= 0.2
            reasons.append("人数減")
        if before_scene and after_scene:
            leaving_delta = after_scene.leaving_risk - before_scene.leaving_risk
            score += (-leaving_delta) * 0.2
            engagement_delta = after_scene.engagement_estimate - before_scene.engagement_estimate
            score += engagement_delta * 0.2
        else:
            engagement_delta = 0.0
        result = "neutral"
        if score >= 0.35:
            result = "success"
        elif score <= -0.2:
            result = "failure"
        return EvaluationResult(result=result, engagement_delta=engagement_delta, reason=";".join(reasons))

