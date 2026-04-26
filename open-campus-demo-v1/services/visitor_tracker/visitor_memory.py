"""visitor_idの記憶管理を担当するモジュール。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import time

import numpy as np

import config
from feature_extractor import (
    clothing_similarity,
    position_similarity,
    same_person_score,
    time_similarity,
)


@dataclass
class VisitorState:
    """1人の来場者状態を保持するデータ構造。"""

    visitor_id: str
    status: str = "active"  # active / lost / expired
    first_seen: float = 0.0
    last_seen: float = 0.0
    lost_since: Optional[float] = None
    total_dwell_time_sec: float = 0.0
    current_dwell_time_sec: float = 0.0
    visit_count: int = 1
    last_position: Tuple[int, int, int, int] = (0, 0, 0, 0)
    clothing_hist: Optional[np.ndarray] = None
    same_person_confidence: float = 0.0
    was_lost_and_returned: bool = False
    current_track_id: Optional[int] = None
    is_long_stay: bool = False
    bbox_width_px: int = 0
    gender: str = "unknown"
    age_estimate: str = "unknown"
    clothing_description: str = "unknown"
    expression: str = "unknown"
    pose_description: str = "unknown"
    accessories: List[str] = field(default_factory=list)
    carried_items: List[str] = field(default_factory=list)
    expression_confidence: float = 0.0
    vlm_analyzed: bool = False

    # 内部計算用
    _active_started_at: float = field(default=0.0, repr=False)

    def to_json_dict(self) -> dict:
        """外部連携しやすいJSON形式辞書へ変換する。"""
        return {
            "visitor_id": self.visitor_id,
            "status": self.status,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "lost_since": self.lost_since,
            "current_dwell_time_sec": round(self.current_dwell_time_sec, 3),
            "total_dwell_time_sec": round(self.total_dwell_time_sec, 3),
            "visit_count": self.visit_count,
            "last_position": list(self.last_position),
            "same_person_confidence": round(self.same_person_confidence, 3),
            "was_lost_and_returned": self.was_lost_and_returned,
            "bbox": list(self.last_position),
            "is_long_stay": self.is_long_stay,
            "bbox_width_px": self.bbox_width_px,
            "gender": self.gender,
            "age_estimate": self.age_estimate,
            "clothing_description": self.clothing_description,
            "expression": self.expression,
            "pose_description": self.pose_description,
            "accessories": self.accessories,
            "carried_items": self.carried_items,
            "expression_confidence": round(self.expression_confidence, 3),
            "vlm_analyzed": self.vlm_analyzed,
        }


class VisitorMemory:
    """visitor_idの生成・追跡・復帰判定を管理するクラス。"""

    def __init__(
        self,
        lost_timeout_sec: float = config.LOST_TIMEOUT_SEC,
        same_person_threshold: float = config.SAME_PERSON_THRESHOLD,
    ) -> None:
        self.lost_timeout_sec = lost_timeout_sec
        self.same_person_threshold = same_person_threshold

        self.visitors: Dict[str, VisitorState] = {}
        self.track_to_visitor: Dict[int, str] = {}
        self._visitor_serial = 0

    def _new_visitor_id(self) -> str:
        """連番のvisitor_idを発行する。"""
        self._visitor_serial += 1
        return f"visitor_{self._visitor_serial:03d}"

    def _create_new_visitor(
        self,
        now: float,
        track_id: Optional[int],
        bbox: Tuple[int, int, int, int],
        hist: np.ndarray,
    ) -> VisitorState:
        """新規来場者を作成する。"""
        vid = self._new_visitor_id()
        visitor = VisitorState(
            visitor_id=vid,
            status="active",
            first_seen=now,
            last_seen=now,
            lost_since=None,
            total_dwell_time_sec=0.0,
            current_dwell_time_sec=0.0,
            visit_count=1,
            last_position=bbox,
            clothing_hist=hist,
            same_person_confidence=1.0,
            was_lost_and_returned=False,
            current_track_id=track_id,
            _active_started_at=now,
        )
        self.visitors[vid] = visitor
        if track_id is not None:
            self.track_to_visitor[track_id] = vid
        return visitor

    @staticmethod
    def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        return inter / float(area_a + area_b - inter)

    def _score_lost_visitor(
        self,
        visitor: VisitorState,
        now: float,
        bbox: Tuple[int, int, int, int],
        hist: np.ndarray,
        frame_shape: Tuple[int, int, int],
    ) -> float:
        """lost中visitorとの同一人物スコアを計算する。"""
        if visitor.lost_since is None:
            return 0.0

        lost_duration = now - visitor.lost_since
        clothing_sim = clothing_similarity(visitor.clothing_hist, hist)
        position_sim = position_similarity(visitor.last_position, bbox, frame_shape)
        time_sim = time_similarity(lost_duration, self.lost_timeout_sec)
        return same_person_score(clothing_sim, position_sim, time_sim)

    def _reactivate_visitor(
        self,
        visitor: VisitorState,
        now: float,
        track_id: Optional[int],
        bbox: Tuple[int, int, int, int],
        hist: np.ndarray,
        score: float,
    ) -> None:
        """lost visitorをactiveへ復帰させる。"""
        lost_duration = (now - visitor.lost_since) if visitor.lost_since is not None else self.lost_timeout_sec
        continuous = lost_duration <= config.CONTINUITY_GRACE_SEC
        visitor.status = "active"
        visitor.last_seen = now
        visitor.current_track_id = track_id
        visitor.last_position = bbox
        visitor.clothing_hist = hist
        visitor.same_person_confidence = score
        visitor.was_lost_and_returned = not continuous
        if not continuous:
            visitor.visit_count += 1
            visitor._active_started_at = now
            visitor.current_dwell_time_sec = 0.0
        else:
            visitor.current_dwell_time_sec = now - visitor._active_started_at
        visitor.lost_since = None
        if track_id is not None:
            self.track_to_visitor[track_id] = visitor.visitor_id

    def _score_active_visitor(
        self,
        visitor: VisitorState,
        now: float,
        bbox: Tuple[int, int, int, int],
        hist: np.ndarray,
        frame_shape: Tuple[int, int, int],
    ) -> float:
        """active中visitorとの近さを計算する（track_id変化対策）。"""
        # active同士の比較では時間差はごく短い想定なので1.0に近く扱う
        # ※ last_seenが古くなりすぎている場合だけ少し下げる
        delta_sec = max(0.0, now - visitor.last_seen)
        time_sim = 1.0 if delta_sec <= 1.0 else max(0.0, 1.0 - (delta_sec / 5.0))
        clothing_sim = clothing_similarity(visitor.clothing_hist, hist)
        position_sim = position_similarity(visitor.last_position, bbox, frame_shape)
        return same_person_score(clothing_sim, position_sim, time_sim)

    def _update_active_visitor(
        self,
        visitor: VisitorState,
        now: float,
        track_id: Optional[int],
        bbox: Tuple[int, int, int, int],
        hist: np.ndarray,
    ) -> None:
        """active visitorの時刻・特徴を更新する。"""
        visitor.last_seen = now
        visitor.current_track_id = track_id
        visitor.last_position = bbox
        visitor.clothing_hist = hist
        visitor.status = "active"
        visitor.lost_since = None
        visitor.same_person_confidence = max(visitor.same_person_confidence, 0.80)
        visitor.current_dwell_time_sec = now - visitor._active_started_at
        visitor.bbox_width_px = max(0, bbox[2] - bbox[0])
        if track_id is not None:
            self.track_to_visitor[track_id] = visitor.visitor_id

    def _mark_not_seen_as_lost(self, seen_visitor_ids: Set[str], now: float) -> None:
        """今フレームで見えなかったactive visitorをlostへ遷移。"""
        for visitor in self.visitors.values():
            if visitor.status != "active":
                continue
            if visitor.visitor_id in seen_visitor_ids:
                continue
            # active区間の滞在時間をtotalへ加算
            visitor.total_dwell_time_sec += now - visitor._active_started_at
            visitor.current_dwell_time_sec = 0.0
            visitor.status = "lost"
            visitor.lost_since = now
            visitor.current_track_id = None

    def _expire_old_lost_visitors(self, now: float) -> None:
        """lost timeoutを超えたvisitorをexpiredにする。"""
        for visitor in self.visitors.values():
            if visitor.status != "lost":
                continue
            if visitor.lost_since is None:
                continue
            if (now - visitor.lost_since) > self.lost_timeout_sec:
                visitor.status = "expired"
                visitor.same_person_confidence = 0.0

    def update(
        self,
        detections: List[dict],
        frame_shape: Tuple[int, int, int],
        now: Optional[float] = None,
    ) -> Dict[int, str]:
        """1フレーム分の検出結果を反映し、track_id->visitor_idを返す。"""
        if now is None:
            now = time.time()

        # 今フレームで使用されたvisitor_id
        seen_visitor_ids: Set[str] = set()
        track_to_visitor_this_frame: Dict[int, str] = {}

        for det in detections:
            track_id = det.get("track_id", None)
            bbox = det["bbox"]
            hist = det["clothing_hist"]

            assigned_visitor: Optional[VisitorState] = None

            # 1) まずtrack_idが既知なら、そのvisitorを優先利用
            if track_id is not None and track_id in self.track_to_visitor:
                vid = self.track_to_visitor[track_id]
                visitor = self.visitors.get(vid)
                if visitor and visitor.status != "expired":
                    assigned_visitor = visitor

            # 2) 未割当ならlost中visitorとの類似度で再マッチ
            if assigned_visitor is None:
                # 2-a) まずactive中（ただし今フレーム未使用）のvisitorに再マッチを試みる。
                #      ByteTrackのtrack_idが切り替わった瞬間でもvisitor_idを維持するため。
                best_active_visitor = None
                best_active_score = 0.0
                for visitor in self.visitors.values():
                    if visitor.status != "active":
                        continue
                    if visitor.visitor_id in seen_visitor_ids:
                        continue
                    score = self._score_active_visitor(
                        visitor=visitor,
                        now=now,
                        bbox=bbox,
                        hist=hist,
                        frame_shape=frame_shape,
                    )
                    if score > best_active_score:
                        best_active_score = score
                        best_active_visitor = visitor

                if (
                    best_active_visitor is not None
                    and best_active_score >= self.same_person_threshold
                ):
                    best_active_visitor.same_person_confidence = best_active_score
                    assigned_visitor = best_active_visitor

            # 2-b) active再マッチで決まらなければ、lost中visitorとの復帰判定へ進む
            if assigned_visitor is None:
                best_visitor = None
                best_score = 0.0
                for visitor in self.visitors.values():
                    if visitor.status != "lost":
                        continue
                    if visitor.lost_since is None:
                        continue
                    if (now - visitor.lost_since) > self.lost_timeout_sec:
                        continue
                    score = self._score_lost_visitor(
                        visitor=visitor,
                        now=now,
                        bbox=bbox,
                        hist=hist,
                        frame_shape=frame_shape,
                    )
                    if score > best_score:
                        best_score = score
                        best_visitor = visitor

                if best_visitor is not None and best_score >= self.same_person_threshold:
                    self._reactivate_visitor(
                        visitor=best_visitor,
                        now=now,
                        track_id=track_id,
                        bbox=bbox,
                        hist=hist,
                        score=best_score,
                    )
                    assigned_visitor = best_visitor

            # 2-c) 単独運用中の一瞬のID揺れは、位置が十分近ければ同じ人として継続する。
            if assigned_visitor is None:
                recent_lost = [
                    visitor
                    for visitor in self.visitors.values()
                    if visitor.status == "lost"
                    and visitor.lost_since is not None
                    and (now - visitor.lost_since) <= config.CONTINUITY_GRACE_SEC
                ]
                if len(recent_lost) == 1:
                    visitor = recent_lost[0]
                    overlap = self._bbox_iou(visitor.last_position, bbox)
                    position_sim = position_similarity(visitor.last_position, bbox, frame_shape)
                    if overlap >= 0.20 or position_sim >= 0.45:
                        score = max(overlap, position_sim, 0.80)
                        self._reactivate_visitor(
                            visitor=visitor,
                            now=now,
                            track_id=track_id,
                            bbox=bbox,
                            hist=hist,
                            score=score,
                        )
                        assigned_visitor = visitor

            # 3) それでも未割当なら新規visitorを作成
            if assigned_visitor is None:
                assigned_visitor = self._create_new_visitor(
                    now=now,
                    track_id=track_id,
                    bbox=bbox,
                    hist=hist,
                )

            # 4) active更新
            self._update_active_visitor(
                visitor=assigned_visitor,
                now=now,
                track_id=track_id,
                bbox=bbox,
                hist=hist,
            )

            seen_visitor_ids.add(assigned_visitor.visitor_id)
            if track_id is not None:
                track_to_visitor_this_frame[track_id] = assigned_visitor.visitor_id

        # 今フレームで見えなかったactive visitorをlost化
        self._mark_not_seen_as_lost(seen_visitor_ids, now)
        # timeout超過lostをexpired化
        self._expire_old_lost_visitors(now)

        return track_to_visitor_this_frame

    def build_frame_json(
        self,
        now: Optional[float] = None,
        include_status: Optional[Set[str]] = None,
    ) -> dict:
        """ダーニャ側へ渡しやすいフレームJSONを生成する。"""
        if now is None:
            now = time.time()
        if include_status is None:
            include_status = {"active", "lost"}

        visitors = []
        visible_people_count = 0

        for visitor in self.visitors.values():
            if visitor.status not in include_status:
                continue
            if visitor.status == "active":
                visible_people_count += 1
                visitor.current_dwell_time_sec = now - visitor._active_started_at
            else:
                visitor.current_dwell_time_sec = 0.0
            visitors.append(visitor.to_json_dict())

        return {
            "timestamp": now,
            "visible_people_count": visible_people_count,
            "visitors": visitors,
        }

    def clear_all(self) -> None:
        """セッション終了時にメモリを破棄する。"""
        self.visitors.clear()
        self.track_to_visitor.clear()
