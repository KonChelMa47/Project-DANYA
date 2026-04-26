"""LangChain経由でVLM推定を行うモジュール。"""

from __future__ import annotations

import base64
import json
import os
from typing import Dict, Tuple

import cv2
import numpy as np
from langchain_openai import ChatOpenAI

import config


class VisitorVLMAnalyzer:
    """人物bbox画像だけを使って属性推定するクラス。"""

    def __init__(self, model_name: str = config.OPENAI_MODEL_NAME) -> None:
        # APIキーがない場合はエラーを投げず、呼び出し側で無効扱いにしやすくする
        self.enabled = bool(os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name
        self.llm = ChatOpenAI(model=self.model_name, temperature=0.2) if self.enabled else None

    def _crop_bbox(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """bbox内だけ切り出す。"""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            return np.zeros((2, 2, 3), dtype=np.uint8)
        return frame[y1:y2, x1:x2]

    def _to_base64_jpeg(self, image: np.ndarray) -> str:
        """画像をJPEG化してbase64文字列化。"""
        ok, buf = cv2.imencode(".jpg", image)
        if not ok:
            raise RuntimeError("画像エンコードに失敗しました。")
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    def _safe_float(self, value, default: float = 0.0) -> float:
        """unknown等が来ても例外にせずfloatへ変換する。"""
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def analyze(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict:
        """bbox画像から属性推定を実行し、辞書で返す。"""
        if not self.enabled or self.llm is None:
            return {
                "vlm_enabled": False,
                "gender": "unknown",
                "age_estimate": "unknown",
                "clothing_description": "unknown",
                "expression": "unknown",
                "pose_description": "unknown",
                "accessories": [],
                "carried_items": [],
                "expression_confidence": 0.0,
            }

        crop = self._crop_bbox(frame, bbox)
        img_b64 = self._to_base64_jpeg(crop)

        prompt = (
            "あなたは展示会向け人物観察アシスタントです。"
            "画像内人物を推定し、必ずJSONのみで返答してください。"
            "キーは gender, age_estimate, clothing_description, expression, pose_description, accessories, carried_items, expression_confidence。"
            "genderは male/female/unknown のいずれか。"
            "age_estimateは 20s, 30s のような年代推定または unknown。"
            "expressionは smiling, neutral, curious, excited, serious, unknown など安全な表情の短い英語。笑顔に見える時はsmilingを優先。"
            "pose_descriptionは standing, leaning, looking_at_display, hands_up, pointing, using_phone, walking など安全な姿勢・動作の短い英語。"
            "accessoriesは glasses, hat, mask, backpack など安全な装着物の配列。"
            "carried_itemsは smartphone, bag, pamphlet, drink など手に持つ物・荷物の配列。"
            "expression_confidenceは0.0-1.0の実数。"
            "体型、容姿の良し悪し、肌の色、国籍、障害、病気、魅力度には触れない。"
            "不確実ならunknownまたは空配列を使ってください。"
        )

        response = self.llm.invoke(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                        },
                    ],
                }
            ]
        )

        text = response.content if isinstance(response.content, str) else str(response.content)
        text = text.strip()
        if text.startswith("```"):
            text = text.replace("```json", "").replace("```", "").strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = {}

        return {
            "vlm_enabled": True,
            "gender": parsed.get("gender", "unknown"),
            "age_estimate": parsed.get("age_estimate", "unknown"),
            "clothing_description": parsed.get("clothing_description", "unknown"),
            "expression": parsed.get("expression", "unknown"),
            "pose_description": parsed.get("pose_description", "unknown"),
            "accessories": parsed.get("accessories", []),
            "carried_items": parsed.get("carried_items", []),
            "expression_confidence": self._safe_float(
                parsed.get("expression_confidence", 0.0), default=0.0
            ),
        }

    def analyze_scene(self, frame: np.ndarray, people_count: int) -> Dict:
        """全体フレームから展示前の雰囲気を推定する。"""
        if not self.enabled or self.llm is None:
            return self.mock_scene(people_count)

        h, w = frame.shape[:2]
        scale = min(1.0, 768 / max(h, w))
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        img_b64 = self._to_base64_jpeg(frame)
        prompt = (
            "あなたはオープンキャンパス展示の会場観察アシスタントです。"
            "画像全体を見て、展示前にいる人の雰囲気を推定してください。"
            "必ずJSONのみで返答してください。"
            "キーは summary, likely_audience, scene_mood, topic_hints, use_visual_detail。"
            "likely_audienceは high_school_student,parent_or_adult,child,group,general,unknown のいずれか。"
            "topic_hintsは短い文字列配列で最大3個。"
            "人数、立ち位置、向いている方向、スマホやパンフレットなどの持ち物、手を振る/指差す/覗き込む等のポーズは使ってよい。"
            "体型、容姿の良し悪し、肌の色、国籍、障害、病気、魅力度には触れない。"
            "不確実ならunknownを使ってください。"
        )
        response = self.llm.invoke(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    ],
                }
            ]
        )
        text = response.content if isinstance(response.content, str) else str(response.content)
        text = text.strip()
        if text.startswith("```"):
            text = text.replace("```json", "").replace("```", "").strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = {}
        return {
            "summary": parsed.get("summary", "会場全体を観察中"),
            "likely_audience": parsed.get("likely_audience", "unknown"),
            "scene_mood": parsed.get("scene_mood", "unknown"),
            "topic_hints": parsed.get("topic_hints", []),
            "use_visual_detail": bool(parsed.get("use_visual_detail", True)),
        }

    def mock_scene(self, people_count: int) -> Dict:
        if people_count <= 0:
            return {
                "summary": "展示前に人は少なく、呼び込み向き",
                "likely_audience": "general",
                "scene_mood": "idle",
                "topic_hints": ["ダーニャ自身", "ナナマルへの嫉妬"],
                "use_visual_detail": False,
            }
        if people_count >= 2:
            return {
                "summary": f"展示前に{people_count}人ほどいて、グループ向けに話せそう",
                "likely_audience": "group",
                "scene_mood": "busy",
                "topic_hints": ["人物追跡", "KITのものづくり教育"],
                "use_visual_detail": True,
            }
        return {
            "summary": "展示前に1人いて、相手に合わせた導入がよさそう",
            "likely_audience": "unknown",
            "scene_mood": "watching",
            "topic_hints": ["画像AIで雰囲気を見る", "情報理工学部"],
            "use_visual_detail": True,
        }
