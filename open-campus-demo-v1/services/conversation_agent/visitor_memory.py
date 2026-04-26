"""発話側の軽量visitor記憶と相手タイプ推定。"""

from __future__ import annotations

from dataclasses import dataclass

from agent_state import AgentState
from schemas import AudienceType, VisitorInfo


@dataclass
class AudienceContext:
    audience_type: AudienceType
    audience_label: str
    visual_hook: str
    vlm_observations: list[str]
    vlm_humor: str
    vlm_confidence_note: str
    is_returning: bool = False


class AgentVisitorMemory:
    CLOTHING_JA = {
        "sweater": "セーター",
        "dark sweater": "暗めのセーター",
        "casual sweater": "カジュアルなセーター",
        "knit sweater": "ニットの服",
        "knitted garment": "ニットの服",
        "knitted cardigan": "ニットのカーディガン",
        "brown cardigan": "茶色のカーディガン",
        "brown cardigan over a black shirt": "茶色のカーディガンに黒いシャツ",
        "dark shirt": "暗めのシャツ",
        "black shirt": "黒いシャツ",
        "blue shirt": "青いシャツ",
        "white shirt": "白いシャツ",
        "hoodie": "パーカー",
        "blue hoodie": "青いパーカー",
        "suit": "スーツ",
        "blazer": "ジャケット",
        "unknown": "",
    }
    EXPRESSION_JA = {
        "smiling": "笑顔",
        "smile": "笑顔",
        "happy": "笑顔",
        "neutral": "落ち着いた表情",
        "curious": "興味ありそうな表情",
        "excited": "楽しそうな表情",
        "serious": "真剣な表情",
        "unknown": "",
    }
    AGE_JA = {
        "20s": "20代くらい",
        "30s": "30代くらい",
        "40s": "40代くらい",
        "50s": "50代くらい",
        "teen": "高校生くらい",
        "unknown": "",
    }
    POSE_JA = {
        "standing": "立ち止まっている",
        "leaning": "少し身を乗り出している",
        "looking_at_display": "展示をじっと見ている",
        "looking_down": "手元か画面を見ている",
        "hands_up": "手を上げて反応している",
        "pointing": "何かを指差している",
        "using_phone": "スマホを見ている",
        "walking": "歩きながら近づいている",
        "unknown": "",
    }
    ITEM_JA = {
        "smartphone": "スマホ",
        "phone": "スマホ",
        "bag": "バッグ",
        "backpack": "リュック",
        "pamphlet": "パンフレット",
        "drink": "飲み物",
        "camera": "カメラ",
        "notebook": "ノート",
        "unknown": "",
    }
    ACCESSORY_JA = {
        "glasses": "メガネ",
        "hat": "帽子",
        "mask": "マスク",
        "backpack": "リュック",
        "unknown": "",
    }

    def update_seen(self, visitors: list[VisitorInfo], now: float, state: AgentState) -> None:
        for visitor in visitors:
            if visitor.visitor_id == "unknown":
                continue
            state.visitor_last_seen[visitor.visitor_id] = now

    def build_context(
        self,
        *,
        visitor: VisitorInfo | None,
        people_count: int,
        state: AgentState,
    ) -> AudienceContext:
        if people_count >= 2:
            return AudienceContext(
                audience_type="group",
                audience_label=self._group_label(people_count, visitor),
                visual_hook=self._visual_hook(visitor, "group") if visitor else "みんなで展示を見に来てくれた雰囲気",
                vlm_observations=self._vlm_observations(visitor) if visitor else ["グループで来てくれた雰囲気"],
                vlm_humor=self._vlm_humor(visitor, "group") if visitor else "人数が増えると僕のカメラの目が急に体育館みたいになるんや。",
                vlm_confidence_note=self._confidence_note(visitor) if visitor else "会場全体の人数と立ち位置を中心に話す",
                is_returning=bool(visitor and self._is_returning(visitor, state)),
            )
        if visitor is None:
            return AudienceContext(
                audience_type="general",
                audience_label="近くを通る人",
                visual_hook="まだ誰も目の前にはおらん状態",
                vlm_observations=["人物特徴はまだ取得前"],
                vlm_humor="見えていない相手に話しかけるの、画面の中の僕としてはなかなか勇気いるんや。",
                vlm_confidence_note="人物が映るまでは会場MCとして広めに話す",
            )

        audience_type = self._infer_audience(visitor)
        label = self._label_for(visitor, audience_type)
        return AudienceContext(
            audience_type=audience_type,
            audience_label=label,
            visual_hook=self._visual_hook(visitor, audience_type),
            vlm_observations=self._vlm_observations(visitor),
            vlm_humor=self._vlm_humor(visitor, audience_type),
            vlm_confidence_note=self._confidence_note(visitor),
            is_returning=self._is_returning(visitor, state),
        )

    def _infer_audience(self, visitor: VisitorInfo) -> AudienceType:
        age = visitor.vlm.age_estimate.lower()
        clothing = visitor.vlm.clothing_description.lower()
        accessory_text = " ".join(visitor.vlm.accessories).lower()
        item_text = " ".join(visitor.vlm.carried_items).lower()
        joined = f"{age} {clothing} {accessory_text} {item_text}"
        if self._has_suit(visitor):
            return "parent_or_adult"
        if any(word in joined for word in ("child", "kid", "小学生", "子供", "子ども", "幼")):
            return "child"
        if any(word in joined for word in ("teen", "10代", "20s", "高校", "student", "学生", "リュック", "backpack", "pamphlet")):
            return "high_school_student"
        if any(word in joined for word in ("adult", "30", "40", "50", "保護者", "parent")):
            return "parent_or_adult"
        return "general"

    def _label_for(self, visitor: VisitorInfo, audience_type: AudienceType) -> str:
        clothing = visitor.vlm.clothing_description
        age = visitor.vlm.age_estimate.lower()
        gender = visitor.vlm.gender.lower()
        clothing_lower = clothing.lower()
        has_glasses = "メガネ" in clothing or "glasses" in clothing_lower
        accessory_text = " ".join(visitor.vlm.accessories).lower()
        item_text = " ".join(visitor.vlm.carried_items).lower()
        has_glasses = has_glasses or "glasses" in accessory_text
        has_backpack = (
            "リュック" in clothing
            or "backpack" in clothing_lower
            or "backpack" in accessory_text
            or "backpack" in item_text
        )
        has_suit = self._has_suit(visitor)
        has_pamphlet = "pamphlet" in item_text or "パンフレット" in item_text
        is_teen = any(word in age for word in ("teen", "10代", "高校"))
        is_young_adult = is_teen or any(word in age for word in ("20s", "student", "学生"))
        if audience_type == "child":
            return "小学生くらいの子"
        if audience_type == "high_school_student":
            if has_glasses and has_backpack:
                return "メガネでリュック背負っとる学生さん"
            if has_glasses:
                return "そこのメガネのお兄さん" if gender == "male" or is_young_adult else "そこのメガネの人"
            if has_backpack:
                return "リュック背負っとる学生さん"
            if has_pamphlet:
                return "パンフレット持っとる学生さん"
            if is_teen:
                return "高校生くらいの学生さん"
            if gender == "male" and is_young_adult:
                return "そこの学生のお兄さん"
            return "学生さん"
        if audience_type == "parent_or_adult":
            if has_suit:
                return "スーツのお偉いさん"
            return "保護者の方"
        if has_glasses:
            return "メガネのお一人さん"
        return "お一人さん"

    def _group_label(self, people_count: int, visitor: VisitorInfo | None) -> str:
        if visitor and self._has_suit(visitor):
            return "スーツのお偉いさんがいるグループ"
        inferred = self._infer_audience(visitor) if visitor else "general"
        if visitor and inferred == "child":
            return "親子で来てくれたみなさん"
        if visitor and inferred == "parent_or_adult" and people_count <= 3:
            return "保護者の方と学生さん"
        if visitor and inferred == "high_school_student":
            return "友達グループのみなさん" if people_count >= 3 else "友達同士のお二人"
        if people_count >= 4:
            return "にぎやかな友達グループのみなさん"
        if people_count == 2:
            return "二人組のお客さん"
        return "友達グループのみなさん"

    def _visual_hook(self, visitor: VisitorInfo, audience_type: AudienceType) -> str:
        parts = self._vlm_observations(visitor)
        if audience_type == "high_school_student":
            parts.append("進路やものづくりに興味がありそう")
        elif audience_type == "parent_or_adult":
            parts.append("技術や教育的な価値を見てくれそう")
        elif audience_type == "child":
            parts.append("ロボットやAIを楽しく見てくれそう")
        return "、".join(parts) if parts else "展示に興味を持って近づいてくれた雰囲気"

    @staticmethod
    def _has_suit(visitor: VisitorInfo) -> bool:
        clothing = visitor.vlm.clothing_description.lower()
        accessory_text = " ".join(visitor.vlm.accessories).lower()
        return any(word in f"{clothing} {accessory_text}" for word in ("suit", "スーツ", "blazer", "ジャケット"))

    def _vlm_observations(self, visitor: VisitorInfo | None) -> list[str]:
        if visitor is None:
            return []
        observations: list[str] = []
        clothing = visitor.vlm.clothing_description
        expression = visitor.vlm.expression
        age = visitor.vlm.age_estimate
        clothing_ja = self._natural_clothing(clothing)
        if clothing_ja:
            observations.append(f"{clothing_ja}が見えている")
        pose_ja = self._natural_pose(visitor.vlm.pose_description)
        if pose_ja:
            observations.append(f"{pose_ja}")
        item_parts = self._natural_list(visitor.vlm.carried_items, self.ITEM_JA)
        if item_parts:
            observations.append(f"{'や'.join(item_parts[:2])}を持っている")
        accessory_parts = self._natural_list(visitor.vlm.accessories, self.ACCESSORY_JA)
        if accessory_parts:
            observations.append(f"{'や'.join(accessory_parts[:2])}が見えている")
        age_ja = self._natural_age(age)
        if age_ja:
            observations.append(f"{age_ja}に見えている")
        expr_ja = self._natural_expression(expression)
        if expr_ja:
            observations.append("笑顔が素敵" if self._is_smiling_expression(expression) else f"{expr_ja}で見てくれている")
        if visitor.bbox_width_px >= 900:
            observations.append("かなり近くまで来てくれている")
        elif visitor.bbox_width_px >= 450:
            observations.append("ちゃんと展示の前で止まってくれている")
        return observations or ["カメラの目が会場を見ている"]

    def _natural_clothing(self, clothing: str) -> str:
        raw = (clothing or "").strip()
        lowered = raw.lower()
        if lowered in self.CLOTHING_JA:
            return self.CLOTHING_JA[lowered]
        for key, value in self.CLOTHING_JA.items():
            if key and key in lowered:
                return value
        if not raw or lowered == "unknown":
            return ""
        return raw

    def _natural_expression(self, expression: str) -> str:
        lowered = (expression or "").strip().lower()
        return self.EXPRESSION_JA.get(lowered, expression if expression and lowered != "unknown" else "")

    @staticmethod
    def _is_smiling_expression(expression: str) -> bool:
        lowered = (expression or "").strip().lower()
        return any(word in lowered for word in ("smil", "happy", "laugh", "笑顔", "にこ"))

    def _natural_age(self, age: str) -> str:
        lowered = (age or "").strip().lower()
        if lowered in self.AGE_JA:
            return self.AGE_JA[lowered]
        for key, value in self.AGE_JA.items():
            if key and key in lowered:
                return value
        if not age or lowered == "unknown":
            return ""
        return f"{age}くらい"

    def _natural_pose(self, pose: str) -> str:
        lowered = (pose or "").strip().lower()
        if lowered in self.POSE_JA:
            return self.POSE_JA[lowered]
        for key, value in self.POSE_JA.items():
            if key and key in lowered:
                return value
        if not pose or lowered == "unknown":
            return ""
        return pose

    def _natural_list(self, values: list[str], mapping: dict[str, str]) -> list[str]:
        result: list[str] = []
        for value in values or []:
            lowered = str(value or "").strip().lower()
            if not lowered or lowered == "unknown":
                continue
            translated = mapping.get(lowered)
            if not translated:
                translated = next((ja for key, ja in mapping.items() if key and key in lowered), "")
            result.append(translated or str(value))
        deduped: list[str] = []
        for item in result:
            if item and item not in deduped:
                deduped.append(item)
        return deduped[:3]

    def _vlm_humor(self, visitor: VisitorInfo | None, audience_type: AudienceType) -> str:
        if visitor is None:
            return "カメラの前に気配だけあると、僕の画像AIが準備運動し始めるんや。"
        clothing = visitor.vlm.clothing_description.lower()
        expression = visitor.vlm.expression.lower()
        pose = visitor.vlm.pose_description.lower()
        item_text = " ".join(visitor.vlm.carried_items).lower()
        accessory_text = " ".join(visitor.vlm.accessories).lower()
        if self._is_smiling_expression(expression):
            return "笑顔が見えた瞬間、僕のカメラの目までちょっと明るくなったげん。素敵な笑顔やじー。"
        if visitor.bbox_width_px >= 1200:
            return "近い近い、カメラの目にどーんって来とる。僕、画面の中でちょっと後ずさりしたいげん。"
        if "hands_up" in pose or "pointing" in pose:
            return "おっ、手の動きまで見えたげん。僕の画像AI、急に『反応きた！』ってざわついとる。"
        if "phone" in item_text or "smartphone" in item_text:
            return "スマホが見えた瞬間、僕の画像AIが『撮られる側の準備いる？』って焦っとる。"
        if "pamphlet" in item_text:
            return "パンフレットが見えると、僕の説明モードが勝手に真面目になるげん。"
        if "glasses" in accessory_text:
            return "メガネが見えた瞬間、僕の観察係スイッチが入りすぎて、逆に緊張するげん。"
        if "knit" in clothing or "sweater" in clothing or "cardigan" in clothing:
            return "ニット系を見た瞬間、僕の画像AIが『今日の会場、ちょっとおしゃれやね』って顔しとる。"
        if "backpack" in clothing or "リュック" in clothing:
            return "リュックを見た瞬間、僕の進路相談センサーが勝手に起動したげん。"
        if "suit" in clothing or "スーツ" in clothing:
            return "スーツっぽさを見た瞬間、僕の説明モードが急に背筋伸ばし始めたんや。"
        if "neutral" in expression:
            return "落ち着いた表情で見てくれとるね。僕の冗談が滑る前から採点されとる気がして、ちょっと怖いげん。"
        if "curious" in expression or "excited" in expression:
            return "興味ありそうな雰囲気、画像AIが拾っとる。僕のテンションメーターが勝手に上がるやつや。"
        if audience_type == "group":
            return "複数人になると、僕の画像AIが『誰から見るんや』って軽く慌てるんや。"
        return "画像AIが今、服装と雰囲気を見ながら『失礼にならん程度に面白く言え』って僕に圧をかけとる。"

    def _confidence_note(self, visitor: VisitorInfo | None) -> str:
        if visitor is None:
            return "画像解析はまだ取得前"
        conf = visitor.vlm.expression_confidence
        if conf >= 0.75:
            return "表情や距離感まで使って勢いよく反応する"
        if conf >= 0.4:
            return "服装、持ち物、ポーズを中心に言い切って反応する"
        return "服装や距離感を中心に、会場MCとして明るく言い切る"

    def _is_returning(self, visitor: VisitorInfo, state: AgentState) -> bool:
        if visitor.returning or visitor.visit_count > 1:
            return True
        return False
