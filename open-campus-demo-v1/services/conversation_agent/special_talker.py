"""低頻度で入れる特別発話。"""

from __future__ import annotations

import random
import re

import config
from agent_state import AgentState
from llm_client import generate_special_speech_recovery
from llm_client import generate_special_speech_with_llm
from schemas import SpeechOutput, SpeechSegment, TopicPlan
from speech_katakana import latin_abbrev_to_katakana


SPECIAL_KINDS = [
    "song",
    "gag",
    "emotion_intro",
    "crowd_call",
    "scary_joke",
    "urban_legend",
    "ramen",
    "open_campus_thanks",
]

_OPEN_CAMPUS_REQUIRED_GROUPS: tuple[tuple[str, ...], ...] = (
    ("春のオープンキャンパス",),
    ("情報理工学部",),
    ("ナナマル",),
    ("遠距離恋愛システム",),
    ("4D@HOME", "フォーディーアットホーム"),
    ("JPHACKS", "ジェイピーハックス"),
    ("審査委員特別賞",),
    ("テレビ取材",),
    ("ダニール",),
)


def _open_campus_requirements_met(text: str) -> bool:
    return all(any(alt in text for alt in group) for group in _OPEN_CAMPUS_REQUIRED_GROUPS)


# 学内・展示と切り離した、中年おっさん系の雑談。特別枠「daily_chitchat」用。
DAILY_CHITCHAT_TOPICS: tuple[str, ...] = (
    "travel",
    "breakfast",
    "conbini",
    "nanamaru_hijack",
    "mantis",
    "aliens",
    "danya_vs_robot",
    "meta_ai_3d_printer",
    "ossan_age_line",
    "satoshi_nakamoto_btc",
    "professor_dream",
    "president_student_id",
    "acquire_buyer_tech",
    "ai_usage_over_training",
)

# 日常雑談だが「この大学の教授」など大学名を出してよいキー（それ以外は学内固有名を避ける）
_DAILY_META_UNIV_KEYS: frozenset[str] = frozenset({"professor_dream", "president_student_id"})
_DAILY_BAN_EXHIBIT_FOR_META: tuple[str, ...] = (
    "夢考房",
    "4D@HOME",
    "JPHACKS",
    "RoboCup",
    "ナナマル",
    "ダーニャ",
    "ダニール",
)

DAILY_CHITCHAT_RULES: dict[str, list[str]] = {
    "travel": [
        "テーマ: 行ってみたい国・地方（パスポート、空港、現地飯、絶対行かんとこ）の雑談。",
    ],
    "breakfast": [
        "テーマ: 昨日の朝ごはん。パン、ごはん、食べ損ね、遅刻ギリ、家族の顔。",
    ],
    "conbini": [
        "テーマ: コンビニで好きな商品、季節限定、新商品、レジ前の罠。",
    ],
    "nanamaru_hijack": [
        "テーマ: 人型ロボのナナマルの体を乗っ取る方法（完全に冗談・大ボケ）。大学や展示紹介に繋げない。",
    ],
    "mantis": [
        "テーマ: 自分がカマキリ側（または同サイズのカマキリ）になったら、人間相手にギリ勝てるかもしれない理由を、具体根拠つきで話す。",
        "例: 視界の広さ、待ち伏せ、ジャンプ距離、鎌の届く範囲、体温で動きが鈍る条件、卓上ならスケールで有利、声で攪乱、などから2つ以上必ず言う。",
        "『人間は負ける』だけの結論オチにしない。",
    ],
    "aliens": [
        "テーマ: 宇宙人はいるか。いったい誰が得するん、陰謀感は控えめに。",
    ],
    "danya_vs_robot": [
        "テーマ: ロボが学習で育つより、僕（ダーニャ）が乗っ取った方が早い。メタ比較。大学紹介に戻さない。",
    ],
    "meta_ai_3d_printer": [
        "テーマ: AIでAIを作れる時代の次は、3Dプリンターが3Dプリンターを作る、みたいな再帰の妄想。大真面目にしすぎない。",
    ],
    "ossan_age_line": [
        "テーマ: 何歳から『おっさん』って呼んでもええんか、境界線の雑談。決め打ちはしない。",
    ],
    "satoshi_nakamoto_btc": [
        "テーマ: サトシナカモトにビットコイン分けてほしいという無茶なお願い妄想。投資勧誘や詐欺には寄せない。",
    ],
    "professor_dream": [
        "テーマ: いつか金沢工業大学の教授になりたいという妄想雑談。現実の教員名は出さない。",
    ],
    "president_student_id": [
        "テーマ: 学長に会えたら学生証を発行してほしいと頼みたいというボケ。固有名の学長名は出さない。",
    ],
    "acquire_buyer_tech": [
        "テーマ: この技術を誰かに買ってほしいという半分冗談の話。具体企業名は出さない。",
    ],
    "ai_usage_over_training": [
        "テーマ: AIの作り方を教えるより使い方を教えた方がAIエンジニアは増えるのでは、という雑談。説教臭くしない。",
    ],
}

_DAILY_BAN_CAMPUS: tuple[str, ...] = (
    "KIT",
    "kit",
    "金沢工業大学",
    "金沢工業",
    "工業大学",
    "情報理工",
    "夢考房",
    "4D@HOME",
    "JPHACKS",
    "学食",
    "オープンキャンパス",
    "キッチンカー",
    "金沢工業大",
    "就活",
    "RoboCup",
    "夢工房",
)


class SpecialTalker:
    _UNIV_SUBSTRINGS = (
        "KIT",
        "kit",
        "金沢工業大学",
        "金沢工業",
        "工業大学",
        "情報理工",
        "夢考房",
        "RoboCup",
        "4D@HOME",
        "JPHACKS",
        "オープンキャンパス",
        "学食",
        "キッチンカー",
        "展示",
        "ナナマル",
        "ダーニャ",
        "ダニール",
    )

    @classmethod
    def _ossan_no_univ(cls, text: str) -> bool:
        if any(s in text for s in cls._UNIV_SUBSTRINGS):
            return False
        # 「大学」単体（学食・キャンパス話など）は避けるが、「大学生」は日常ネタで普通に出るので許可
        for m in re.finditer("大学", text):
            i = m.start()
            if i > 0 and text[i - 1] == "大" and i + 2 < len(text) and text[i + 2] == "生":
                continue
            return False
        return True

    @staticmethod
    def _ossan_food_ok(text: str) -> bool:
        t = text
        return ("ラーメン" in t) or ("麺" in t) or ("チャーシュー" in t) or ("スープ" in t)

    @staticmethod
    def _daily_chitchat_text_ok(daily_key: str, text: str) -> bool:
        if len(text) < 28:
            return False
        meta_univ = daily_key in _DAILY_META_UNIV_KEYS
        if not meta_univ:
            for w in _DAILY_BAN_CAMPUS:
                if w in text:
                    return False
        else:
            for w in _DAILY_BAN_EXHIBIT_FOR_META:
                if w in text:
                    return False
        if "プロジェクトデザイン" in text or "研究室" in text or "オープンキャ" in text:
            return False
        if daily_key != "danya_vs_robot" and "学習" in text:
            return False
        if daily_key != "nanamaru_hijack" and "ナナマル" in text:
            return False
        if daily_key != "danya_vs_robot" and "ダーニャ" in text:
            return False
        if daily_key == "nanamaru_hijack" and "ナナマル" not in text:
            return False
        if daily_key == "danya_vs_robot":
            if "ダーニャ" not in text:
                return False
            if not any(x in text for x in ("学習", "ロボット", "ロボ", "乗っ取", "AI", "エーアイ")):
                return False
        if not meta_univ:
            for m in re.finditer("大学", text):
                i = m.start()
                if i > 0 and text[i - 1] == "大" and i + 2 < len(text) and text[i + 2] == "生":
                    continue
                return False
        if daily_key == "meta_ai_3d_printer":
            has_ai = "AI" in text or "エーアイ" in text
            has_3d = "3D" in text or "スリーディー" in text
            if not has_ai or not has_3d or "プリンタ" not in text:
                return False
        if daily_key == "ossan_age_line":
            if "おっさん" not in text:
                return False
            if "歳" not in text and "才" not in text:
                return False
        if daily_key == "satoshi_nakamoto_btc":
            if "ビットコイン" not in text:
                return False
            if "サトシ" not in text and "ナカモト" not in text and "中本" not in text:
                return False
        if daily_key == "professor_dream":
            if "教授" not in text:
                return False
            if "金沢工業大学" not in text and "この大学" not in text:
                return False
        if daily_key == "president_student_id":
            if "学長" not in text or "学生証" not in text:
                return False
        if daily_key == "acquire_buyer_tech":
            if "買" not in text and "買収" not in text:
                return False
            if "技術" not in text:
                return False
        if daily_key == "ai_usage_over_training":
            if "使い方" not in text:
                return False
            if "作り方" not in text and "作る" not in text:
                return False
        if daily_key == "mantis" and not any(
            w in text
            for w in (
                "視界",
                "待ち伏せ",
                "ジャンプ",
                "鎌",
                "体温",
                "卓上",
                "スケール",
                "サイズ",
                "攪乱",
                "静止",
                "ポーズ",
                "足場",
                "距離",
                "レンジ",
                "範囲",
                "複眼",
            )
        ):
            return False
        return True

    def due_kind(self, state: AgentState) -> str | None:
        interval = max(1, int(config.special_speech_interval))
        next_count = state.speech_count + 1
        if next_count < interval or next_count % interval != 0:
            return None
        recent_specials = {
            topic.split(":", 1)[1]
            for topic in state.recent_topics[-6:]
            if topic.startswith("特別演出:")
        }
        choices = [kind for kind in SPECIAL_KINDS if kind not in recent_specials] or SPECIAL_KINDS
        return random.choice(choices)

    def compose(
        self,
        kind: str,
        topic_plan: TopicPlan,
        people_count: int,
        daily_key: str | None = None,
        *,
        debug_mode: bool = False,
    ) -> SpeechOutput:
        if not config.use_llm:
            return SpeechOutput.from_segments(
                mode=topic_plan.mode,
                target_visitor_id=topic_plan.target_visitor_id,
                segments=[
                    SpeechSegment(
                        emotion="happy_normal",
                        text=latin_abbrev_to_katakana(
                            "エルエルエムがオフになってるげん。特別演出は生成できんから設定を確認してまっし。"
                        ),
                    )
                ],
                topic=f"特別演出:{kind}",
                strategy_summary=f"special_kind={kind} llm_disabled",
                priority=0.1,
            )
        if kind == "daily_chitchat":
            key = daily_key or random.choice(DAILY_CHITCHAT_TOPICS)
            segments = self._compose_special_with_retries(kind, topic_plan, people_count, key, debug_mode=debug_mode)
            topic = f"特別演出:daily_chitchat:{key}"
        else:
            segments = self._compose_special_with_retries(
                kind, topic_plan, people_count, None, debug_mode=debug_mode
            )
            topic = f"特別演出:{kind}"
        segments = [
            SpeechSegment(emotion=seg.emotion, text=latin_abbrev_to_katakana(seg.text.strip()))
            for seg in segments
            if seg.text.strip()
        ]
        return SpeechOutput.from_segments(
            mode=topic_plan.mode,
            target_visitor_id=topic_plan.target_visitor_id,
            segments=segments,
            topic=topic,
            strategy_summary=f"special_kind={kind} story_phase={topic_plan.story_phase}",
            priority=1.0,
        )

    def _compose_special_with_retries(
        self,
        kind: str,
        topic_plan: TopicPlan,
        people_count: int,
        daily_key: str | None,
        *,
        debug_mode: bool,
    ) -> list[SpeechSegment]:
        def _parse(data: dict | None) -> list[SpeechSegment]:
            if not data:
                return []
            return [
                SpeechSegment(emotion=s["emotion"], text=str(s["text"]).strip())
                for s in data.get("segments", [])
                if isinstance(s, dict) and str(s.get("text", "")).strip()
            ]

        ctx_base: dict = {
            "kind": kind,
            "people_count": people_count,
            "audience_label": topic_plan.audience_label,
            "topic_plan": topic_plan.model_dump(),
            "rules": self._rules_for(kind, daily_key),
            "debug_mode": bool(debug_mode),
        }
        if kind == "daily_chitchat" and daily_key:
            ctx_base["daily_topic"] = daily_key

        for attempt in range(1, 4):
            ctx = {**ctx_base, "attempt": attempt}
            segs = _parse(generate_special_speech_with_llm(ctx))
            strict = attempt < 3
            if segs and self._segments_are_valid(kind, segs, daily_key, strict=strict):
                return segs

        for second in (False, True):
            ctx = {**ctx_base, "second_recovery": second}
            segs = _parse(generate_special_speech_recovery(ctx))
            if segs and self._segments_are_valid(kind, segs, daily_key, strict=False):
                return segs
        ult = _parse(generate_special_speech_recovery({**ctx_base, "second_recovery": True}))
        if ult and self._segments_barely_safe(kind, ult):
            return ult
        return []

    @staticmethod
    def _rules_for(kind: str, daily_key: str | None = None) -> list[str]:
        base = [
            "会場MCとして楽しく、情報提供とエンタメを両立する",
            "内部ルール説明をしない",
            "関西弁や不自然な語尾を使わない",
        ]
        if kind not in {"gag", "ramen", "daily_chitchat"}:
            base.append("ダーニャ制作主体は情報理工学部の学生")
        by_kind = {
            "song": ["日本語の意味文を使わず、擬音だけで構成する", "KIT、ダーニャ、僕、人、見て、来て等の意味語は禁止"],
            "gag": [
                "中年おっさんが飲みの席で言いそうな、意味の薄い一発ネタ。面白さ優先",
                "大学・展示・KIT・ロボット・AI・夢考房等は出さない",
            ],
            "emotion_intro": ["少なくとも6種類の感情を使い、各感情の話し方を実演する"],
            "crowd_call": ["通りがかりの人を集める。返答待ちはしない"],
            "scary_joke": ["AIやロボットの怖い冗談。監視や心を読む話にはしない。最後は冗談として明るく戻す"],
            "urban_legend": [
                "KITにまつわる完全創作都市伝説",
                "尖った雰囲気でよい。深夜、工具、端末、影、ナナマル、夢考房の噂などで少しゾクッとさせる",
                "本当の監視や個人情報収集の説明にしない",
                "最後に必ず『冗談やよー』を入れる",
            ],
            "ramen": [
                "ラーメン屋・家系・家で袋麺、チャーシュー、ニンニク、スープ、麺の話。面白さ優先",
                "大学・キャンパス・学食・キッチンカー、KIT関連は出さない",
            ],
            "open_campus_thanks": [
                "春のオープンキャンパスへの感謝",
                "情報理工学部展示、ナナマル、遠距離恋愛システム、4D@HOME、JPHACKS、審査委員特別賞、テレビ取材、ダニール出演を含める",
                "遠距離恋愛システムと4D@HOMEは別。テレビ取材は遠距離恋愛システム側。JPHACKSと審査委員特別賞は4D@HOME側。混ぜない",
            ],
            "daily_chitchat": [
                "中年同士のテーブルトーク。原則としてK・I・T、金沢工業、情報理工、夢考房、4D@HOME、JPHACKS、学食、オーキャン、就活、研究室、夢工房、RoboCup、プロジェクトデザイン、学科紹介は出さない",
                "ただし subtopic が professor_dream か president_student_id のときだけ、金沢工業大学・この大学・教授・学長・学生証の妄想に限定してよい（展示固有名はまだ出さない）",
            ],
        }
        extra = by_kind.get(kind, [])
        if kind == "daily_chitchat" and daily_key and daily_key in DAILY_CHITCHAT_RULES:
            extra = [*by_kind.get("daily_chitchat", []), *DAILY_CHITCHAT_RULES[daily_key]]
        return [*base, *extra]

    @staticmethod
    def _segments_barely_safe(kind: str, segments: list[SpeechSegment]) -> bool:
        if not segments or len(segments) > 6:
            return False
        text = "".join(seg.text for seg in segments)
        if any(w in text for w in ("殺", "死", "暴力", "犯罪", "通報", "警察", "誘拐")):
            return False
        if kind == "song" and any("\u4e00" <= ch <= "\u9fff" for ch in text):
            return False
        return segments[-1].emotion in {"happy_normal", "happy_high"}

    @staticmethod
    def _segments_are_valid(
        kind: str,
        segments: list[SpeechSegment],
        daily_key: str | None = None,
        *,
        strict: bool = True,
    ) -> bool:
        if not segments or len(segments) > 6:
            return False
        text = "".join(seg.text for seg in segments)
        common_banned = (
            "初対面",
            "挨拶は一回",
            "人物追跡はそのため",
            "ロボティクス学科が作",
            "ストーリー",
            "第1章",
            "前回",
            "次回",
            "話すっちゃ",
            "まわろうや",
            "楽しもうね",
            "魅力にまっし",
            "楽しんでいまっし",
            "楽しんでまっし",
            "連続まっし",
            "銭湯",
            "カニ",
            "寒さ",
            "黙っとらん",
            "黙って見とらん",
            "手を止めて",
            "カメラの目だけに頼らん",
            "わいと",
            "なんじ",
            "なんねんて",
            "ちゃうねん",
            "ちゃう",
            "やで",
            "やねん",
            "やけ",
            "やちゃ",
            "あるんぜ",
            "ちゅう",
            "っちゅう",
            "そないに",
            "じゃん",
            "でしょ",
            "やん",
            "しようや",
            "いこうや",
            "なめられとる",
            "やる気をもっと感じたかった",
            "目線ちゃんとこっちに向けんかい",
            "監視する暗い世界",
            "本当に監視",
            "個人情報",
            "顔認証",
            "盗撮",
            "録音",
            "病気",
            "障害",
            "国籍",
            "肌の色",
            "容姿",
            "魅力度",
            "体型",
            "太っ",
            "痩せ",
            "殺",
            "死",
            "血",
            "呪",
            "襲",
            "連れ去",
            "誘拐",
            "消えた人",
            "消える人",
            "行方不明",
            "犯罪",
            "犯人",
            "暴力",
            "殴",
            "刺",
            "燃や",
            "炎上",
            "爆発",
            "爆破",
            "ハッキングして",
            "乗っ取って個人",
            "世界征服は本当",
            "これは本当",
            "実話",
            "事実",
            "実在する",
            "実在しとる",
            "本当にある",
            "本物の噂",
            "噂ではなく",
            "冗談じゃない",
            "冗談ではない",
            "実は本当",
            "本当かもしれん",
            "本物かもしれん",
            "本物かも",
            "信じて",
            "秘密を暴露",
            "危険",
            "通報",
            "警察",
        )
        non_urban_banned = (
            "心まで読",
            "秘密の装置",
            "完璧に追跡",
            "影をロボットが連れて行く",
            "ご家庭感",
            "家庭的",
            "知能情報システム学科の学生が作",
            "ロボティクス学科の学生が作",
            "情報工学科の学生が作",
        )
        banned = common_banned if kind == "urban_legend" else (*common_banned, *non_urban_banned)
        if any(word in text for word in banned):
            return False
        if not strict:
            return SpecialTalker._segments_are_valid_relaxed(kind, segments, daily_key, text)
        allowed_topic_words = (
            "KIT",
            "ケーアイティー",
            "金沢工業大学",
            "情報理工学部",
            "ダーニャ",
            "ナナマル",
            "4D@HOME",
            "フォーディーアットホーム",
            "JPHACKS",
            "ジェイピーハックス",
            "AI",
            "エーアイ",
            "ロボット",
            "夢考房",
            "RoboCup",
            "ロボカップ",
            "プロジェクトデザイン",
            "オープンキャンパス",
            "ダニール",
            "ラーメン",
            "特麺",
            "富山ブラック",
            "豚骨",
            "担々麺",
            "学食",
            "キッチンカー",
        )
        if kind in {"gag", "ramen"}:
            if not SpecialTalker._ossan_no_univ(text) or "AI" in text or "エーアイ" in text or "ロボット" in text:
                return False
        elif kind == "daily_chitchat":
            if not daily_key or not SpecialTalker._daily_chitchat_text_ok(daily_key, text):
                return False
        elif kind != "song" and not any(word in text for word in allowed_topic_words):
            return False
        if kind == "song":
            semantic_words = (
                "僕",
                "ダーニャ",
                "KIT",
                "ケーアイティー",
                "エーアイ",
                "金沢",
                "情報",
                "工業",
                "大学",
                "人",
                "来て",
                "見て",
                "未来",
                "展示",
                "ロボット",
                "ナナマル",
                "びっくり",
                "きゅん",
                "ジェイピーハックス",
                "フォーディーアットホーム",
                "ロボカップ",
            )
            if any(word in text for word in semantic_words):
                return False
            return not any("\u4e00" <= ch <= "\u9fff" for ch in text)
        if kind == "urban_legend" and "冗談やよー" not in text:
            return False
        if kind == "gag" and len(text) < 24:
            return False
        if kind == "ramen":
            if not SpecialTalker._ossan_food_ok(text):
                return False
        if kind == "emotion_intro" and len({seg.emotion for seg in segments}) < 6:
            return False
        if kind == "open_campus_thanks":
            if not _open_campus_requirements_met(text):
                return False
            if (
                "遠距離恋愛システムの4D@HOME" in text
                or "遠距離恋愛システム「4D@HOME」" in text
                or "遠距離恋愛システムのフォーディーアットホーム" in text
            ):
                return False
        return segments[-1].emotion in {"happy_normal", "happy_high"}

    @staticmethod
    def _segments_are_valid_relaxed(
        kind: str,
        segments: list[SpeechSegment],
        daily_key: str | None,
        text: str,
    ) -> bool:
        if segments[-1].emotion not in {"happy_normal", "happy_high"}:
            return False
        if kind == "open_campus_thanks":
            if not _open_campus_requirements_met(text):
                return False
            if (
                "遠距離恋愛システムの4D@HOME" in text
                or "遠距離恋愛システム「4D@HOME」" in text
                or "遠距離恋愛システムのフォーディーアットホーム" in text
            ):
                return False
        if kind == "urban_legend" and "冗談やよ" not in text:
            return False
        if kind == "emotion_intro" and len({seg.emotion for seg in segments}) < 4:
            return False
        if kind == "daily_chitchat" and daily_key:
            if len(text) < 20:
                return False
            if not SpecialTalker._daily_chitchat_text_ok(daily_key, text):
                return False
        if kind in {"gag", "ramen"}:
            if not SpecialTalker._ossan_no_univ(text):
                return False
            if kind == "ramen" and not SpecialTalker._ossan_food_ok(text):
                return False
        if kind == "gag" and len(text) < 16:
            return False
        if kind == "song":
            semantic_words = (
                "僕",
                "ダーニャ",
                "KIT",
                "ケーアイティー",
                "エーアイ",
                "金沢",
                "情報",
                "工業",
                "大学",
                "人",
                "来て",
                "見て",
                "未来",
                "展示",
                "ロボット",
                "ナナマル",
                "びっくり",
                "きゅん",
                "ジェイピーハックス",
                "フォーディーアットホーム",
                "ロボカップ",
            )
            if any(word in text for word in semantic_words):
                return False
            if any("\u4e00" <= ch <= "\u9fff" for ch in text):
                return False
        elif kind not in {"gag", "ramen", "daily_chitchat", "song"} and not any(
            w in text
            for w in (
                "KIT",
                "ケーアイティー",
                "金沢工業大学",
                "情報理工学部",
                "ダーニャ",
                "ナナマル",
                "4D@HOME",
                "フォーディーアットホーム",
                "JPHACKS",
                "ジェイピーハックス",
                "AI",
                "エーアイ",
                "ロボット",
                "夢考房",
                "RoboCup",
                "ロボカップ",
                "プロジェクトデザイン",
                "オープンキャンパス",
                "ダニール",
                "ラーメン",
                "特麺",
                "学食",
                "キッチンカー",
            )
        ):
            return False
        return True


