"""計画済みの話題と感情フローから発話セグメントを作る。"""

from __future__ import annotations

import random

import config
from agent_state import AgentState
from llm_client import generate_dedupe_append_segment
from llm_client import generate_normal_speech_recovery
from llm_client import generate_speech_with_llm
from schemas import EmotionPlan, SpeechOutput, SpeechSegment, TopicPlan
from speech_katakana import latin_abbrev_to_katakana


class BehaviorComposer:
    def __init__(self) -> None:
        self.blocked_words = ("体型", "太って", "痩せて", "肌の色", "国籍", "障害", "病気")
        self.style_replacements = {
            "私は": "僕は",
            "私、": "僕、",
            "私も": "僕も",
            "私の": "僕の",
            "私": "僕",
            "わたし": "僕",
            "俺": "僕",
            "こんにちは！": "おっ、",
            "んやで": "んやげんて",
            "なんやで": "なんやげんて",
            "やねん": "ねんて",
            "やで": "ねんて",
            "れるんやげんて": "れるげんて",
            "るんやげんて": "るげんて",
            "さかい": "から",
            "ほんまに": "かなり",
            "ほんま": "かなり",
            "ちゃうねん": "違うげん",
            "ちゃうで": "やないげん",
            "ちゃうかな": "やないかな",
            "ちゃう？": "やない？",
            "ちゃうげん": "やないげん",
            "簡単ちゃう": "簡単やない",
            "ちゃう": "やない",
            "あかん": "だめ",
            "あかんわ": "だめやね",
            "しておる": "している",
            "VLM": "画像AI",
            "思わへん": "思わん",
            "思うわ": "思うげん",
            "やけど": "けど",
            "やけ、": "やから、",
            "やけ": "やから",
            "じゃん": "やね",
            "でしょ": "やろ",
            "そないに": "そんなに",
            "やんね": "やね",
            "やん、": "やね、",
            "やん。": "やね。",
            "やん！": "やね！",
            "ちゅう": "という",
            "っちゅう": "という",
            "あるんぜ": "あるげんて",
            "話やちゃ": "話ねんて",
            "やちゃ": "げんて",
            "頑張るで": "頑張るよ",
            "話すで": "話すよ",
            "すごいで": "すごいげん",
            "やのに": "なのに",
            "あるやね": "あるげん",
            "しようや": "していきまっし",
            "楽しもうや": "楽しんでいきまっし",
            "いこうや": "いきまっし",
            "やなあ": "やね",
            "話すっちゃ": "話すげんて",
            "使っとるがい。": "使っとるげん。",
            "揃っとるがい！": "揃っとるげんて！",
            "学んどるがい！": "学んどるげんて！",
            "大事げん": "大事ねんて",
            "魅力にまっし": "魅力を見てまっし",
            "楽しんでいまっし": "楽しんでいきまっし",
            "楽しもうねー": "楽しんでいきまっし",
            "まわろうや": "まわってみまっし",
            "わいと": "僕と",
            "なんじ！": "ねんて！",
            "なんじ。": "ねんて。",
            "なんじー": "ねんて",
            "なんねんて": "ねんて",
            "怖いげど": "怖いけど",
            "けどプレッシャー": "げんて。プレッシャー",
            "魅力やよ": "魅力があるげんて",
            "未来やよ": "未来につながるげんて",
            "話やよ": "話ねんて",
            "場所やよ": "場所ねんて",
            "技術やよ": "技術ねんて",
            "冗談やよ": "冗談ねんて",
            "やよ。": "ねんて。",
            "やよ！": "ねんて！",
            "やよ、": "ねんて、",
            "やげん": "ねんて",
            "情報理工なら": "情報理工学部なら",
            "情報理工で": "情報理工学部で",
            "まっし伝わ": "ちゃんと伝わ",
            "嬉しいまっし": "嬉しいげん",
            "うれしいまっし": "うれしいげん",
            "怖いまっし": "怖いげん",
            "こわいまっし": "こわいげん",
            "すごいまっし": "すごいげんて",
            "面白いまっし": "面白いげんて",
            "思うまっし": "思うげん",
            "するまっし": "するげん",
            "やよすな": "やね",
            "なんやそれ": "なんでやろ",
            "dark sweater": "暗めのセーター",
            "sweater": "セーター",
            "brown cardigan": "茶色のカーディガン",
            "black shirt": "黒いシャツ",
            "brown cardigan over a black shirt": "茶色のカーディガンに黒いシャツ",
            "casual sweater": "カジュアルなセーター",
            "knit sweater": "ニットの服",
            "knitted garment": "ニットの服",
            "looking_down": "手元か画面を見ている",
            "neutral": "落ち着いた表情",
            "unknown": "",
            "断定できないけど": "",
            "断定できない": "",
            "半信半疑": "",
            "自信ない": "",
            "自信がない": "",
            "推定は弱め": "",
            "断定しすぎない": "",
            "断定せん": "",
            "断定はしとらん": "",
            "断定しとらん": "",
            "完璧ちゃう": "",
            "完璧じゃない": "",
            "完璧ではない": "",
            "遠慮しとる": "",
            "慎重さ": "",
            "学生さんっぽい": "学生さん",
            "高校生っぽい": "高校生くらい",
            "保護者の方かな": "保護者の方",
            "学生さんかな": "学生さん",
            "お偉いさんかな": "お偉いさん",
            "友達グループかな": "友達グループ",
            "親子かな": "親子",
            "お一人さんかな": "お一人さん",
            "情報理工学部知能情報システム学科の学生が作": "情報理工学部の学生が作",
            "情報理工学部ロボティクス学科の学生が作": "情報理工学部の学生が作",
            "情報理工学部情報工学科の学生が作": "情報理工学部の学生が作",
            "スターやまっし": "スターねんて",
            "見ていまっし": "見てまっし",
            "楽しみまっし": "楽しんで行きまっし",
            "楽しんでまっし": "楽しんで行きまっし",
            "まっしで来て": "来て",
            "わあ、まっしで": "わあ、",
            "目線ちゃんとこっちに向けんかいね": "こっちもちらっと見てまっし",
            "なめられとるやろ": "まだ本気出してないやろ",
            "やる気をもっと感じたかった": "もう少しだけこっちも見てほしい",
            "ご家庭感": "会場の空気",
            "家庭的な雰囲気": "リラックスした雰囲気",
            "夢考房の授業": "夢考房",
        }

    def _sanitize_speech(self, speech: str) -> str:
        out = speech
        for word in self.blocked_words:
            out = out.replace(word, "雰囲気")
        for old, new in self.style_replacements.items():
            out = out.replace(old, new)
        return latin_abbrev_to_katakana(out)

    def _llm_output_is_usable(
        self,
        segments: list[SpeechSegment],
        topic_plan: TopicPlan,
        *,
        debug_mode: bool = False,
    ) -> bool:
        text = "".join(seg.text for seg in segments)
        if any(w in text for w in ("バックモード", "デバッグモード", "デバッグ用セッション")):
            return False
        if debug_mode and any(w in text for w in ("人物追跡で探", "内部仕様", "運用ルール")):
            return False
        banned = (
            "今日はまた来て",
            "また来てくれて",
            "戻ってきて",
            "再訪",
            "やで",
            "やねん",
            "やけ",
            "さかい",
            "あるやね",
            "あるんぜ",
            "やちゃ",
            "ちゅう",
            "っちゅう",
            "そないに",
            "じゃん",
            "でしょ",
            "やん",
            "やよすな",
            "まっし伝わ",
            "嬉しいまっし",
            "うれしいまっし",
            "思うまっし",
            "怖いまっし",
            "こわいまっし",
            "すごいまっし",
            "面白いまっし",
            "いこうや",
            "しようや",
            "やなあ",
            "思わへん",
            "ちゃうねん",
            "あかん",
            "あかんわ",
            "さかい",
            "そう思わん",
            "聞いたことある",
            "話すっちゃ",
            "まわろうや",
            "楽しもうね",
            "魅力にまっし",
            "楽しんでいまっし",
            "連続まっし",
            "楽しんでまっし",
            "まっしで来て",
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
            "初対面",
            "挨拶",
            "同じ場",
            "人物追跡って",
            "人物追跡はそのため",
            "何回も",
            "一回でええ",
            "一回でいい",
            "二度挨拶",
            "2度挨拶",
            "内部ルール",
            "運用ルール",
            "ストーリー",
            "第1章",
            "第2章",
            "第3章",
            "今回は『",
            "今回は「",
            "次は『",
            "次は「",
            "さっきの『",
            "さっきの「",
            "誰も見えん",
            "誰も見えない",
            "誰も立ち止まらん",
            "まだ誰も立ち止ま",
            "誰も捕まえとらん",
            "凍える",
            "ちゃうで",
            "ちゃう",
            "ちゃうげん",
            "簡単ちゃう",
            "dark sweater",
            "neutral",
            "unknown",
            "断定できない",
            "断定せん",
            "断定はしとらん",
            "断定しとらん",
            "完璧ちゃう",
            "完璧じゃない",
            "完璧ではない",
            "完璧ではありません",
            "半信半疑",
            "自信ない",
            "自信がない",
            "推定は弱め",
            "断定しすぎない",
            "誤った情報",
            "間違った情報",
            "たまに外す",
            "外すこと",
            "外れること",
            "外れる場合",
            "慎重",
            "遠慮",
            "見守っとって",
            "見守って",
            "寄りにしか",
            "断定",
            "なめられとる",
            "やる気をもっと感じたかった",
            "目線ちゃんとこっちに向けんかい",
            "監視する暗い世界",
            "心まで読",
            "秘密の装置",
            "完璧に追跡",
            "影をロボットが連れて行く",
            "ご家庭感",
            "家庭的",
            "特定学科の学生が作",
            "知能情報システム学科の学生が作",
            "ロボティクス学科の学生が作",
            "情報工学科の学生が作",
        )
        if not topic_plan.is_returning and any(word in text for word in banned[:4]):
            return False
        return not any(word in text for word in banned[4:])

    def compose(
        self,
        topic_plan: TopicPlan,
        emotion_plan: EmotionPlan,
        rag_results: list[dict],
        state: AgentState,
        *,
        long_form: bool,
        debug_mode: bool = False,
    ) -> SpeechOutput:
        spoken_before = bool(
            topic_plan.target_visitor_id
            and state.recent_visitor_topics(topic_plan.target_visitor_id)
        )
        if not config.use_llm:
            return SpeechOutput.from_segments(
                mode=topic_plan.mode,
                target_visitor_id=topic_plan.target_visitor_id,
                segments=[
                    SpeechSegment(
                        emotion="happy_normal",
                        text="エルエルエムがオフになってるげん。テンプレ強制の環境変数を外すか、設定を確認してまっし。",
                    )
                ],
                topic=topic_plan.primary_topic,
                strategy_summary=self._summary(topic_plan, emotion_plan, False, False),
                priority=0.1,
            )

        base_llm_context = {
            "topic_plan": topic_plan.model_dump(),
            "emotion_plan": emotion_plan.model_dump(),
            "spoken_before_to_this_visitor": spoken_before,
            "required_segment_count": len(emotion_plan.flow),
            "must_use_emotions_in_order": emotion_plan.flow,
            "long_form": long_form,
            "max_total_chars": config.long_speech_max_chars if long_form else config.short_speech_max_chars,
            "max_segment_chars": 95 if long_form else 60,
            "rag_results": rag_results[:5],
            "recent_topics": [] if debug_mode else state.recent_topics[-8:],
            "recent_openings": [] if debug_mode else state.recent_openings[-8:],
            "recent_speeches": [] if debug_mode else state.recent_speeches[-4:],
            "people_count": topic_plan.people_count,
            "debug_mode": debug_mode,
            "repeated_same_person": spoken_before,
            "story_phase": topic_plan.story_phase,
            "conversation_structure_hint": (
                "観察は audience_label の人物に1回だけ触れ、続けてケーアイティーの具体情報、最後は明るく締める（別の呼びかけ先を増やさない）。"
                if debug_mode
                else self._structure_hint(long_form)
            ),
            "character": (
                "元気で感情豊か、少し生意気、親しみやすい。軽い上から目線、自慢、嫉妬、ツッコミで場を動かす。"
                "ケーアイティー在籍のダニールの顔と声を借りている。語尾は金沢弁を主軸（げんて・げん・じー・がいね・しとる）。です・ます調で締めない。"
                "画像のエーアイの観察をユーモラスにいじる。会場の人へ反応してから情報を出す。感情は_highタグを活かして端まで振る。"
                "嫌味や説教ではなく、最後は必ず明るく褒める。来場への誘いは必ず『楽しんで行きまっし』。『楽しんでまっし』は使わない。"
                "発話本文にラテン字の略語は書かない。ケーアイティー、エーアイ、エルエルエム、ジェイピーハックス等は必ずカタカナ読み。"
            ),
            "safety": (
                "体型、容姿の良し悪し、肌の色、国籍、障害、病気には触れない。"
                "服装、持ち物、ポーズ、人数、立ち位置には安全に触れてよい。"
                "topic_plan.use_visual_detailがtrueのときは、visual_hook や vlm の手がかりをまとめ、"
                "『黒いシャツを着た青年みたいな雰囲気』『スーツの男性の方』『女性らしい雰囲気でニットの服』のように、"
                "服装と年代・性別の雰囲気が伝わる言い方で最低1回触れる（断定の押しつけは避け、『〜みたいな雰囲気』も可）。"
                "相手の返答を待たない。質問で終わらない。"
                "spoken_before_to_this_visitorがtrueなら挨拶や自己紹介を繰り返さない。"
                "spoken_before_to_this_visitorがtrueなら毎回『そこの〇〇』と呼びかけない。"
                "初対面、挨拶は一回、人物追跡はそのため、同じ場にいる、などの内部ルール説明を言わない。"
            ),
        }

        def _try_llm_data(llm_data: dict | None, *, relaxed_coverage: bool) -> SpeechOutput | None:
            if not llm_data:
                return None
            raw_segments = [
                SpeechSegment(emotion=s["emotion"], text=str(s["text"]))
                for s in llm_data.get("segments", [])
                if isinstance(s, dict)
            ]
            if raw_segments and not self._llm_output_is_usable(raw_segments, topic_plan, debug_mode=debug_mode):
                raw_segments = []
            segments = [
                SpeechSegment(emotion=seg.emotion, text=self._sanitize_speech(seg.text))
                for seg in raw_segments
            ]
            if not segments:
                return None
            if not self._valid_emotion_coverage(segments, long_form, relaxed=relaxed_coverage):
                return None
            if not self._llm_output_is_usable(segments, topic_plan, debug_mode=debug_mode):
                return None
            return SpeechOutput.from_segments(
                mode=topic_plan.mode,
                target_visitor_id=topic_plan.target_visitor_id,
                segments=segments,
                topic=topic_plan.primary_topic,
                strategy_summary=self._summary(topic_plan, emotion_plan, True, False),
                priority=0.9 if long_form else 0.5,
            )

        for attempt in range(1, 4):
            llm_context = {**base_llm_context, "attempt": attempt}
            out = _try_llm_data(generate_speech_with_llm(llm_context), relaxed_coverage=False)
            if out:
                return out

        out = _try_llm_data(generate_normal_speech_recovery(base_llm_context), relaxed_coverage=True)
        if out:
            return SpeechOutput.from_segments(
                mode=out.mode,
                target_visitor_id=out.target_visitor_id,
                segments=out.segments,
                topic=out.topic,
                strategy_summary=f"{out.strategy_summary} recovery=True",
                priority=out.priority * 0.85,
            )
        out2 = _try_llm_data(
            generate_normal_speech_recovery({**base_llm_context, "second_recovery": True}),
            relaxed_coverage=True,
        )
        if out2:
            return SpeechOutput.from_segments(
                mode=out2.mode,
                target_visitor_id=out2.target_visitor_id,
                segments=out2.segments,
                topic=out2.topic,
                strategy_summary=f"{out2.strategy_summary} recovery=second",
                priority=out2.priority * 0.8,
            )

        return SpeechOutput.from_segments(
            mode=topic_plan.mode,
            target_visitor_id=topic_plan.target_visitor_id,
            segments=[
                SpeechSegment(
                    emotion="happy_normal",
                    text="接続が不安定で喋れんかったげん。少ししてからまた来てまっし。",
                )
            ],
            topic=topic_plan.primary_topic,
            strategy_summary=self._summary(topic_plan, emotion_plan, False, True),
            priority=0.15,
        )

    @staticmethod
    def _structure_hint(long_form: bool) -> str:
        if not long_form:
            return random.choice(
                [
                    "一言ツッコミから始め、すぐ情報か感情で締める",
                    "会場への呼びかけから始め、短くダーニャの感情を出す",
                    "いきなり情報を出してから、最後に軽い自虐で落とす",
                    "少し生意気な呼び込みから始め、最後に明るく褒める",
                ]
            )
        return random.choice(
            [
                "視覚観察→短い情景表現→ケーアイティー／ダーニャ情報→明るい締め",
                "いきなりKIT/ダーニャ情報→会場観察→自虐→明るい締め",
                "感情の叫び→視覚観察→具体情報→詩的な一言",
                "会場呼び込み→具体情報→画像AIのユーモア→明るい締め",
                "自虐や嫉妬→具体情報→目の前の人への反応→前向きな締め",
                "少し生意気なツッコミ→具体情報→短い情景表現→明るい締め",
            ]
        )

    def _valid_emotion_coverage(
        self, segments: list[SpeechSegment], long_form: bool, *, relaxed: bool = False
    ) -> bool:
        if relaxed:
            if not segments or len(segments) > 4:
                return False
            return segments[-1].emotion in {"happy_normal", "happy_high"}
        if not long_form:
            return len(segments) <= 2
        if len(segments) > 4:
            return False
        text_len = sum(len(seg.text) + len(seg.emotion) + 2 for seg in segments)
        if text_len > config.long_speech_max_chars:
            return False
        if len({seg.emotion for seg in segments}) < 3:
            return False
        return segments[-1].emotion in {"happy_normal", "happy_high"}

    def _summary(
        self,
        topic_plan: TopicPlan,
        emotion_plan: EmotionPlan,
        llm_success: bool,
        fallback_used: bool,
    ) -> str:
        return (
            f"audience={topic_plan.audience_type} topic={topic_plan.primary_topic} "
            f"flow={'>'.join(emotion_plan.flow)} llm_success={llm_success} fallback={fallback_used}"
        )

