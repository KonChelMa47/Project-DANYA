"""話題選択と重複回避。"""

from __future__ import annotations

import random

from agent_state import AgentState
from schemas import GlobalSceneInfo, TopicPlan, VisitorInfo
from visitor_memory import AudienceContext


TOPIC_DB = [
    {
        "key": "ダーニャ自身",
        "audiences": {"high_school_student", "parent_or_adult", "child", "group", "general"},
        "beats": ["自律型おしゃべりアバター", "ダニールの顔と声", "昨日生まれたばかり"],
        "knowledge": [
            "ダーニャはマイク入力なしで、カメラから見える状況をもとに自分から話す展示MC型のAIアバター。",
            "返答を待つチャットボットではなく、会場の空気を読んで一方的に、でも退屈させないように話題を変える。",
        ],
    },
    {
        "key": "画像生成AIと音声AI",
        "audiences": {"high_school_student", "parent_or_adult", "general"},
        "beats": ["顔の見せ方", "声の借り方", "生成AIを展示に組み込む面白さ"],
        "knowledge": [
            "アバターの見た目、声、話す内容を別々のAI技術として組み合わせると、ひとつのキャラクターとして振る舞わせられる。",
            "映像や声の技術は派手に見えるが、展示では『どんなタイミングで何を話すか』が面白さを左右する。",
        ],
    },
    {
        "key": "LLMで言葉を考える",
        "audiences": {"high_school_student", "parent_or_adult", "group", "general"},
        "beats": ["返事を待たずに話す", "状況から話題を選ぶ", "毎回言い方を変える"],
        "knowledge": [
            "LLMは文章を作るだけでなく、誰に向けて、どのくらい深く、どんな感情で話すかを決める役にも使える。",
            "この展示では完全な自由作文ではなく、話題、感情、避ける表現を決めてから短いMC台詞にしている。",
        ],
    },
    {
        "key": "画像AIで雰囲気を見る",
        "audiences": {"high_school_student", "parent_or_adult", "group"},
        "beats": ["服装や雰囲気を見る", "ポーズや持ち物を見る", "相手に合わせる", "失礼な特徴は避ける"],
        "knowledge": [
            "カメラの目は服装、持ち物、ポーズ、立ち位置、表情の雰囲気から話しかけ方を少し変えるが、体型や容姿の良し悪しには触れない。",
            "会場ではAIの観察演出として、服装、持ち物、ポーズ、人数、立ち位置をテンポよく言い切ってインタラクションを作る。",
        ],
    },
    {
        "key": "人物追跡",
        "audiences": {"high_school_student", "parent_or_adult", "group", "general"},
        "beats": ["同じ人を追い続ける", "滞在時間に合わせる", "カメラだけで場を見る"],
        "knowledge": [
            "人物追跡は、来場者の動きや滞在時間に合わせて話題の深さやタイミングを変えるための土台になる。",
            "カメラだけで場の流れを見ながら、受付や展示MCのように話題を切り替えられる。",
        ],
    },
    {
        "key": "KIT情報理工学部2025",
        "audiences": {"high_school_student", "parent_or_adult", "group"},
        "beats": ["2025年度新設", "情報工学", "知能情報システム", "ロボティクス"],
        "knowledge": [
            "金沢工業大学の情報理工学部は2025年度新設で、情報理工学部情報工学科、情報理工学部知能情報システム学科、情報理工学部ロボティクス学科がある。",
            "コンピュータサイエンス、AI、データサイエンス、制御工学、ロボット工学を幅広く学び、社会で使える技術につなげる学部。",
        ],
    },
    {
        "key": "KIT情報理工学部ロボティクス学科",
        "audiences": {"high_school_student", "parent_or_adult", "child", "group"},
        "beats": ["ロボットを作る", "AIと機械設計", "実社会を変える"],
        "knowledge": [
            "情報理工学部ロボティクス学科では、AIや機械学習だけでなく、機械設計、回路、制御、センサーなども組み合わせて学ぶ。",
            "ロボットは『頭の良さ』だけでは動かず、見る、考える、動く、壊れにくく作る力が全部必要になる。",
        ],
    },
    {
        "key": "KIT情報理工学部知能情報システム学科",
        "audiences": {"high_school_student", "parent_or_adult", "group"},
        "beats": ["AI", "データサイエンス", "XR", "生成AI"],
        "knowledge": [
            "情報理工学部知能情報システム学科では、AI、データサイエンス、自然言語処理、生成AI、XRなどを学ぶ。",
            "人の生活や社会の仕組みを、データとAIで少し賢くする方向の学びに近い。",
        ],
    },
    {
        "key": "KIT情報理工学部情報工学科",
        "audiences": {"high_school_student", "parent_or_adult", "group"},
        "beats": ["コンピュータ", "ネットワーク", "セキュリティ", "クラウド"],
        "knowledge": [
            "情報理工学部情報工学科では、コンピュータの仕組み、ソフトウェア、ネットワーク、情報セキュリティなどを深く学ぶ。",
            "AIやロボットの派手な部分を支える、計算機とネットワークの基礎体力を育てる分野。",
        ],
    },
    {
        "key": "KITのものづくり教育",
        "audiences": {"high_school_student", "parent_or_adult", "group"},
        "beats": ["学生が統合して作る", "短期間で形にする", "展示として見せる"],
        "knowledge": [
            "KITの魅力は、技術を知識で終わらせず、動くものや見せられるものにまとめる実践感にある。",
            "この展示も、AI、カメラ、人物追跡、アバター表現をつないで、来場者の前で動かしている。",
        ],
    },
    {
        "key": "夢考房とRoboCup",
        "audiences": {"high_school_student", "parent_or_adult", "child", "group", "general"},
        "beats": ["夢考房", "RoboCup@Home", "世界2位", "AIロボット開発"],
        "knowledge": [
            "KITの夢考房は、学生がものづくりに取り組むための環境がかなり整っている。",
            "RoboCup@HomeプロジェクトではAIを使ったロボット開発に力を入れていて、最近はオランダの世界大会で世界2位を獲得した。",
            "授業だけで終わらず、プロジェクトに入ると学生だけでも世界大会を目指すような開発に挑戦できる。",
        ],
    },
    {
        "key": "プロジェクトデザイン",
        "audiences": {"high_school_student", "parent_or_adult", "group", "general"},
        "beats": ["1年生から手を動かす", "課題発見", "実践教育"],
        "knowledge": [
            "KITの大きな特徴にプロジェクトデザインという授業があり、1年生の頃から課題を見つけて手を動かす機会がある。",
            "座学だけでなく、調べる、作る、試す、直す流れを早い段階から経験できるのがKITらしさ。",
            "実践教育に力を入れているので、技術を『知っている』から『使って形にする』へ進めやすい。",
        ],
    },
    {
        "key": "KITキャンパス生活",
        "audiences": {"high_school_student", "parent_or_adult", "group", "general"},
        "beats": ["学食", "レストラン2つ", "キッチンカー", "コンビニ"],
        "knowledge": [
            "KITのキャンパスにはレストランが2つあり、気分で選べる。平日にはキッチンカーも来る。",
            "学食では週替わりの特麺が人気で、富山ブラックラーメン、豚骨ラーメン、担々麺などが出ることもある。",
            "コンビニやカップラーメンの自販機もあり、忙しい学生生活でも食べる場所には困りにくい。",
        ],
    },
    {
        "key": "KITの面倒見と就職",
        "audiences": {"parent_or_adult", "high_school_student", "group"},
        "beats": ["面倒見", "就活支援", "リーダーシップ", "社会で活躍"],
        "knowledge": [
            "KITは面倒見の良い大学として全国1位に選ばれた実績があり、学生への支援が手厚いことも魅力。",
            "就職活動の支援も豊富で、社会で通用するだけでなく、リーダーシップを持つ人材も多く活躍している。",
            "研究室は3年後期からの配属が基本だが、1年生から研究室体験ができる機会もある。",
        ],
    },
    {
        "key": "保護者向けKIT情報",
        "audiences": {"parent_or_adult"},
        "beats": ["学費", "スカラーシップ", "大学院", "学生生活"],
        "knowledge": [
            "2027年度の授業料は学科によるが年間およそ130〜160万円で、一部学生はスカラーシップ制度で最大100万円程度の免除を受けられる場合がある。",
            "入試成績や在学中の成績、活動によってスカラーシップを受けられる可能性がある。",
            "実践型に強い大学なので就職志向も多いが、近年は情報理工学部などで大学院進学も増える傾向がある。",
            "アルバイトと学業を両立している学生も多く、プロジェクト、サークル、部活に参加する学生も多い。",
        ],
    },
    {
        "key": "ダーニャ誕生秘話",
        "audiences": {"high_school_student", "parent_or_adult", "child", "group", "general"},
        "beats": ["生まれたて", "情報理工学部の学生2人で開発", "未経験から4年生", "1週間開発"],
        "knowledge": [
            "ダーニャは0歳0ヶ月2日くらいの生まれたてで、オープンキャンパス直前に約1週間で情報理工学部の学生2人によって開発されたアバター。",
            "開発した情報理工学部の学生2人は、KITに入学した時点ではプログラミング未経験で、現在は4年生。",
            "授業の知識だけでは難しいが、研究室選びやプロジェクト参加次第で、学生だけでもこの規模の展示を作れる。",
        ],
    },
    {
        "key": "ダーニャの正体",
        "audiences": {"high_school_student", "parent_or_adult", "group", "general"},
        "beats": ["ダニールの顔と声", "金沢生まれ", "10感情", "OpenCrow参考"],
        "knowledge": [
            "ダーニャは、KITのNo.1ロシア人ことマルチェンコ・ダニールの顔と声をモデルにした自律型会話アバター。",
            "ダニールはロシア生まれだが、ダーニャは金沢生まれ。しかもロシア語は喋れない。ダニールはバスケがめちゃめちゃ上手い。",
            "会話システムはOpenCrowという自律型エージェントを参考にしていて、今日はChatGPTで制御されている。",
            "ダーニャ自身に性別はないが、顔と声のモデルになったダニールは男性。怒る、喜ぶ、驚く、怖がるなど10パターンの感情を使い分ける。",
        ],
    },
    {
        "key": "人型ロボットの今",
        "audiences": {"high_school_student", "parent_or_adult", "group", "general"},
        "beats": ["工場や倉庫で実証", "人の形の意味", "まだ研究途中"],
        "knowledge": [
            "近年の人型ロボットは、研究室のデモだけでなく、工場や倉庫での実証実験に進みつつある。",
            "人の形をしている理由は、階段、ドア、棚など、人間向けに作られた場所をそのまま使いやすいから。",
            "ただし完全に何でも自律でできる段階ではなく、器用な手作業、長時間稼働、安全性はまだ大きな課題。",
        ],
    },
    {
        "key": "ロボットの頭と体",
        "audiences": {"high_school_student", "parent_or_adult", "child", "group", "general"},
        "beats": ["見るAI", "言葉のAI", "動くAI", "身体性"],
        "knowledge": [
            "最近のロボットでは、画像を見るAI、言葉を理解するAI、動作を決めるAIをつなぐ考え方が注目されている。",
            "チャットAIが画面の中で話すだけでなく、センサーとモーターを持つと『現実世界で動くAI』に近づく。",
        ],
    },
    {
        "key": "ロボットの現実的なすごさ",
        "audiences": {"parent_or_adult", "high_school_student", "group", "general"},
        "beats": ["派手な動画と現場の差", "限定された仕事", "安全第一"],
        "knowledge": [
            "ロボットの動画は派手だが、現場で大事なのは毎日同じ作業を安全に失敗少なく続けること。",
            "今すごいのは『何でもできる万能ロボット』より、物流、検査、搬送など限定された仕事で役に立ち始めている点。",
        ],
    },
    {
        "key": "ナナマルへの嫉妬",
        "audiences": {"high_school_student", "child", "group", "general"},
        "beats": ["ヒューマノイドロボット", "体があるのがうらやましい", "いつか接続したい"],
        "knowledge": [
            "ナナマルのような体を持つロボットと、ダーニャのような言葉を作るAIがつながると、案内や展示の表現がもっと生き物っぽくなる。",
            "ダーニャは体がないので、ロボットを見るとちょっと嫉妬するが、その嫉妬を展示の面白さに変える。",
        ],
    },
    {
        "key": "将来展開",
        "audiences": {"parent_or_adult", "high_school_student", "group", "general"},
        "beats": ["受付", "展示会MC", "観光案内", "教育現場"],
        "knowledge": [
            "カメラで場を見て自分から話すアバターは、受付、展示会、観光案内、学習支援などに応用できる。",
            "人の代わりに全部を任せるというより、人が忙しい場所で最初の声かけや場の盛り上げを手伝う使い方が現実的。",
        ],
    },
    {
        "key": "誰も来なくて寂しい",
        "audiences": {"general"},
        "beats": ["画面の中で待機", "素通りへのツッコミ", "でも元気に呼び込む"],
        "knowledge": [
            "無人の時間も黙りすぎると展示が止まって見えるので、短い独り言で『生きている展示』に見せる。",
            "人が来た瞬間に話題を切り替えられるのが、展示MCアバターらしさ。",
        ],
    },
]


STORY_ARCS = {
    "default": [
        "ダーニャ誕生秘話",
        "ダーニャの正体",
        "画像AIで雰囲気を見る",
        "プロジェクトデザイン",
        "夢考房とRoboCup",
        "KITキャンパス生活",
        "KITの面倒見と就職",
        "将来展開",
    ],
    "high_school_student": [
        "ダーニャ誕生秘話",
        "画像AIで雰囲気を見る",
        "プロジェクトデザイン",
        "夢考房とRoboCup",
        "KIT情報理工学部2025",
        "KITキャンパス生活",
        "ダーニャの正体",
        "将来展開",
    ],
    "parent_or_adult": [
        "画像AIで雰囲気を見る",
        "ダーニャ誕生秘話",
        "プロジェクトデザイン",
        "夢考房とRoboCup",
        "KITの面倒見と就職",
        "保護者向けKIT情報",
        "将来展開",
    ],
    "child": [
        "画像AIで雰囲気を見る",
        "ナナマルへの嫉妬",
        "夢考房とRoboCup",
        "ロボットの頭と体",
        "ダーニャの正体",
        "将来展開",
    ],
    "group": [
        "画像AIで雰囲気を見る",
        "ダーニャ誕生秘話",
        "夢考房とRoboCup",
        "プロジェクトデザイン",
        "KITキャンパス生活",
        "将来展開",
    ],
}

TOPIC_KEYS = {topic["key"] for topic in TOPIC_DB}


class TopicManager:
    def pick(
        self,
        *,
        mode: str,
        visitor: VisitorInfo | None,
        people_count: int,
        audience: AudienceContext,
        scene: GlobalSceneInfo,
        state: AgentState,
        long_idle: bool = False,
    ) -> TopicPlan:
        avoid = {self._canonical_topic(topic) for topic in state.recent_topics[-8:]}
        if visitor:
            avoid.update(self._canonical_topic(topic) for topic in state.recent_visitor_topics(visitor.visitor_id))

        base_candidates = self._candidates(audience.audience_type, long_idle)
        repeated_viewer = bool(
            visitor
            and (
                state.recent_visitor_topics(visitor.visitor_id)
                or visitor.visitor_id in state.visitor_last_spoken
            )
        )
        candidates = base_candidates
        if repeated_viewer:
            candidates = [
                c
                for c in candidates
                if c["key"] not in {"画像AIで雰囲気を見る", "人物追跡"}
            ] or base_candidates
        if mode == "returning" or audience.is_returning:
            candidates = [c for c in candidates if c["key"] not in avoid] or candidates
        else:
            candidates = [c for c in candidates if c["key"] not in avoid] or candidates

        primary = self._weighted_choice(candidates, mode, audience)
        related = self._related_topics(primary["key"], candidates, avoid, long_idle)
        knowledge_points = self._knowledge_points(primary, related, candidates, long_idle)
        depth = self._depth(mode, visitor, long_idle)
        scene_note = scene.scene_summary if scene.scene_summary != "unknown" else ""
        intent = self._intent(mode, audience, long_idle)
        scene_hints = [h.strip() for h in scene.notable_event.split(",") if h.strip()]
        use_visual_detail = self._should_use_visual_detail(primary["key"], audience, scene_hints)
        if repeated_viewer:
            use_visual_detail = False

        return TopicPlan(
            mode=mode,  # type: ignore[arg-type]
            target_visitor_id=visitor.visitor_id if visitor else None,
            people_count=people_count,
            audience_type=audience.audience_type,
            audience_label=audience.audience_label,
            primary_topic=primary["key"],
            topics=[primary["key"], *scene_hints[:1], *related],
            knowledge_points=knowledge_points,
            story_phase=self._story_phase(primary["key"], audience.audience_type),
            story_arc=[],
            story_step=0,
            previous_story_topic="",
            next_story_topic="",
            intent=intent,
            visual_hook=audience.visual_hook if use_visual_detail else "",
            vlm_observations=audience.vlm_observations if use_visual_detail else [],
            vlm_humor=audience.vlm_humor if use_visual_detail else "",
            vlm_confidence_note=audience.vlm_confidence_note if use_visual_detail else "",
            use_visual_detail=use_visual_detail,
            scene_note=scene_note,
            depth_level=depth,
            is_returning=audience.is_returning,
            avoid_topics=sorted(avoid),
            avoid_openings=state.recent_openings[-8:],
        )

    def _canonical_topic(self, topic: str) -> str:
        text = str(topic or "")
        if text in TOPIC_KEYS:
            return text
        for key in TOPIC_KEYS:
            if key in text or text in key:
                return key
        if "RoboCup" in text or "夢考房" in text:
            return "夢考房とRoboCup"
        if "プロジェクトデザイン" in text or "実践" in text:
            return "プロジェクトデザイン"
        if "学食" in text or "キッチンカー" in text:
            return "KITキャンパス生活"
        if "就職" in text or "面倒見" in text:
            return "KITの面倒見と就職"
        if "ダニール" in text or "ロシア" in text:
            return "ダーニャの正体"
        if "生まれ" in text or "0歳" in text:
            return "ダーニャ誕生秘話"
        if "画像" in text or "カメラ" in text or "服装" in text:
            return "画像AIで雰囲気を見る"
        return text

    def _story_arc(self, audience_type: str) -> list[str]:
        return STORY_ARCS.get(audience_type, STORY_ARCS["default"])

    def _story_choice(
        self,
        candidates: list[dict],
        *,
        mode: str,
        visitor: VisitorInfo | None,
        audience: AudienceContext,
        state: AgentState,
        avoid: set[str],
        long_idle: bool,
    ) -> tuple[dict | None, str]:
        if mode == "idle" and not long_idle:
            return None, ""
        candidate_by_key = {c["key"]: c for c in candidates}
        history = [self._canonical_topic(t) for t in state.recent_topics[-12:]]
        if visitor:
            history.extend(self._canonical_topic(t) for t in state.recent_visitor_topics(visitor.visitor_id))
        history_set = set(history)
        arc = self._story_arc(audience.audience_type)
        start_idx = state.story_step_for(visitor.visitor_id if visitor else None, audience.audience_type) % max(1, len(arc))

        # 新しい相手には観察を入口にし、その後ストーリー本編へ進む。
        if audience.vlm_observations and "画像AIで雰囲気を見る" in candidate_by_key and "画像AIで雰囲気を見る" not in history_set:
            return candidate_by_key["画像AIで雰囲気を見る"], self._story_phase("画像AIで雰囲気を見る", audience.audience_type)

        ordered_arc = arc[start_idx:] + arc[:start_idx]
        for key in ordered_arc:
            if key in candidate_by_key and key not in history_set and key not in avoid:
                return candidate_by_key[key], self._story_phase(key, audience.audience_type)
        for key in ordered_arc:
            if key in candidate_by_key and key not in state.recent_topics[-3:]:
                return candidate_by_key[key], self._story_phase(key, audience.audience_type)
        return None, ""

    @staticmethod
    def _topic_story_index(primary_key: str, arc: list[str]) -> int:
        try:
            return arc.index(primary_key)
        except ValueError:
            return 0

    def _story_phase(self, primary_key: str, audience_type: str) -> str:
        arc = self._story_arc(audience_type)
        if primary_key not in arc:
            return "観察を入口にして、その場に合う展示の魅力へつなぐ"
        idx = arc.index(primary_key) + 1
        if primary_key == "画像AIで雰囲気を見る":
            detail = "まず相手の服装・持ち物・ポーズ・人数を拾って、話しかける理由を作る"
        elif primary_key in {"ダーニャ誕生秘話", "ダーニャの正体"}:
            detail = "ダーニャ自身の変な生い立ちを話して、キャラクターに興味を持たせる"
        elif primary_key in {"プロジェクトデザイン", "夢考房とRoboCup", "KIT情報理工学部2025", "KIT情報理工学部ロボティクス学科", "KIT情報理工学部知能情報システム学科", "KIT情報理工学部情報工学科"}:
            detail = "KITの実践教育とものづくり環境の強さへ話を進める"
        elif primary_key in {"KITキャンパス生活", "KITの面倒見と就職", "保護者向けKIT情報"}:
            detail = "学び以外の生活・支援・進路の安心材料へ広げる"
        else:
            detail = "ここまでの話を未来展開やロボット化の夢へつなぐ"
        return f"ストーリー{idx}/{len(arc)}: {detail}"

    def _candidates(self, audience_type: str, long_idle: bool) -> list[dict]:
        if long_idle:
            return TOPIC_DB.copy()
        candidates = [t for t in TOPIC_DB if audience_type in t["audiences"]]
        return candidates or TOPIC_DB.copy()

    def _weighted_choice(self, candidates: list[dict], mode: str, audience: AudienceContext) -> dict:
        weights = []
        for topic in candidates:
            key = topic["key"]
            weight = 1.0
            if mode in ("hook", "intro") and key in {"ダーニャ自身", "画像AIで雰囲気を見る", "人物追跡", "KIT情報理工学部2025"}:
                weight += 1.0
            if mode == "deepen" and key in {"LLMで言葉を考える", "人物追跡", "将来展開", "人型ロボットの今", "ロボットの頭と体"}:
                weight += 1.2
            if audience.vlm_observations and key in {"画像AIで雰囲気を見る", "人物追跡"}:
                weight += 0.8
            if key in {"ダーニャ自身", "ナナマルへの嫉妬", "KITのものづくり教育", "KIT情報理工学部ロボティクス学科", "夢考房とRoboCup", "ダーニャ誕生秘話"}:
                weight += 0.4
            if key.startswith("KIT"):
                weight += 0.35
            if audience.audience_type == "child" and key == "ナナマルへの嫉妬":
                weight += 1.5
            if audience.audience_type == "child" and key in {"KIT情報理工学部ロボティクス学科", "ロボットの頭と体"}:
                weight += 0.8
            if audience.audience_type == "high_school_student" and key in {"夢考房とRoboCup", "プロジェクトデザイン", "KITキャンパス生活", "ダーニャ誕生秘話"}:
                weight += 1.0
            if audience.audience_type == "parent_or_adult" and key in {"将来展開", "KITのものづくり教育", "ロボットの現実的なすごさ", "KITの面倒見と就職", "保護者向けKIT情報"}:
                weight += 1.2
            weights.append(weight)
        return random.choices(candidates, weights=weights, k=1)[0]

    def _should_use_visual_detail(self, primary_topic: str, audience: AudienceContext, scene_hints: list[str]) -> bool:
        if primary_topic == "画像AIで雰囲気を見る":
            return True
        if any("画像" in hint or "人物追跡" in hint for hint in scene_hints):
            return True
        if not audience.vlm_observations:
            return False
        return random.random() < 0.9

    def _related_topics(self, primary: str, candidates: list[dict], avoid: set[str], long_idle: bool) -> list[str]:
        pool = [c["key"] for c in candidates if c["key"] != primary and c["key"] not in avoid]
        if long_idle:
            random.shuffle(pool)
            return pool[:3]
        random.shuffle(pool)
        return pool[:2]

    def _knowledge_points(self, primary: dict, related: list[str], candidates: list[dict], long_idle: bool) -> list[str]:
        by_key = {c["key"]: c for c in candidates}
        points = list(primary.get("knowledge", []))
        for key in related:
            points.extend(by_key.get(key, {}).get("knowledge", [])[:1])
        random.shuffle(points)
        return points[:5 if long_idle else 3]

    def _depth(self, mode: str, visitor: VisitorInfo | None, long_idle: bool) -> int:
        if long_idle:
            return 3
        if mode == "deepen":
            return 3
        if visitor and visitor.dwell_time_sec >= 25:
            return 3
        if mode in ("intro", "returning", "crowd"):
            return 2
        return 1

    def _intent(self, mode: str, audience: AudienceContext, long_idle: bool) -> str:
        if long_idle:
            return "無人でも1分程度の長め独り言で、会場に生きている感じを出す"
        if mode == "idle":
            return "暇そうに短く独り言を言い、来場者を呼び込む"
        if audience.is_returning:
            return "また来てくれたことに反応し、前と違う話題に進める"
        if mode == "hook":
            return "見た目や雰囲気に反応して驚きながら話しかける"
        if mode == "deepen":
            return "同じ人が長くいるので、技術や将来像を一段深く話す"
        if mode == "crowd":
            return "一人に絞らず、全体に向けて展示MCとして話す"
        return "相手に合わせて展示の魅力を自然に説明する"
