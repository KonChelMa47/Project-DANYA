"""ダーニャ自律発話エージェントのメインループ。"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

import config
from agent_state import AgentState
from behavior_composer import BehaviorComposer
from dotenv import load_dotenv
from emotion_planner import EmotionPlanner
from event_input import EventInput
from global_vlm import GlobalVLMService
from idle_talker import IdleTalker
from rag_manager import RagManager
from llm_client import generate_dedupe_append_segment
from schemas import GlobalSceneInfo, SpeechOutput, SpeechSegment, VisitorInfo, VLMInfo
from self_improvement.event_logger import EventLogger
from self_improvement.rag_write_guard import RagWriteGuard
from self_improvement.reaction_evaluator import ReactionEvaluator
from self_improvement.strategy_memory_writer import StrategyMemoryWriter
from situation_analyzer import SituationAnalyzer
from speech_history import SpeechHistory
from speech_loop import SpeechLoop
from speech_server import start_speech_server
from speech_split import split_segments_at_nearest_period
from special_talker import DAILY_CHITCHAT_TOPICS
from special_talker import SpecialTalker
from special_talker import SPECIAL_KINDS
from terminal_output import print_speech
from topic_manager import TopicManager
from visitor_memory import AgentVisitorMemory

# 本番 run(): 発話ごとに 通常→特別→通常→日常 を繰り返す（speech_count を 4 で割った余り）
_SPEECH_OUTPUT_CYCLE = ("normal", "special", "normal", "daily")


def _publish_speech_maybe_halved(
    speech: SpeechOutput,
    speech_loop: SpeechLoop,
    speech_broadcaster,
    *,
    debug: bool,
) -> None:
    """全文の中央付近の『。』で前半・後半に分け、前半の推定読了時間だけ待ってから後半を送る。"""
    split = split_segments_at_nearest_period(speech.segments)
    if split is None:
        wait_sec = speech_loop.estimate_busy_sec(speech.speech)
        print_speech(speech, wait_sec=wait_sec, debug=debug)
        speech_broadcaster.publish(speech, wait_sec)
        return
    left_segs, right_segs = split
    speech_left = SpeechOutput.from_segments(
        mode=speech.mode,
        target_visitor_id=speech.target_visitor_id,
        segments=left_segs,
        topic=speech.topic,
        strategy_summary=speech.strategy_summary,
        priority=speech.priority,
    )
    speech_right = SpeechOutput.from_segments(
        mode=speech.mode,
        target_visitor_id=speech.target_visitor_id,
        segments=right_segs,
        topic=speech.topic,
        strategy_summary=speech.strategy_summary,
        priority=speech.priority,
    )
    w1 = speech_loop.estimate_busy_sec(speech_left.speech)
    print_speech(speech_left, wait_sec=w1, debug=debug)
    speech_broadcaster.publish(speech_left, w1)
    time.sleep(w1)
    w2 = speech_loop.estimate_busy_sec(speech_right.speech)
    print_speech(speech_right, wait_sec=w2, debug=debug)
    speech_broadcaster.publish(speech_right, w2)


def _on_spoken_after_halved_delivery(speech: SpeechOutput, speech_loop: SpeechLoop, state: AgentState) -> None:
    """二段送信時は前半分の待機を既に sleep 済みなので、speech_busy は後半の推定時間だけ足す。"""
    split = split_segments_at_nearest_period(speech.segments)
    if split is None:
        speech_loop.on_spoken(time.time(), speech.speech, speech.topic, speech.target_visitor_id, state)
        return
    _, right_segs = split
    speech_right = SpeechOutput.from_segments(
        mode=speech.mode,
        target_visitor_id=speech.target_visitor_id,
        segments=right_segs,
        topic=speech.topic,
        strategy_summary=speech.strategy_summary,
        priority=speech.priority,
    )
    w2 = speech_loop.estimate_busy_sec(speech_right.speech)
    speech_loop.on_spoken(
        time.time(), speech.speech, speech.topic, speech.target_visitor_id, state, busy_sec=w2
    )


def _build_visitors(person_event, people_count: int) -> list[VisitorInfo]:
    if people_count <= 0 or not person_event:
        return []
    return [VisitorInfo.from_person_info(person_event.person_info)]


def _people_count(env_event) -> int:
    if not env_event:
        return 0
    try:
        return int(float(env_event.environment_info.get("current_people_count", 0) or 0))
    except Exception:
        return 0


def _has_visual_detail(visitor: VisitorInfo | None) -> bool:
    if visitor is None:
        return False
    vlm = visitor.vlm
    values = [
        vlm.clothing_description,
        vlm.pose_description,
        vlm.expression,
        vlm.age_estimate,
        *vlm.accessories,
        *vlm.carried_items,
    ]
    return any(str(value or "").strip().lower() not in {"", "unknown", "-"} for value in values)


def _event_is_fresh(event, now: float, ttl_sec: float) -> bool:
    if event is None:
        return False
    try:
        ts = float(event.timestamp or 0.0)
    except Exception:
        return False
    return ts > 0 and (now - ts) <= ttl_sec


def _rag_audience(audience_type: str) -> str:
    if audience_type == "high_school_student":
        return "high_school_student"
    if audience_type == "child":
        return "child"
    if audience_type == "parent_or_adult":
        return "parent"
    return "general"


def _debug_mode_visitor(index: int) -> VisitorInfo:
    expressions = ["smiling", "curious", "neutral", "excited"]
    clothes = ["sweater", "blue hoodie", "white shirt", "knit sweater"]
    poses = ["standing", "looking_at_display", "leaning", "using_phone"]
    items = [[], ["smartphone"], ["pamphlet"], ["bag"]]
    return VisitorInfo(
        visitor_id="debug_mode_visitor",
        dwell_time_sec=12.0 + index,
        bbox_width_px=620,
        vlm=VLMInfo(
            age_estimate="20s",
            clothing_description=random.choice(clothes),
            expression=random.choice(expressions),
            pose_description=random.choice(poses),
            accessories=["glasses"] if index % 2 == 0 else [],
            carried_items=random.choice(items),
            expression_confidence=0.9,
        ),
    )


def run_debug_mode() -> AgentState:
    """Enterを押すたびに発話を生成する端末確認用（本番の4拍子サイクルは run が担当）。"""
    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / "visitor_tracker/.env", override=False)

    state = AgentState()
    rag = RagManager(config.static_rag_dir, config.dynamic_rag_dir)
    rag.ensure_placeholders()
    visitor_memory = AgentVisitorMemory()
    topic_manager = TopicManager()
    emotion_planner = EmotionPlanner()
    composer = BehaviorComposer()
    special_talker = SpecialTalker()
    speech_loop = SpeechLoop()
    speech_broadcaster = start_speech_server()

    print(
        "[debug_mode] Enterで通常発話。special / ramen / urban_legend / gag / daily（日常）などで特別。qで終了。",
        flush=True,
    )
    turn = 0
    while True:
        command = input("[debug_mode] > ").strip()
        if command.lower() in {"q", "quit", "exit"}:
            break
        turn += 1
        state.visitor_topic_memory["debug_mode_visitor"] = []
        del state.recent_topics[:]
        del state.recent_speeches[:]
        del state.recent_openings[:]
        state.visitor_opening_memory.pop("debug_mode_visitor", None)
        state.visitor_story_steps.pop("debug_mode_visitor", None)
        state.already_explained_topics.clear()
        visitor = _debug_mode_visitor(turn)
        people_count = 1
        visitor_memory.update_seen([visitor], time.time(), state)
        state.detect_new_visitors({visitor.visitor_id})

        scene = GlobalSceneInfo(
            scene_summary="デバッグ確認用。来場1名の想定。",
            people_flow="stable",
            attention_target="danya",
            crowd_state="single",
            movement_state="stopped",
            engagement_estimate=0.8,
            notable_event="デバッグ",
            recommended_interaction="intro",
        )
        mode = "intro"
        audience = visitor_memory.build_context(visitor=visitor, people_count=people_count, state=state)
        topic_plan = topic_manager.pick(
            mode=mode,
            visitor=visitor,
            people_count=people_count,
            audience=audience,
            scene=scene,
            state=state,
            long_idle=False,
        )
        rag_query = rag.build_query(
            mode=topic_plan.mode,
            audience=_rag_audience(topic_plan.audience_type),
            returning=topic_plan.is_returning,
        )
        rag_results = rag.search(rag_query) if config.rag_enabled else []
        emotion_plan = emotion_planner.plan(topic_plan, state, long_form=True)

        use_daily = False
        special_kind: str | None = None
        if command:
            if command in {"daily", "日常"}:
                use_daily = True
            elif command == "special":
                special_kind = random.choice(SPECIAL_KINDS)
            elif command in SPECIAL_KINDS:
                special_kind = command
            else:
                print(f"[debug_mode] 未知の指定: {command}。通常発話にします。", flush=True)

        if use_daily:
            dkey = random.choice(DAILY_CHITCHAT_TOPICS)
            speech = special_talker.compose(
                "daily_chitchat", topic_plan, people_count, daily_key=dkey, debug_mode=True
            )
        elif special_kind:
            speech = special_talker.compose(special_kind, topic_plan, people_count, debug_mode=True)
        else:
            speech = composer.compose(
                topic_plan, emotion_plan, rag_results, state, long_form=True, debug_mode=True
            )

        _publish_speech_maybe_halved(speech, speech_loop, speech_broadcaster, debug=True)
        _on_spoken_after_halved_delivery(speech, speech_loop, state)
        state.remember_story_step(topic_plan.target_visitor_id, topic_plan.audience_type, topic_plan.story_step)

    return state


def run() -> AgentState:
    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / "visitor_tracker/.env", override=False)

    state = AgentState()
    rag = RagManager(config.static_rag_dir, config.dynamic_rag_dir)
    rag.ensure_placeholders()

    input_source = EventInput(config.tracker_log_dir, debug=config.debug)
    # 日付ログは過去イベントを丸ごと保持するため、起動時点の末尾までは既読扱いにする。
    state.last_processed_event_id, state.last_processed_timestamp = input_source.latest_marker()
    global_vlm = GlobalVLMService(
        use_openai=config.use_openai,
        timeout_sec=config.global_vlm_timeout_sec,
        debug=config.debug,
    )
    analyzer = SituationAnalyzer()
    visitor_memory = AgentVisitorMemory()
    topic_manager = TopicManager()
    emotion_planner = EmotionPlanner()
    composer = BehaviorComposer()
    idle_talker = IdleTalker()
    speech_history = SpeechHistory()
    speech_loop = SpeechLoop()
    speech_broadcaster = start_speech_server()
    special_talker = SpecialTalker()
    logger = EventLogger(config.agent_log_dir)
    evaluator = ReactionEvaluator()
    guard = RagWriteGuard(config.dynamic_rag_dir, config.static_rag_dir)
    memory_writer = StrategyMemoryWriter(config.dynamic_rag_dir, guard, batch_size=10)

    before_visitor = None
    before_people_count = 0
    before_scene = GlobalSceneInfo()
    current_person_event = None
    current_env_event = None
    current_scene_event = None

    while True:
        now = time.time()
        person_event, env_event, scene_event, ev_id, ev_ts = input_source.fetch_latest(
            state.last_processed_event_id, state.last_processed_timestamp
        )
        fresh_person_event = person_event is not None
        if ev_id:
            state.last_processed_event_id = ev_id
            state.last_processed_timestamp = ev_ts
        if person_event is not None:
            current_person_event = person_event
        if env_event is not None:
            current_env_event = env_event
        if scene_event is not None:
            current_scene_event = scene_event
            state.last_camera_scene_info = scene_event.scene_info

        if not _event_is_fresh(current_person_event, now, config.person_event_ttl_sec):
            current_person_event = None
        if not _event_is_fresh(current_scene_event, now, config.camera_scene_event_ttl_sec):
            current_scene_event = None

        people_count = _people_count(current_env_event)
        if people_count <= 0 and fresh_person_event:
            people_count = 1
        elif people_count <= 0:
            current_person_event = None
        visitors = _build_visitors(current_person_event, people_count)
        for visitor in visitors:
            if visitor.visitor_id != "unknown":
                state.visitor_profiles[visitor.visitor_id] = visitor.model_dump()
        env_returning_ids = set(env_event.returning_visitor_ids) if env_event is not None else set()
        if not visitors and env_returning_ids:
            for visitor_id in sorted(env_returning_ids):
                profile = state.visitor_profiles.get(visitor_id)
                if not profile:
                    continue
                profile = dict(profile)
                profile["returning"] = True
                profile["visit_count"] = int(profile.get("visit_count", 1) or 1) + 1
                visitors.append(VisitorInfo(**profile))
                break
        if people_count <= 0 and visitors:
            people_count = 1

        state.update_idle_presence(now, people_count)
        visitor_memory.update_seen(visitors, now, state)
        env_new_ids = set(env_event.new_visitor_ids) if env_event is not None else set()
        active_ids = {v.visitor_id for v in visitors if v.visitor_id != "unknown"}
        new_ids = state.detect_new_visitors(env_new_ids | active_ids)
        longest_dwell = max([v.dwell_time_sec for v in visitors], default=0.0)
        has_returning = bool(env_returning_ids) or any(v.returning or v.visit_count > 1 for v in visitors)

        if global_vlm.should_run(now, bool(new_ids), state):
            scene = global_vlm.run(
                state,
                people_count=people_count,
                longest_dwell_sec=longest_dwell,
                has_new_visitor=bool(new_ids),
                returning=has_returning,
            )
        else:
            scene = state.last_global_scene_info
        if current_scene_event is not None:
            scene_info = current_scene_event.scene_info
            scene.scene_summary = scene_info.get("summary", scene.scene_summary)
            scene.notable_event = ", ".join(scene_info.get("topic_hints", [])[:3]) or scene.notable_event
            if scene_info.get("likely_audience") == "group":
                scene.crowd_state = "small_group" if people_count < 4 else "crowd"

        idle_kind = idle_talker.due_kind(now, people_count, state)
        if people_count <= 0 and idle_kind is None:
            time.sleep(1.0)
            continue

        situation = analyzer.analyze(
            visitors=visitors,
            people_count=people_count,
            global_scene=scene,
            state=state,
            new_visitor_ids=new_ids,
        )
        target = max(visitors, key=lambda v: v.dwell_time_sec) if visitors else None
        audience = visitor_memory.build_context(visitor=target, people_count=people_count, state=state)
        if audience.is_returning and situation.mode not in ("closing", "crowd"):
            situation.mode = "returning"
            situation.reason = "発話側memoryで再訪扱い"
        state.current_mode = situation.mode

        # 画像解析のperson_infoが来る前は、相手特徴に触れる長文を出さず短い反応に留める。
        # 通常枠は常に長め（知識点・感情の幅を確保）にする。
        slot_for_turn = _SPEECH_OUTPUT_CYCLE[state.speech_count % 4]
        long_form = (
            (target is not None and _has_visual_detail(target))
            or idle_kind == "long"
            or slot_for_turn == "normal"
        )
        topic_plan = topic_manager.pick(
            mode=situation.mode,
            visitor=target,
            people_count=people_count,
            audience=audience,
            scene=scene,
            state=state,
            long_idle=(idle_kind == "long"),
        )
        state.current_strategy = topic_plan.intent

        rag_query = rag.build_query(
            mode=topic_plan.mode,
            audience=_rag_audience(topic_plan.audience_type),
            returning=topic_plan.is_returning,
        )
        rag_results = rag.search(rag_query) if config.rag_enabled else []
        emotion_plan = emotion_planner.plan(topic_plan, state, long_form=long_form)

        can_speak, _reason, _interval = speech_loop.can_emit(now, people_count, state, debug=config.debug)
        if not can_speak:
            time.sleep(1.0)
            continue

        slot = slot_for_turn
        if slot == "normal":
            speech = composer.compose(topic_plan, emotion_plan, rag_results, state, long_form=long_form)
        elif slot == "special":
            pool = [k for k in SPECIAL_KINDS if k != state.last_cycle_special_kind] or list(SPECIAL_KINDS)
            sk = random.choice(pool)
            state.last_cycle_special_kind = sk
            speech = special_talker.compose(sk, topic_plan, people_count)
        else:
            dkey = random.choice(DAILY_CHITCHAT_TOPICS)
            speech = special_talker.compose("daily_chitchat", topic_plan, people_count, daily_key=dkey)
        if speech_history.is_too_similar(speech, state) and config.use_llm:
            dedupe = generate_dedupe_append_segment(
                {
                    "recent_speech": speech.speech,
                    "reason": "states.recent と構成が似すぎるので別角度の一文を追加",
                }
            )
            if dedupe and isinstance(dedupe.get("segments"), list) and dedupe["segments"]:
                row = dedupe["segments"][0]
                if isinstance(row, dict) and row.get("text"):
                    speech.segments.append(
                        SpeechSegment(
                            emotion=str(row.get("emotion", "happy_normal")),
                            text=str(row["text"]).strip(),
                        )
                    )
                    speech = speech.__class__.from_segments(
                        mode=speech.mode,
                        target_visitor_id=speech.target_visitor_id,
                        segments=speech.segments,
                        topic=speech.topic,
                        strategy_summary=f"{speech.strategy_summary} duplicate_adjusted=llm",
                        priority=speech.priority,
                    )

        _publish_speech_maybe_halved(speech, speech_loop, speech_broadcaster, debug=config.debug)
        _on_spoken_after_halved_delivery(speech, speech_loop, state)
        state.remember_story_step(
            topic_plan.target_visitor_id,
            topic_plan.audience_type,
            topic_plan.story_step,
        )
        speech_history.remember(speech, state)
        if speech.target_visitor_id:
            state.visitor_last_spoken[speech.target_visitor_id] = now
        if idle_kind:
            idle_talker.mark_spoken(now, state, idle_kind)

        after_visitor = target
        evaluation = evaluator.evaluate(
            before_visitor=before_visitor,
            after_visitor=after_visitor,
            before_people_count=before_people_count,
            after_people_count=people_count,
            before_scene=before_scene,
            after_scene=scene,
        )
        log = {
            "timestamp_iso": datetime.now().astimezone().isoformat(timespec="seconds"),
            "mode": speech.mode,
            "emotion_tag": speech.emotion_tag,
            "emotion_flow": [seg.emotion for seg in speech.segments],
            "topic": speech.topic,
            "topic_plan": topic_plan.model_dump(),
            "emotion_plan": emotion_plan.model_dump(),
            "strategy_summary": speech.strategy_summary,
            "used_rag_files": [r.get("file") for r in rag_results],
            "use_llm": config.use_llm,
            "llm_success": "llm_success=True" in speech.strategy_summary,
            "fallback_used": "fallback=True" in speech.strategy_summary,
            "speech": speech.model_dump(),
            "situation": situation.model_dump(),
            "scene": scene.model_dump(),
            "before_state": {"people_count": before_people_count, "mode": state.current_mode},
            "after_state": {"people_count": people_count, "mode": situation.mode},
            "evaluation": evaluation.model_dump(),
            "speech_cycle_slot": slot,
        }
        logger.append_json_line(log)
        memory_writer.append(log)
        memory_writer.maybe_flush()
        before_visitor = after_visitor
        before_people_count = people_count
        before_scene = scene

        time.sleep(1.0)


def _print_final_state(state: AgentState) -> None:
    print("[final_state]", json.dumps(state.to_debug_dict(), ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ダーニャ自律発話エージェント")
    parser.add_argument(
        "--debug-mode",
        "--back-mode",
        "--batch-mode",
        dest="debug_mode",
        action="store_true",
        help="Enterで発話を確認する端末用デバッグモード（--back-mode は互換別名）",
    )
    args = parser.parse_args()
    try:
        if args.debug_mode:
            run_debug_mode()
        else:
            run()
    except KeyboardInterrupt:
        pass
