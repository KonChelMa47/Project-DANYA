"""展示会向けの短期人物トラッキング実行スクリプト。"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from typing import List

import cv2
from dotenv import load_dotenv
from ultralytics import YOLO

import config
from daily_json_logger import DailyJsonLogger
from feature_extractor import extract_upper_body_histogram
from visitor_memory import VisitorMemory
from vlm_analyzer import VisitorVLMAnalyzer


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ダーニャ人物追跡")
    parser.add_argument(
        "--terminal-only",
        action="store_true",
        help="OpenCVウィンドウを出さず、ターミナル出力だけで動作確認する",
    )
    parser.add_argument(
        "--mock-vision",
        "--mock-vlm",
        action="store_true",
        help="OpenAI画像AIを呼ばず、bbox由来の軽いモック人物特徴を出す",
    )
    parser.add_argument(
        "--status-interval-sec",
        type=float,
        default=2.0,
        help="terminal-only時のステータス表示間隔",
    )
    parser.add_argument(
        "--scene-interval-sec",
        type=float,
        default=25.0,
        help="カメラ全体の場面観察イベントを出す間隔",
    )
    return parser


def _parse_yolo_results(results) -> List[dict]:
    """YOLO+ByteTrackの結果を扱いやすい辞書配列へ変換する。"""
    detections = []
    if not results:
        return detections

    result = results[0]
    if result.boxes is None:
        return detections

    boxes = result.boxes
    xyxy = boxes.xyxy.cpu().numpy().astype(int) if boxes.xyxy is not None else []
    ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

    for i, box in enumerate(xyxy):
        x1, y1, x2, y2 = box.tolist()
        track_id = int(ids[i]) if ids is not None and i < len(ids) else None
        detections.append(
            {
                "bbox": (x1, y1, x2, y2),
                "track_id": track_id,
            }
        )
    return detections


def _bbox_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)


def _dedupe_detections(detections: List[dict]) -> List[dict]:
    """同じ人物に複数bboxが出た時、最大面積のbboxだけ残す。"""
    ordered = sorted(
        detections,
        key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]),
        reverse=True,
    )
    kept: List[dict] = []
    for det in ordered:
        if any(_bbox_iou(det["bbox"], k["bbox"]) >= config.DUPLICATE_BBOX_IOU_THRESHOLD for k in kept):
            continue
        kept.append(det)
    return kept


def _draw_overlay(frame, detections: List[dict], track_to_visitor: dict, memory: VisitorMemory):
    """画面にbboxやvisitor情報を描画する。"""
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        track_id = det.get("track_id")
        visitor_id = track_to_visitor.get(track_id)
        if not visitor_id:
            continue

        visitor = memory.visitors.get(visitor_id)
        if visitor is None:
            continue

        if visitor.is_long_stay and visitor.status == "active":
            color = config.LONG_STAY_BBOX_COLOR
        else:
            color = (0, 255, 0) if visitor.status == "active" else (0, 180, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = (
            f"{visitor.visitor_id} | {visitor.status} | "
            f"dwell:{visitor.current_dwell_time_sec:.1f}s"
        )
        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )


def _mock_vlm_result(visitor, det: dict) -> dict:
    """カメラ追跡は使い、人物特徴だけ軽くモックする。"""
    bbox_width = max(0, det["bbox"][2] - det["bbox"][0])
    if bbox_width >= 420:
        distance_note = "近くで展示を見ている"
    elif bbox_width >= 260:
        distance_note = "少し離れて立ち止まっている"
    else:
        distance_note = "通りがかりに近い"
    age = "teen high school student" if visitor.visit_count <= 1 else "returning visitor"
    pose = "looking_at_display" if bbox_width >= 260 else "walking"
    carried_items = ["smartphone"] if visitor.current_dwell_time_sec >= 6 else []
    accessories = ["backpack"] if visitor.visit_count <= 1 else []
    return {
        "vlm_enabled": False,
        "gender": "unknown",
        "age_estimate": age,
        "clothing_description": f"{distance_note}来場者。服装詳細はモックのためunknown",
        "expression": "curious" if visitor.current_dwell_time_sec >= 8 else "neutral",
        "pose_description": pose,
        "accessories": accessories,
        "carried_items": carried_items,
        "expression_confidence": 0.5,
    }


def _print_terminal_status(memory: VisitorMemory, now: float) -> None:
    active = [v for v in memory.visitors.values() if v.status == "active"]
    lost = [v for v in memory.visitors.values() if v.status == "lost"]
    summary = {
        "event_type": "tracker_status",
        "timestamp": now,
        "active_count": len(active),
        "lost_count": len(lost),
        "active_visitors": [
            {
                "visitor_id": v.visitor_id,
                "dwell_time_sec": round(v.current_dwell_time_sec, 1),
                "visit_count": v.visit_count,
                "returning": v.was_lost_and_returned,
                "vlm_analyzed": v.vlm_analyzed,
            }
            for v in active
        ],
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def _open_preferred_camera():
    """設定された候補順でカメラを開き、最初に読めたものを返す。"""
    candidates = []
    if hasattr(config, "CAMERA_CANDIDATES"):
        candidates.extend(config.CAMERA_CANDIDATES)
    if config.CAMERA_INDEX not in candidates:
        candidates.append(config.CAMERA_INDEX)

    tested = []
    for cam_idx in candidates:
        cap = cv2.VideoCapture(cam_idx)
        if not cap.isOpened():
            cap.release()
            tested.append(f"{cam_idx}:open_failed")
            continue
        ok, _ = cap.read()
        if ok:
            print(f"使用カメラ: index={cam_idx}")
            return cap
        cap.release()
        tested.append(f"{cam_idx}:read_failed")

    tested_text = ", ".join(tested) if tested else "no_candidates"
    raise RuntimeError(
        f"Webカメラを開けませんでした。試行結果: {tested_text} / "
        "config.py の CAMERA_CANDIDATES を確認してください。"
    )


def _build_environment_info(memory: VisitorMemory, now: float) -> dict:
    """現在の環境情報（人数・滞在時間統計・時刻）を作る。"""
    active_visitors = [v for v in memory.visitors.values() if v.status == "active"]
    dwell_list = [round(now - v._active_started_at, 3) for v in active_visitors]

    return {
        "current_people_count": len(active_visitors),
        "longest_dwell_time_sec": max(dwell_list) if dwell_list else 0.0,
        "shortest_dwell_time_sec": min(dwell_list) if dwell_list else 0.0,
        "real_world_time": datetime.now().isoformat(timespec="seconds"),
    }


def main():
    """メイン処理。"""
    args = _build_arg_parser().parse_args()
    # .env を読み込み、OPENAI_API_KEY を使えるようにする
    load_dotenv()

    # モデル読み込み（personクラスのみ追跡で利用）
    model = YOLO(config.YOLO_MODEL_NAME)
    memory = VisitorMemory(
        lost_timeout_sec=config.LOST_TIMEOUT_SEC,
        same_person_threshold=config.SAME_PERSON_THRESHOLD,
    )
    vlm = VisitorVLMAnalyzer(model_name=config.OPENAI_MODEL_NAME)
    daily_logger = DailyJsonLogger(log_dir="logs")
    known_visitor_ids = set()
    announced_returning_visit_keys = set()
    last_returning_announcement = {}
    last_status_print = 0.0
    last_scene_analysis = 0.0
    if args.mock_vision:
        print("画像AIモック: OpenAI画像AIは呼ばず、bbox由来の仮特徴でperson_infoを出します。", flush=True)
    elif config.ENABLE_VLM_ANALYSIS and not os.getenv("OPENAI_API_KEY"):
        print("注意: OPENAI_API_KEY が未設定のため、VLM解析は無効で動作します。")

    cap = _open_preferred_camera()

    if args.terminal_only:
        print("トラッキング開始: terminal-only。Ctrl+Cで終了します。", flush=True)
    else:
        print("トラッキング開始: qキーで終了します。", flush=True)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("フレーム取得に失敗したため終了します。")
                break

            now = time.time()
            results = model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[0],  # personのみ
                verbose=False,
            )
            detections = _dedupe_detections(_parse_yolo_results(results))

            # 服装ヒストグラムを各検出に追加
            for det in detections:
                det["clothing_hist"] = extract_upper_body_histogram(frame, det["bbox"])

            # visitorメモリ更新
            track_to_visitor = memory.update(
                detections=detections,
                frame_shape=frame.shape,
                now=now,
            )

            # 新規visitorまたは復帰visitorのタイミングだけ環境情報を更新
            active_visitor_ids = {
                v.visitor_id for v in memory.visitors.values() if v.status == "active"
            }
            new_visitor_ids = active_visitor_ids - known_visitor_ids
            returning_visitor_ids = []
            for visitor in memory.visitors.values():
                if visitor.status != "active" or not visitor.was_lost_and_returned:
                    continue
                visit_key = f"{visitor.visitor_id}:{visitor.visit_count}"
                if visit_key in announced_returning_visit_keys:
                    continue
                last_announced = last_returning_announcement.get(visitor.visitor_id, 0.0)
                if now - last_announced < 8.0:
                    continue
                returning_visitor_ids.append(visitor.visitor_id)
                announced_returning_visit_keys.add(visit_key)
                last_returning_announcement[visitor.visitor_id] = now
            if new_visitor_ids or returning_visitor_ids:
                environment_info = _build_environment_info(memory=memory, now=now)
                env_record = {
                    "event_type": "environment_info_updated",
                    "timestamp": now,
                    "new_visitor_ids": sorted(new_visitor_ids),
                    "returning_visitor_ids": sorted(returning_visitor_ids),
                    "environment_info": environment_info,
                }
                daily_logger.append(env_record)
                print(json.dumps(env_record, ensure_ascii=False), flush=True)
                known_visitor_ids.update(new_visitor_ids)

            if now - last_scene_analysis >= max(10.0, args.scene_interval_sec):
                scene_info = vlm.mock_scene(len(active_visitor_ids)) if args.mock_vision else vlm.analyze_scene(frame, len(active_visitor_ids))
                scene_record = {
                    "event_type": "camera_scene_updated",
                    "timestamp": now,
                    "scene_info": scene_info,
                }
                daily_logger.append(scene_record)
                print(json.dumps(scene_record, ensure_ascii=False), flush=True)
                last_scene_analysis = now

            # 10秒以上滞在した人物に対して、色変更＆VLM解析を実施
            for det in detections:
                track_id = det.get("track_id")
                visitor_id = track_to_visitor.get(track_id)
                if not visitor_id:
                    continue
                visitor = memory.visitors.get(visitor_id)
                if visitor is None:
                    continue
                if visitor.status != "active":
                    continue

                # ロジックベース値
                visitor.current_dwell_time_sec = now - visitor._active_started_at
                visitor.bbox_width_px = max(0, det["bbox"][2] - det["bbox"][0])

                # 10秒超え判定
                if visitor.current_dwell_time_sec >= config.LONG_STAY_THRESHOLD_SEC:
                    visitor.is_long_stay = True

                    # 初回だけVLM解析（bbox内画像のみ送信）
                    if (config.ENABLE_VLM_ANALYSIS or args.mock_vision) and not visitor.vlm_analyzed:
                        try:
                            if args.mock_vision:
                                result = _mock_vlm_result(visitor, det)
                            else:
                                result = vlm.analyze(frame=frame, bbox=det["bbox"])
                            visitor.gender = result.get("gender", "unknown")
                            visitor.age_estimate = result.get("age_estimate", "unknown")
                            visitor.clothing_description = result.get(
                                "clothing_description", "unknown"
                            )
                            visitor.expression = result.get("expression", "unknown")
                            visitor.pose_description = result.get("pose_description", "unknown")
                            visitor.accessories = result.get("accessories", []) or []
                            visitor.carried_items = result.get("carried_items", []) or []
                            visitor.expression_confidence = float(
                                result.get("expression_confidence", 0.0) or 0.0
                            )
                            visitor.vlm_analyzed = True

                            # VLM+ロジック値をまとめて「人物情報」として扱う
                            person_info = {
                                "visitor_id": visitor.visitor_id,
                                "dwell_time_sec": round(visitor.current_dwell_time_sec, 2),
                                "bbox_width_px": visitor.bbox_width_px,
                                "visit_count": visitor.visit_count,
                                "was_lost_and_returned": visitor.was_lost_and_returned,
                                "same_person_confidence": round(visitor.same_person_confidence, 3),
                                "gender": visitor.gender,
                                "age_estimate": visitor.age_estimate,
                                "clothing_description": visitor.clothing_description,
                                "expression": visitor.expression,
                                "pose_description": visitor.pose_description,
                                "accessories": visitor.accessories,
                                "carried_items": visitor.carried_items,
                                "expression_confidence": round(
                                    visitor.expression_confidence, 3
                                ),
                            }

                            person_record = {
                                "event_type": "person_info_updated",
                                "timestamp": now,
                                "person_info": person_info,
                            }

                            # ターミナル表示
                            print(json.dumps(person_record, ensure_ascii=False), flush=True)
                            # 日付JSONへ保存
                            daily_logger.append(person_record)
                        except Exception as e:
                            print(f"[VLM_ERROR] visitor={visitor.visitor_id} error={e}")
                            visitor.vlm_analyzed = True

            if args.terminal_only:
                if now - last_status_print >= max(0.5, args.status_interval_sec):
                    _print_terminal_status(memory, now)
                    last_status_print = now
            else:
                # 描画
                _draw_overlay(frame, detections, track_to_visitor, memory)
                cv2.imshow(config.WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

    finally:
        # 終了時は必ずリソースとメモリを解放
        cap.release()
        cv2.destroyAllWindows()
        memory.clear_all()
        print("終了: visitorメモリを削除しました。")


if __name__ == "__main__":
    main()
