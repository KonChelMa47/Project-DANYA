"""trackerログ入力の堅牢化モジュール。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from schemas import CameraSceneEvent, EnvironmentInfoEvent, PersonInfoEvent


class EventInput:
    def __init__(self, log_dir: Path, debug: bool = False) -> None:
        self.log_dir = log_dir
        self.debug = debug

    def _latest_file(self) -> Optional[Path]:
        if not self.log_dir.exists():
            return None
        files = sorted(self.log_dir.glob("*.json"))
        return files[-1] if files else None

    def _parse_json_array(self, text: str) -> list[dict]:
        try:
            data = json.loads(text)
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            return []

    def _parse_jsonl(self, text: str) -> tuple[list[dict], int]:
        events = []
        skipped = 0
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    events.append(obj)
            except json.JSONDecodeError:
                skipped += 1
        return events, skipped

    def _load_events(self, path: Path) -> tuple[list[dict], int]:
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            return [], 0
        if not text.strip():
            return [], 0
        arr = self._parse_json_array(text)
        if arr:
            return [e for e in arr if isinstance(e, dict)], 0
        return self._parse_jsonl(text)

    @staticmethod
    def _event_id(path: Path, idx: int, event: dict) -> str:
        ts = float(event.get("timestamp", 0.0) or 0.0)
        return str(event.get("event_id") or f"{path.name}:{idx}:{ts}")

    def latest_marker(self) -> tuple[Optional[str], float]:
        """起動済みログを既読扱いにするため、現在の末尾イベントを返す。"""
        path = self._latest_file()
        if path is None:
            return None, 0.0
        events, _skipped = self._load_events(path)
        if not events:
            return None, 0.0
        best_idx = -1
        best_ts = 0.0
        for idx, event in enumerate(events):
            try:
                ts = float(event.get("timestamp", 0.0) or 0.0)
            except Exception:
                ts = 0.0
            if (ts, idx) >= (best_ts, best_idx):
                best_ts = ts
                best_idx = idx
        if best_idx < 0:
            return None, 0.0
        return self._event_id(path, best_idx, events[best_idx]), best_ts

    def fetch_latest(self, last_processed_event_id: Optional[str], last_processed_ts: float):
        path = self._latest_file()
        if path is None:
            if self.debug:
                print("[event_input] ログファイル未作成")
            return None, None, None, last_processed_event_id, last_processed_ts

        events, skipped = self._load_events(path)
        if not events:
            if self.debug:
                print("[event_input] 読み取りイベント0件")
            return None, None, None, last_processed_event_id, last_processed_ts

        new_events = []
        for idx, e in enumerate(events):
            ts = float(e.get("timestamp", 0.0) or 0.0)
            ev_id = self._event_id(path, idx, e)
            if last_processed_event_id and ev_id == last_processed_event_id:
                continue
            if ts <= last_processed_ts:
                continue
            e["_source_index"] = idx
            e["_resolved_event_id"] = ev_id
            new_events.append(e)

        latest_person = None
        latest_env = None
        latest_scene = None
        latest_ev_id = last_processed_event_id
        latest_ts = last_processed_ts
        latest_overall_idx = -1
        latest_type_ts = {
            "person_info_updated": -1.0,
            "environment_info_updated": -1.0,
            "camera_scene_updated": -1.0,
        }

        for e in sorted(new_events, key=lambda item: (float(item.get("timestamp", 0.0) or 0.0), int(item.get("_source_index", 0) or 0))):
            et = e.get("event_type")
            ts = float(e.get("timestamp", 0.0) or 0.0)
            ev_id = e.get("_resolved_event_id")
            source_idx = int(e.get("_source_index", 0) or 0)
            if (ts, source_idx) >= (latest_ts, latest_overall_idx):
                latest_ev_id = ev_id
                latest_ts = ts
                latest_overall_idx = source_idx
            if et == "person_info_updated":
                try:
                    if ts >= latest_type_ts[et]:
                        latest_person = PersonInfoEvent(**e)
                        latest_type_ts[et] = ts
                except Exception:
                    pass
            elif et == "environment_info_updated":
                try:
                    if ts >= latest_type_ts[et]:
                        latest_env = EnvironmentInfoEvent(**e)
                        latest_type_ts[et] = ts
                except Exception:
                    pass
            elif et == "camera_scene_updated":
                try:
                    if ts >= latest_type_ts[et]:
                        latest_scene = CameraSceneEvent(**e)
                        latest_type_ts[et] = ts
                except Exception:
                    pass

        if self.debug:
            print(f"[event_input] total={len(events)} new={len(new_events)} skipped_lines={skipped}")
        return latest_person, latest_env, latest_scene, latest_ev_id, latest_ts

