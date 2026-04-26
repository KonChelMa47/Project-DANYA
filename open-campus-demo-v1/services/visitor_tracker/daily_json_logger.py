"""日付ごとのJSONログ保存を担当するモジュール。"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List


class DailyJsonLogger:
    """その日付のJSONファイルへイベントを追記するロガー。"""

    def __init__(self, log_dir: str = "logs") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _today_file_path(self) -> Path:
        """今日の日付ファイルパスを返す。"""
        date_str = datetime.now().strftime("%Y-%m-%d")
        return self.log_dir / f"{date_str}.json"

    def _load_or_create(self, path: Path) -> List[Dict[str, Any]]:
        """既存JSONを読み込む。なければ空配列を作る。"""
        if not path.exists():
            return []
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            return []
        except Exception:
            # 壊れていても運用停止しないよう、空配列で継続
            return []

    def append(self, record: Dict[str, Any]) -> Path:
        """レコードを今日のJSONへ追記して保存する。"""
        file_path = self._today_file_path()
        data = self._load_or_create(file_path)
        data.append(record)
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return file_path
