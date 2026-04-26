"""戦略・発話ログをJSONL保存。"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

class EventLogger:
    def __init__(self, log_dir: Path) -> None:
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _path(self) -> Path:
        date = datetime.now().strftime("%Y-%m-%d")
        return self.log_dir / f"{date}_strategy_events.jsonl"

    def append_json_line(self, data: dict) -> None:
        path = self._path()
        with path.open("a", encoding="utf-8") as f:
            import json
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

