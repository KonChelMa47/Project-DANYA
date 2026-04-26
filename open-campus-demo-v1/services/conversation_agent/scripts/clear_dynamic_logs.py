"""dynamic_rag/event_logs をクリア。"""

from __future__ import annotations

from pathlib import Path

base = Path(__file__).resolve().parents[1]
log_dir = base / "dynamic_rag/event_logs"
if not log_dir.exists():
    print("no log dir")
else:
    count = 0
    for p in log_dir.glob("*.jsonl"):
        p.unlink(missing_ok=True)
        count += 1
    print(f"deleted {count} files")
