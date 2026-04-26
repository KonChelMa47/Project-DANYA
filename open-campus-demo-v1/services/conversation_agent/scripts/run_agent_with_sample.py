"""サンプルログを作ってagentを短時間起動。"""

from __future__ import annotations

import signal
import subprocess
import time
from pathlib import Path
import sys

base = Path(__file__).resolve().parents[1]
scenario = sys.argv[1] if len(sys.argv) > 1 else "high_school"
subprocess.check_call(["python3", str(base / "scripts/create_sample_tracker_log.py"), "--scenario", scenario])
p = subprocess.Popen(["python3", str(base / "main.py")])
time.sleep(20)
p.send_signal(signal.SIGINT)
try:
    p.wait(timeout=5)
except subprocess.TimeoutExpired:
    p.kill()
