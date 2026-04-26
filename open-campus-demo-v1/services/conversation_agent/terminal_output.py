"""ターミナル出力。"""

from __future__ import annotations

from datetime import datetime

# ANSIカラー（対応端末のみ色付き表示）
GREEN = "\033[92m"
BLUE = "\033[94m"
RESET = "\033[0m"


def print_speech(output, wait_sec: float, debug: bool = False) -> None:
    now_iso = datetime.now().astimezone().isoformat(timespec="seconds")
    if getattr(output, "segments", None):
        rendered = " ".join(f"<{seg.emotion}>{seg.text}" for seg in output.segments)
    else:
        rendered = output.speech
    line = f"[{now_iso}] {rendered} | wait:{wait_sec:.1f}s"
    print(f"{GREEN}{line}{RESET}")
    if debug:
        print(
            f"{BLUE}mode={output.mode} topic={output.topic} "
            f"target={output.target_visitor_id}{RESET}"
        )


def print_internal_thinking(message: str) -> None:
    """裏側の考察・戦略ログを青で表示する。"""
    now_iso = datetime.now().astimezone().isoformat(timespec="seconds")
    line = f"[{now_iso}] {message}"
    print(f"{BLUE}{line}{RESET}")

