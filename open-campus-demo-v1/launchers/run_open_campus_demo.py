"""Open Campus Demo v1 の visitor tracker と conversation agent を同時起動する。"""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path


def _python_bin(demo_root: Path) -> str:
    project_python = demo_root.parent / ".venv" / "bin" / "python"
    if project_python.exists():
        return str(project_python)
    return sys.executable


def _spawn(name: str, cmd: list[str], cwd: Path, quiet: bool = False):
    stdout = subprocess.DEVNULL if quiet else None
    print(f"起動中: {name}")
    return subprocess.Popen(cmd, cwd=str(cwd), stdout=stdout, stderr=stdout)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-only", action="store_true")
    parser.add_argument("--tracker-only", action="store_true")
    parser.add_argument("--show-tracker", action="store_true", help="tracker標準出力を表示")
    parser.add_argument(
        "--terminal-only",
        "--tracker-terminal-only",
        action="store_true",
        help="trackerのOpenCVウィンドウを出さずにターミナル確認だけで動かす",
    )
    parser.add_argument(
        "--mock-vision",
        "--mock-vlm",
        "--tracker-mock-vision",
        action="store_true",
        help="OpenAI画像AIを呼ばず、trackerのモック画像解析を使う",
    )
    parser.add_argument("--status-interval-sec", type=float, default=None)
    parser.add_argument("--scene-interval-sec", type=float, default=None)
    args = parser.parse_args()

    demo_root = Path(__file__).resolve().parents[1]
    services_dir = demo_root / "services"
    tracker_dir = services_dir / "visitor_tracker"
    agent_dir = services_dir / "conversation_agent"
    python_bin = _python_bin(demo_root)

    tracker_proc = None
    agent_proc = None

    if not args.agent_only:
        tracker_cmd = [python_bin, "main.py"]
        if args.terminal_only:
            tracker_cmd.append("--terminal-only")
        if args.mock_vision:
            tracker_cmd.append("--mock-vision")
        if args.status_interval_sec is not None:
            tracker_cmd.extend(["--status-interval-sec", str(args.status_interval_sec)])
        if args.scene_interval_sec is not None:
            tracker_cmd.extend(["--scene-interval-sec", str(args.scene_interval_sec)])
        tracker_proc = _spawn(
            "visitor_tracker",
            tracker_cmd,
            tracker_dir,
            quiet=not args.show_tracker,
        )
        time.sleep(2.0)

    if not args.tracker_only:
        agent_proc = _spawn("conversation_agent", [python_bin, "main.py"], agent_dir, quiet=False)

    print("起動完了。Ctrl+Cで終了します。")

    try:
        while True:
            # どちらかが終了したら全体終了
            if tracker_proc is not None and tracker_proc.poll() is not None:
                print("trackerが終了したため、agentも停止します。")
                break
            if agent_proc is not None and agent_proc.poll() is not None:
                print("agentが終了したため、trackerも停止します。")
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        for p in (agent_proc, tracker_proc):
            if p is None:
                continue
            if p.poll() is None:
                try:
                    p.send_signal(signal.SIGINT)
                except Exception:
                    pass
        time.sleep(0.5)
        for p in (agent_proc, tracker_proc):
            if p is None:
                continue
            if p.poll() is None:
                p.terminate()
        print("終了しました。")


if __name__ == "__main__":
    main()
