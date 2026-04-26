"""カメラあり・TTSなしで、ターミナルだけ確認するデモランチャー。"""

from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


GREEN = "\033[92m"
GRAY = "\033[90m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"


def _python_bin(demo_root: Path) -> str:
    project_python = demo_root.parent / ".venv" / "bin" / "python"
    if project_python.exists():
        return str(project_python)
    return sys.executable


def _reader(name: str, proc: subprocess.Popen, out_queue: queue.Queue[tuple[str, str]]) -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
        out_queue.put((name, line.rstrip("\n")))


def _spawn(name: str, cmd: list[str], cwd: Path, env: dict[str, str]) -> subprocess.Popen:
    print(f"{YELLOW}起動中: {name} -> {' '.join(cmd)}{RESET}", flush=True)
    return subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


def _format_tracker_line(line: str, verbose: bool) -> str | None:
    if verbose:
        return line
    text = line.strip()
    if not text.startswith("{"):
        if "使用カメラ" in text or "トラッキング開始" in text:
            return text
        return None
    try:
        event = json.loads(text)
    except json.JSONDecodeError:
        return None
    event_type = event.get("event_type")
    if event_type == "tracker_status":
        return None
    if event_type == "environment_info_updated":
        env = event.get("environment_info", {})
        count = env.get("current_people_count", "?")
        new_ids = event.get("new_visitor_ids") or []
        returning_ids = event.get("returning_visitor_ids") or []
        if new_ids:
            return f"人を検出: {count}人 / new={','.join(new_ids)}"
        if returning_ids:
            return f"戻ってきた人を検出: {','.join(returning_ids)}"
        return f"人数変化: {count}人"
    if event_type == "person_info_updated":
        info = event.get("person_info", {})
        desc = info.get("clothing_description", "unknown")
        age = info.get("age_estimate", "unknown")
        expr = info.get("expression", "unknown")
        pose = info.get("pose_description", "unknown")
        items = ",".join(info.get("carried_items") or [])
        accessories = ",".join(info.get("accessories") or [])
        return (
            f"画像AI解析完了: {info.get('visitor_id')} / age={age} / clothes={desc} "
            f"/ pose={pose} / items={items or '-'} / accessories={accessories or '-'} / expression={expr}"
        )
    if event_type == "camera_scene_updated":
        scene = event.get("scene_info", {})
        return f"場面観察: {scene.get('summary', 'unknown')}"
    return None


def _print_line(name: str, line: str, verbose_tracker: bool) -> None:
    if not line:
        return
    if name == "tracker":
        line = _format_tracker_line(line, verbose_tracker)
        if not line:
            return
        color = GRAY
    elif name == "danya":
        color = GREEN
    else:
        color = RED
    print(f"{color}[{name}] {line}{RESET}", flush=True)


def _drain_output(out_queue: queue.Queue[tuple[str, str]], verbose_tracker: bool, raw: bool = False) -> None:
    while True:
        try:
            name, line = out_queue.get_nowait()
        except queue.Empty:
            break
        if raw:
            color = GREEN if name == "danya" else GRAY if name == "tracker" else RED
            print(f"{color}[{name}] {line}{RESET}", flush=True)
            continue
        _print_line(name, line, verbose_tracker)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="カメラ入力あり、TTSなし、ターミナル表示だけでダーニャを動かす"
    )
    parser.add_argument(
        "--real-vision",
        "--real-vlm",
        action="store_true",
        help="trackerのOpenAI画像AIを使う。指定しない場合は画像解析だけモック",
    )
    parser.add_argument(
        "--agent-llm",
        action="store_true",
        help="agentのOpenAI LLM発話生成を使う（通常のカメラモードでは未指定だとテンプレ発話）",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="DANYA_FORCE_TEMPLATE=1 を付け、テンプレ発話に固定（バックモードでLLMを切りたい時）",
    )
    parser.add_argument(
        "--status-interval-sec",
        type=float,
        default=2.0,
        help="tracker状態表示の間隔",
    )
    parser.add_argument(
        "--scene-interval-sec",
        type=float,
        default=25.0,
        help="カメラ全体の場面観察間隔",
    )
    parser.add_argument(
        "--verbose-tracker",
        action="store_true",
        help="trackerの生JSONや状態ログもすべて表示する",
    )
    parser.add_argument(
        "--debug-mode",
        "--back-mode",
        "--batch-mode",
        dest="debug_mode",
        action="store_true",
        help="カメラを使わず、Enterごとにダーニャ発話を生成する端末デバッグ（--back-mode は互換別名）",
    )
    args = parser.parse_args()
    if args.agent_llm and args.no_llm:
        print(f"{RED}--agent-llm と --no-llm は同時に使えません。{RESET}", flush=True)
        sys.exit(2)

    demo_root = Path(__file__).resolve().parents[1]
    services_dir = demo_root / "services"
    tracker_dir = services_dir / "visitor_tracker"
    agent_dir = services_dir / "conversation_agent"
    python_bin = _python_bin(demo_root)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if args.no_llm:
        env["DANYA_FORCE_TEMPLATE"] = "1"
    elif not args.debug_mode and not args.agent_llm:
        env["DANYA_FORCE_TEMPLATE"] = "1"

    if args.debug_mode:
        agent_cmd = [python_bin, "main.py", "--debug-mode"]
        try:
            proc = subprocess.Popen(agent_cmd, cwd=str(agent_dir), env=env)
            proc.wait()
        except KeyboardInterrupt:
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
        print(f"{YELLOW}終了しました。{RESET}", flush=True)
        return

    tracker_cmd = [
        python_bin,
        "main.py",
        "--terminal-only",
        "--status-interval-sec",
        str(args.status_interval_sec),
        "--scene-interval-sec",
        str(args.scene_interval_sec),
    ]
    if not args.real_vision:
        tracker_cmd.append("--mock-vision")

    agent_cmd = [python_bin, "main.py"]

    procs: list[subprocess.Popen] = []
    out_queue: queue.Queue[tuple[str, str]] = queue.Queue()
    try:
        tracker_proc = _spawn("tracker", tracker_cmd, tracker_dir, env)
        procs.append(tracker_proc)
        threading.Thread(target=_reader, args=("tracker", tracker_proc, out_queue), daemon=True).start()

        time.sleep(2.0)
        agent_proc = _spawn("danya", agent_cmd, agent_dir, env)
        procs.append(agent_proc)
        threading.Thread(target=_reader, args=("danya", agent_proc, out_queue), daemon=True).start()

        print(
            f"{YELLOW}起動完了。TTSは使いません。カメラの前に立つと、trackerログとダーニャ発話がここに出ます。Ctrl+Cで終了。{RESET}",
            flush=True,
        )
        while True:
            for proc in procs:
                if proc.poll() is not None:
                    time.sleep(0.1)
                    _drain_output(out_queue, args.verbose_tracker, raw=True)
                    raise RuntimeError(f"process exited: pid={proc.pid} code={proc.returncode}")
            try:
                name, line = out_queue.get(timeout=0.2)
                _print_line(name, line, args.verbose_tracker)
            except queue.Empty:
                pass
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"{RED}[runner] {exc}{RESET}", flush=True)
    finally:
        _drain_output(out_queue, args.verbose_tracker, raw=True)
        for proc in procs:
            if proc.poll() is None:
                try:
                    proc.send_signal(signal.SIGINT)
                except Exception:
                    pass
        time.sleep(0.5)
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        print(f"{YELLOW}終了しました。{RESET}", flush=True)


if __name__ == "__main__":
    main()
