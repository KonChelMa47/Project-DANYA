"""外部システム向けの発話配信HTTPサーバー。"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

import config
from schemas import SpeechOutput


class SpeechBroadcaster:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._latest: dict[str, Any] = {
            "sequence": 0,
            "timestamp_iso": "",
            "text": "",
            "tagged_text": "",
            "speech": "",
            "segments": [],
            "topic": "",
            "mode": "",
            "target_visitor_id": None,
            "wait_sec": 0.0,
        }

    def publish(self, output: SpeechOutput, wait_sec: float) -> None:
        plain_text = "\n".join(seg.text.strip() for seg in output.segments if seg.text.strip())
        tagged_text = output.speech
        with self._condition:
            self._latest = {
                "sequence": int(self._latest["sequence"]) + 1,
                "timestamp_iso": datetime.now().astimezone().isoformat(timespec="seconds"),
                "text": plain_text,
                "tagged_text": tagged_text,
                "speech": tagged_text,
                "segments": [seg.model_dump() for seg in output.segments],
                "topic": output.topic,
                "mode": output.mode,
                "target_visitor_id": output.target_visitor_id,
                "wait_sec": wait_sec,
            }
            self._condition.notify_all()

    def latest(self) -> dict[str, Any]:
        with self._condition:
            return dict(self._latest)

    def wait_next(self, after_sequence: int, timeout_sec: float = 30.0) -> dict[str, Any]:
        with self._condition:
            self._condition.wait_for(lambda: int(self._latest["sequence"]) > after_sequence, timeout=timeout_sec)
            return dict(self._latest)


class _Handler(BaseHTTPRequestHandler):
    broadcaster: SpeechBroadcaster

    def _send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict[str, Any]) -> None:
        self._send_bytes(json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._send_bytes(b"ok\n", "text/plain; charset=utf-8")
            return
        if parsed.path == "/latest.txt":
            latest = self.broadcaster.latest()
            self._send_bytes(str(latest.get("tagged_text", "")).encode("utf-8"), "text/plain; charset=utf-8")
            return
        if parsed.path == "/latest_plain.txt":
            latest = self.broadcaster.latest()
            self._send_bytes(str(latest.get("text", "")).encode("utf-8"), "text/plain; charset=utf-8")
            return
        if parsed.path == "/latest.json":
            self._send_json(self.broadcaster.latest())
            return
        if parsed.path == "/stream":
            self._stream(text_only=True)
            return
        if parsed.path == "/stream.json":
            self._stream(text_only=False)
            return
        self._send_json(
            {
                "error": "not_found",
                "endpoints": ["/health", "/latest.txt", "/latest_plain.txt", "/latest.json", "/stream", "/stream.json"],
            }
        )

    def _stream(self, *, text_only: bool) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        sequence = int(self.broadcaster.latest()["sequence"])
        while True:
            latest = self.broadcaster.wait_next(sequence)
            next_sequence = int(latest["sequence"])
            if next_sequence <= sequence:
                self.wfile.write(b": keep-alive\n\n")
                self.wfile.flush()
                continue
            sequence = next_sequence
            data = str(latest.get("tagged_text", "")) if text_only else json.dumps(latest, ensure_ascii=False)
            escaped_data = data.replace("\n", "\ndata: ")
            packet = f"id: {sequence}\nevent: speech\ndata: {escaped_data}\n\n"
            self.wfile.write(packet.encode("utf-8"))
            self.wfile.flush()

    def log_message(self, _format: str, *args: Any) -> None:
        if config.debug:
            super().log_message(_format, *args)


def start_speech_server() -> SpeechBroadcaster:
    broadcaster = SpeechBroadcaster()
    if not config.speech_server_enabled:
        return broadcaster

    class Handler(_Handler):
        pass

    Handler.broadcaster = broadcaster
    server = ThreadingHTTPServer((config.speech_server_host, config.speech_server_port), Handler)
    thread = threading.Thread(target=server.serve_forever, name="danya-speech-server", daemon=True)
    thread.start()
    print(f"[speech_server] http://{config.speech_server_host}:{config.speech_server_port}")
    return broadcaster
