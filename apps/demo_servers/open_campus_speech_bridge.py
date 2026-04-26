from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.error import URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8767
DEFAULT_ALTERTALK_URL = "http://127.0.0.1:8765"
DEFAULT_POLL_SEC = 1.0
MAX_STORED_OUTPUTS = 32


class OutputStore:
    def __init__(self, altertalk_url: str, poll_sec: float) -> None:
        self.altertalk_url = altertalk_url.rstrip("/")
        self.poll_sec = poll_sec
        self._lock = threading.Lock()
        self._items: list[tuple[int, str]] = []
        self._latest_seq = 0
        self._source_sequence = 0
        self._last_error = ""
        self._last_seen_at = 0.0

    def append_from_altertalk(self, payload: dict[str, Any]) -> bool:
        try:
            source_sequence = int(payload.get("sequence", 0) or 0)
        except (TypeError, ValueError):
            source_sequence = 0
        tagged_text = _tagged_text_from_payload(payload)
        if not tagged_text:
            return False

        with self._lock:
            if source_sequence and source_sequence <= self._source_sequence:
                return False
            if not source_sequence and self._items and self._items[-1][1] == tagged_text:
                return False
            self._source_sequence = max(self._source_sequence, source_sequence)
            self._latest_seq += 1
            self._items.append((self._latest_seq, tagged_text))
            if len(self._items) > MAX_STORED_OUTPUTS:
                self._items = self._items[-MAX_STORED_OUTPUTS:]
            self._last_error = ""
            self._last_seen_at = time.time()
            return True

    def set_error(self, message: str) -> None:
        with self._lock:
            self._last_error = message

    def payload_since(self, since: int) -> dict[str, Any]:
        with self._lock:
            return {
                "outputs": [text for seq, text in self._items if seq > since],
                "latest_seq": self._latest_seq,
            }

    def health_payload(self) -> dict[str, Any]:
        with self._lock:
            return {
                "ok": True,
                "latest_seq": self._latest_seq,
                "stored_outputs": len(self._items),
                "source_url": self.altertalk_url,
                "source_sequence": self._source_sequence,
                "last_seen_at": self._last_seen_at,
                "last_error": self._last_error,
                "poll_sec": self.poll_sec,
            }


def _tagged_text_from_payload(payload: dict[str, Any]) -> str:
    tagged_text = str(payload.get("tagged_text") or payload.get("speech") or "").strip()
    if tagged_text:
        return tagged_text

    segments = payload.get("segments")
    if not isinstance(segments, list):
        text = str(payload.get("text") or "").strip()
        return text

    lines: list[str] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        emotion = str(segment.get("emotion") or "").strip().strip("<>")
        lines.append(f"<{emotion}>{text}" if emotion else text)
    return "\n".join(lines).strip()


def start_altertalk_poll_loop(store: OutputStore, stop_event: threading.Event) -> threading.Thread:
    def worker() -> None:
        while not stop_event.is_set():
            try:
                with urlopen(f"{store.altertalk_url}/latest.json", timeout=2.5) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                if store.append_from_altertalk(payload):
                    print(
                        f"[ALTERTALK BRIDGE] accepted source_seq={payload.get('sequence')}",
                        flush=True,
                    )
            except (OSError, URLError, json.JSONDecodeError) as exc:
                store.set_error(str(exc))
            stop_event.wait(store.poll_sec)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return thread


def make_handler(store: OutputStore) -> type[BaseHTTPRequestHandler]:
    class AlterTalkBridgeHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/api/health":
                self._send_json(store.health_payload())
                return

            if parsed.path == "/api/output":
                params = parse_qs(parsed.query)
                try:
                    since = int(params.get("since", ["0"])[0])
                except (TypeError, ValueError):
                    since = 0
                payload = store.payload_since(since)
                self._send_json(payload)
                return

            self._send_json({"ok": False, "error": "not found"}, status=404)

        def log_message(self, format: str, *args: Any) -> None:
            print(f"[ALTERTALK BRIDGE] {self.address_string()} - {format % args}", flush=True)

        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return AlterTalkBridgeHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Bridge Open Campus Demo v1 speech output to DANYA /api/output")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Bind host")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Bind port")
    parser.add_argument("--altertalk-url", default=DEFAULT_ALTERTALK_URL, help="Open Campus Demo v1 speech server base URL")
    parser.add_argument("--poll-sec", type=float, default=DEFAULT_POLL_SEC, help="Seconds between source polls")
    args = parser.parse_args()

    store = OutputStore(args.altertalk_url, max(0.2, args.poll_sec))
    stop_event = threading.Event()
    start_altertalk_poll_loop(store, stop_event)

    server = ThreadingHTTPServer((args.host, args.port), make_handler(store))
    print(f"[ALTERTALK BRIDGE] source: {args.altertalk_url}", flush=True)
    print(f"[ALTERTALK BRIDGE] listening on http://{args.host}:{args.port}", flush=True)
    print(f"[ALTERTALK BRIDGE] health: http://127.0.0.1:{args.port}/api/health", flush=True)
    print(f"[ALTERTALK BRIDGE] output: http://127.0.0.1:{args.port}/api/output?since=0", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[ALTERTALK BRIDGE] stopping")
    finally:
        stop_event.set()
        server.server_close()


if __name__ == "__main__":
    main()
