from __future__ import annotations

import argparse
import json
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse


DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8767
DEFAULT_INTERVAL_SEC = 20.0
DEFAULT_EMIT_IMMEDIATELY = False
MAX_STORED_OUTPUTS = 8

SAMPLE_OUTPUTS = [
    "<happy|mid>こんにちは、今日もええ感じに動いてるね。次は何を話そっか。",
    "<surprised_high>おっと、それはちょっと予想してなかった。けど面白い流れやね。",
    "<happy_high>やった、今の反応かなり自然に見える気がする！",
    "<sad_mid>うーん、少しだけ迷ってるけど、ゆっくり整えていけば大丈夫。",
    "<fear_mid>ちょっとだけ不安やけど、状況を確認しながら進めよう。",
    "<angry_mid>そこは少し気になるな。もう一回ちゃんと見直した方がよさそう。",
    "<happy|mid>あのまんま呼ぶん。そりゃ大胆やわ。ほいで、出身はどこなが。",
]


class OutputStore:
    def __init__(self, interval_sec: float) -> None:
        self._lock = threading.Lock()
        self._items: list[tuple[int, str]] = []
        self._latest_seq = 0
        self.interval_sec = interval_sec

    def append(self, text: str) -> int:
        with self._lock:
            self._latest_seq += 1
            self._items.append((self._latest_seq, text))
            if len(self._items) > MAX_STORED_OUTPUTS:
                self._items = self._items[-MAX_STORED_OUTPUTS:]
            return self._latest_seq

    def payload_since(self, since: int) -> dict[str, Any]:
        with self._lock:
            outputs = [text for seq, text in self._items if seq > since]
            return {
                "outputs": outputs,
                "latest_seq": self._latest_seq,
            }

    def health_payload(self) -> dict[str, Any]:
        with self._lock:
            return {
                "ok": True,
                "latest_seq": self._latest_seq,
                "stored_outputs": len(self._items),
                "interval_sec": self.interval_sec,
            }


def start_random_output_loop(
    store: OutputStore,
    interval_sec: float,
    stop_event: threading.Event,
    emit_immediately: bool,
) -> threading.Thread:
    def worker() -> None:
        if emit_immediately:
            seq = store.append(random.choice(SAMPLE_OUTPUTS))
            print(f"[LLM LOCAL] seq={seq} emitted", flush=True)

        while not stop_event.wait(interval_sec):
            seq = store.append(random.choice(SAMPLE_OUTPUTS))
            print(f"[LLM LOCAL] seq={seq} emitted", flush=True)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return thread


def make_handler(store: OutputStore) -> type[BaseHTTPRequestHandler]:
    class LLMOutputHandler(BaseHTTPRequestHandler):
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
                print(
                    f"[LLM LOCAL] output since={since} count={len(payload['outputs'])} latest_seq={payload['latest_seq']}",
                    flush=True,
                )
                self._send_json(payload)
                return

            self._send_json({"ok": False, "error": "not found"}, status=404)

        def log_message(self, format: str, *args: Any) -> None:
            print(f"[LLM LOCAL] {self.address_string()} - {format % args}", flush=True)

        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return LLMOutputHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Local dummy LLM output server for DANYA")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Bind host")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Bind port")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_SEC, help="Output interval seconds")
    parser.add_argument("--immediate", action="store_true", help="Emit the first dummy output immediately")
    args = parser.parse_args()

    interval_sec = max(1.0, args.interval)
    store = OutputStore(interval_sec)
    stop_event = threading.Event()
    start_random_output_loop(
        store,
        interval_sec,
        stop_event,
        emit_immediately=args.immediate or DEFAULT_EMIT_IMMEDIATELY,
    )

    server = ThreadingHTTPServer((args.host, args.port), make_handler(store))
    print(f"[LLM LOCAL] emitting dummy output every {interval_sec:.0f}s", flush=True)
    print(f"[LLM LOCAL] listening on http://{args.host}:{args.port}", flush=True)
    print(f"[LLM LOCAL] health: http://127.0.0.1:{args.port}/api/health", flush=True)
    print(f"[LLM LOCAL] output: http://127.0.0.1:{args.port}/api/output?since=0", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[LLM LOCAL] stopping")
    finally:
        stop_event.set()
        server.server_close()


if __name__ == "__main__":
    main()
