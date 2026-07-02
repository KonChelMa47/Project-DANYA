from __future__ import annotations

import argparse
import base64
import bisect
import concurrent.futures
import html
import http.server
import io
import json
import math
import shutil
import struct
import threading
import time
import os
import queue
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pyglet
import soundfile as sf
from PIL import Image
from pyglet.gl import (
    GL_BLEND,
    GL_DEPTH_TEST,
    GL_LINEAR,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_REPEAT,
    GL_SRC_ALPHA,
    GL_TEXTURE0,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    glActiveTexture,
    glBlendFunc,
    glBindTexture,
    glClearColor,
    glDisable,
    glEnable,
    glTexParameteri,
)
from pyglet.graphics.shader import Shader, ShaderProgram

from tts_client import play_audio, synthesize_audio

WINDOW_TITLE = "DANYA Avatar"
DISPLAY_ROTATION_DEG = 90.0
WARP_GRID_COLS = 5
WARP_GRID_ROWS = 5
MAX_WARP_POINTS = 64
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parents[1]
BASE_DIR = PROJECT_ROOT
MODEL_PATH = PROJECT_ROOT / "assets" / "models" / "avatar.glb"
# アニメーション保存/再生の既定パス（runtime配下）
ANIMATION_DATA_PATH = BASE_DIR / "runtime" / "motion_records" / "animation_data.json"
# 旧パス互換: 既存データがある場合はこちらも読み込み対象にする
LEGACY_ANIMATION_DATA_PATH = BASE_DIR / "data" / "animation_data.json"
MOTION_RECORD_DIR = BASE_DIR / "runtime" / "motion_records"
EMOTION_MOTION_ALIASES = {
    "happy": "smile",
}
TTS_PROVIDER_URLS = {
    "gpt-sovits": "http://127.0.0.1:8000",
    "gptsovits": "http://127.0.0.1:8000",
    "cosyvoice": "http://192.168.73.239:8010",
}
DEFAULT_TTS_PROVIDER = os.environ.get("DANYA_TTS_PROVIDER", "gpt-sovits").strip().lower() or "gpt-sovits"
DEFAULT_TTS_SERVER = os.environ.get(
    "DANYA_TTS_SERVER",
    TTS_PROVIDER_URLS.get(DEFAULT_TTS_PROVIDER, TTS_PROVIDER_URLS["gpt-sovits"]),
)
DEFAULT_TTS_LANGUAGE = os.environ.get("DANYA_TTS_LANGUAGE", "ja").strip() or "ja"
DEFAULT_CONTROL_HOST = os.environ.get("DANYA_CONTROL_HOST", "0.0.0.0")
DEFAULT_CONTROL_PORT = int(os.environ.get("DANYA_CONTROL_PORT", "8020"))
DEFAULT_TTS_OUTPUT = BASE_DIR / "runtime" / "output.wav"
TTS_SEGMENT_DIR = BASE_DIR / "runtime" / "tts_segments"
TTS_SEGMENT_MAX_CHARS = max(12, int(os.environ.get("DANYA_TTS_SEGMENT_MAX_CHARS", "42")))
TTS_SEGMENT_GAP_SEC = max(0.0, float(os.environ.get("DANYA_TTS_SEGMENT_GAP_SEC", "0.04")))
TTS_RETRY_MAX = max(1, int(os.environ.get("DANYA_TTS_RETRY_MAX", "1")))
TTS_FALLBACK_MIN_CHARS = max(4, int(os.environ.get("DANYA_TTS_FALLBACK_MIN_CHARS", "14")))
TTS_FALLBACK_RETRY_MAX = max(1, int(os.environ.get("DANYA_TTS_FALLBACK_RETRY_MAX", "1")))
TTS_STABLE_REF_ID = os.environ.get("DANYA_TTS_STABLE_REF_ID", "").strip() or None
TTS_SEND_REF_ID = os.environ.get("DANYA_TTS_SEND_REF_ID", "0").strip().lower() in {"1", "true", "on", "yes"}
DEFAULT_AUDIO_DEVICE = os.environ.get("DANYA_AUDIO_DEVICE", "").strip() or None
DEFAULT_TTS_REF_ID = os.environ.get("DANYA_TTS_REF_ID", "").strip() or None
DEFAULT_LLM_OUTPUT_SERVER = os.environ.get("DANYA_LLM_OUTPUT_SERVER", "http://127.0.0.1:8767").strip()
DEFAULT_LLM_OUTPUT_INTERVAL = float(os.environ.get("DANYA_LLM_OUTPUT_INTERVAL", "20"))
DEFAULT_LLM_OUTPUT_DEBUG = os.environ.get("DANYA_LLM_OUTPUT_DEBUG", "1").strip().lower() not in {"0", "false", "off", "no"}
DEFAULT_EMOTION_INTENSITY = "normal"
EMOTION_REF_PREFIXES = {"happy", "angry", "sad", "surprised", "fear"}
EMOTION_REF_ALIASES = {
    "smile": "happy",
    "surprise": "surprised",
    "scared": "fear",
    "afraid": "fear",
}
EMOTION_LEVEL_ALIASES = {
    "mid": "normal",
    "low": "normal",
}
WINDOW_STATE_PATH = BASE_DIR / ".cache" / "window_state.json"
LIPSYNC_FRAME_SEC = 0.032
LIPSYNC_HOP_SEC = 0.010
SPEECH_FACE_BLEND_IN_SEC = 0.55
SPEECH_FACE_BLEND_OUT_SEC = 0.45
SPEECH_BASE_FACE_KEEP = 0.28
SPEECH_BASE_MOUTH_KEEP = 0.12

SPEECH_LIPSYNC_KEYS = {
    "jawopen",
    "mouthopen",
    "mouthpucker",
}

EXPRESSION_LIPSYNC_OVERRIDE_KEYS = {
    "jawopen",
    "mouthopen",
}

GAZE_KEYS = {
    "eyelookdownleft",
    "eyelookdownright",
    "eyelookinleft",
    "eyelookinright",
    "eyelookoutleft",
    "eyelookoutright",
    "eyelookupleft",
    "eyelookupright",
}
GAZE_MAX_WEIGHT = 0.34
GAZE_SMOOTH_GAIN = 0.045
GAZE_DECAY_GAIN = 0.035
BLINK_INTERVAL_MIN_SEC = 2.2
BLINK_INTERVAL_MAX_SEC = 5.8
BLINK_CLOSE_SEC = 0.075
BLINK_HOLD_SEC = 0.045
BLINK_OPEN_SEC = 0.13

BLINK_KEYS = {
    "eyesclosed",
    "eyeblinkleft",
    "eyeblinkright",
}

EYE_CONFLICT_KEYS = GAZE_KEYS | {
    "eyeslookup",
    "eyeslookdown",
    "eyesquintleft",
    "eyesquintright",
    "eyewideleft",
    "eyewideright",
}

MOUTH_KEYS = {
    "jawopen",
    "mouthopen",
    "mouthpucker",
    "mouthfunnel",
    "mouthclose",
    "mouthsmile",
    "mouthsmileleft",
    "mouthsmileright",
    "mouthstretchleft",
    "mouthstretchright",
    "mouthpressleft",
    "mouthpressright",
    "mouthlowerdownleft",
    "mouthlowerdownright",
    "mouthdimpleleft",
    "mouthdimpleright",
    "mouthrolllower",
    "mouthrollupper",
    "mouthshruglower",
    "mouthshrugupper",
    "mouthright",
    "mouthleft",
    "mouthroundleft",
    "mouthroundright",
    "mouthupperupleft",
    "mouthupperupright",
}

FACE_MESHES = {
    "Head_Mesh",
    "Eye_Mesh",
    "Teeth_Mesh",
    "Tongue_Mesh",
    "avaturn_hair_0",
    "avaturn_hair_1",
}

ALIAS_WEIGHTS = {
    "jawopen": ["mouthopen"],
    "mouthopen": ["jawopen"],
    "mouthsmile": ["mouthsmileleft", "mouthsmileright"],
    "mouthsmileleft": ["mouthsmile"],
    "mouthsmileright": ["mouthsmile"],
    "eyesclosed": ["eyeblinkleft", "eyeblinkright"],
    "eyeblinkleft": ["eyesclosed"],
    "eyeblinkright": ["eyesclosed"],
    "browinnerup": ["browraise"],
    "browraise": ["browinnerup"],
}

@dataclass
class MeshPart:
    name: str
    vertex_list: Any
    texture: Optional[Any]
    alpha_mode: str
    base_positions: np.ndarray
    base_normals: np.ndarray
    base_texcoords: np.ndarray
    indices: np.ndarray
    morph_names: list[str]
    morph_positions: np.ndarray
    morph_normals: np.ndarray
    mesh_transform: np.ndarray
    normal_transform: np.ndarray


@dataclass
class SpeechSegment:
    text: str
    ref_id: Optional[str] = None
    language: str = DEFAULT_TTS_LANGUAGE


class ControlWindow:
    def __init__(self, on_submit: Any, host: str = DEFAULT_CONTROL_HOST, port: int = DEFAULT_CONTROL_PORT) -> None:
        self.on_submit = on_submit
        self.host = host
        self.port = port
        self.httpd: Optional[http.server.ThreadingHTTPServer] = None
        self.thread: Optional[threading.Thread] = None

    def start(self) -> None:
        on_submit = self.on_submit

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                print(f"[CONTROL WEB] {self.client_address[0]} - {format % args}")

            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/health":
                    self._send_json({"ok": True, "service": "danya-avatar-control"})
                    return
                if self.path not in {"/", "/index.html"}:
                    self.send_error(404)
                    return
                body = ControlWindow._html().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802
                if self.path != "/speak":
                    self.send_error(404)
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    raw_body = self.rfile.read(length)
                    payload = json.loads(raw_body.decode("utf-8"))
                    text = str(payload.get("text") or payload.get("message") or "").strip()
                    if not text:
                        self._send_json({"ok": False, "error": "Text is empty"}, status=400)
                        return
                    on_submit(json.dumps(payload, ensure_ascii=False))
                    self._send_json({"ok": True, "queued": True})
                except Exception as exc:
                    self._send_json({"ok": False, "error": str(exc)}, status=500)

            def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        try:
            self.httpd = http.server.ThreadingHTTPServer((self.host, self.port), Handler)
        except Exception as exc:
            print(f"[CONTROL ERROR] Could not start web control UI: {exc}")
            return

        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        print(f"[CONTROL WEB] open http://127.0.0.1:{self.port}/")
        if self.host in {"0.0.0.0", "::"}:
            print(f"[CONTROL WEB] from phone: http://<this-pc-ip>:{self.port}/")

    def close_from_app(self) -> None:
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
            self.httpd = None

    @staticmethod
    def _html() -> str:
        sample_ja = html.escape("こんにちは。今日は日本語の音声合成を確認しています。")
        sample_ru = html.escape("Привет. Это проверка выбора русского языка.")
        return f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DANYA Control</title>
<style>
:root {{
  color-scheme: light;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  --ink: #18201f;
  --muted: #66736f;
  --line: #d7e3df;
  --paper: #fbfffd;
  --panel: rgba(255, 255, 255, 0.92);
  --mint: #00a88f;
  --mint-dark: #007f6d;
  --coral: #ff6b4a;
  --gold: #f2b705;
  --blue: #2176ff;
  --shadow: 0 18px 50px rgba(20, 70, 64, 0.16);
}}
* {{ box-sizing: border-box; }}
body {{
  min-height: 100vh;
  margin: 0;
  color: var(--ink);
  background:
    radial-gradient(circle at 12% 0%, rgba(0, 168, 143, 0.20), transparent 34%),
    radial-gradient(circle at 88% 16%, rgba(255, 107, 74, 0.20), transparent 30%),
    linear-gradient(135deg, #f3fbf7 0%, #fff8ec 54%, #eef8ff 100%);
}}
main {{ width: min(940px, 100%); margin: 0 auto; padding: 18px; }}
.topbar {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; margin: 4px 0 16px; }}
.brand {{ display: flex; align-items: center; gap: 12px; min-width: 0; }}
.mark {{
  width: 46px; height: 46px; border-radius: 8px;
  background: linear-gradient(135deg, var(--mint), #3ed7b5 54%, var(--gold));
  box-shadow: 0 10px 24px rgba(0, 168, 143, 0.28);
}}
h1 {{ margin: 0; font-size: 34px; line-height: 1; letter-spacing: 0; }}
.sub {{ color: var(--muted); font-size: 13px; margin-top: 4px; }}
.status-pill {{
  flex: 0 0 auto;
  border: 1px solid rgba(0, 127, 109, 0.22);
  background: rgba(255, 255, 255, 0.72);
  color: var(--mint-dark);
  border-radius: 999px;
  padding: 8px 11px;
  font-size: 13px;
  font-weight: 800;
  box-shadow: 0 8px 24px rgba(20, 70, 64, 0.08);
}}
.shell {{
  display: grid;
  grid-template-columns: minmax(0, 1.35fr) minmax(260px, 0.65fr);
  gap: 14px;
  align-items: start;
}}
.panel {{
  background: var(--panel);
  border: 1px solid rgba(215, 227, 223, 0.92);
  border-radius: 8px;
  box-shadow: var(--shadow);
  backdrop-filter: blur(18px);
}}
.composer {{ padding: 16px; }}
.side {{ display: grid; gap: 12px; }}
.tools {{ padding: 14px; }}
.field-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
label {{ display: block; font-weight: 900; font-size: 13px; margin: 0 0 7px; color: #25302d; }}
select, textarea, button, input {{ width: 100%; font: inherit; border-radius: 8px; }}
select, textarea {{
  border: 1px solid var(--line);
  background: rgba(255, 255, 255, 0.95);
  color: var(--ink);
  padding: 13px 12px;
  outline: none;
}}
select:focus, textarea:focus {{ border-color: var(--mint); box-shadow: 0 0 0 4px rgba(0, 168, 143, 0.15); }}
textarea {{ min-height: 260px; resize: vertical; line-height: 1.55; font-size: 17px; }}
.meter {{ display: flex; justify-content: space-between; gap: 10px; color: var(--muted); font-size: 12px; margin: 8px 0 12px; }}
.presets {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 12px; }}
.quick {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
button {{
  min-height: 48px;
  border: 0;
  background: var(--mint);
  color: #fff;
  font-weight: 900;
  padding: 13px 14px;
  cursor: pointer;
  touch-action: manipulation;
  box-shadow: 0 12px 28px rgba(0, 168, 143, 0.26);
}}
button:active {{ transform: translateY(1px); filter: brightness(0.96); }}
button:disabled {{ cursor: wait; opacity: 0.62; }}
button.secondary {{ background: #ffffff; color: var(--ink); border: 1px solid var(--line); box-shadow: none; }}
button.emotion {{ min-height: 42px; padding: 9px 10px; background: #ffffff; color: #26312e; border: 1px solid var(--line); box-shadow: none; font-size: 13px; }}
button.emotion.active {{ color: #fff; border-color: transparent; background: linear-gradient(135deg, var(--mint), var(--blue)); }}
#send {{
  min-height: 62px;
  font-size: 20px;
  background: linear-gradient(135deg, var(--coral), #ff8e51 48%, var(--gold));
  box-shadow: 0 16px 34px rgba(255, 107, 74, 0.28);
}}
.sendbar {{
  position: sticky;
  bottom: 0;
  margin: 0 -16px -16px;
  padding: 12px 16px 16px;
  background: linear-gradient(to bottom, rgba(255, 255, 255, 0), rgba(251, 255, 253, 0.96) 32%);
}}
#status {{ min-height: 24px; margin-top: 10px; color: var(--muted); font-weight: 800; }}
#status.sending {{ color: var(--blue); }}
#status.ok {{ color: var(--mint-dark); }}
#status.error {{ color: #c43328; }}
.stat {{ padding: 13px; }}
.stat-title {{ font-size: 12px; color: var(--muted); font-weight: 900; text-transform: uppercase; letter-spacing: 0; }}
.stat-value {{ margin-top: 6px; font-weight: 900; overflow-wrap: anywhere; }}
.history-list {{ display: grid; gap: 8px; margin-top: 10px; max-height: 340px; overflow: auto; }}
.history-item {{
  border: 1px solid var(--line);
  background: rgba(255, 255, 255, 0.78);
  border-radius: 8px;
  padding: 10px;
  color: #33403c;
  font-size: 13px;
  line-height: 1.45;
}}
.history-meta {{ color: var(--muted); font-weight: 800; margin-bottom: 4px; }}
@media (max-width: 760px) {{
  main {{ padding: 12px; }}
  .topbar {{ align-items: flex-start; }}
  .shell, .field-grid {{ grid-template-columns: 1fr; }}
  .side {{ grid-template-columns: 1fr 1fr; }}
  textarea {{ min-height: 230px; }}
  h1 {{ font-size: 28px; }}
}}
@media (max-width: 560px) {{
  .mark {{ width: 40px; height: 40px; }}
  .status-pill {{ display: none; }}
  .side, .presets, .quick {{ grid-template-columns: 1fr; }}
  textarea {{ min-height: 260px; font-size: 18px; }}
}}
</style>
</head>
<body>
<main>
<header class="topbar">
  <div class="brand">
    <div class="mark" aria-hidden="true"></div>
    <div>
      <h1>DANYA Voice Deck</h1>
      <div class="sub">スマホから文章を送って、そのままアバターに喋らせる</div>
    </div>
  </div>
  <div id="server" class="status-pill">checking...</div>
</header>

<div class="shell">
<section class="panel composer">
<div class="field-grid">
  <div>
    <label for="language">Language</label>
    <select id="language">
      <option value="ja">Japanese</option>
      <option value="ru">Russian</option>
    </select>
  </div>
  <div>
    <label for="emotion">Voice mood</label>
    <select id="emotion">
      <option value="neutral">Neutral</option>
      <option value="happy">Happy</option>
      <option value="sad">Sad</option>
      <option value="angry">Angry</option>
      <option value="surprised">Surprised</option>
      <option value="fear">Fear</option>
    </select>
  </div>
</div>

<div style="margin-top: 14px;">
  <label for="text">Script</label>
  <textarea id="text" autocomplete="off" spellcheck="false">{sample_ja}</textarea>
  <div class="meter"><span id="count">0 chars</span><span>送信後、TTSキューに入ります</span></div>
</div>

<div class="presets" aria-label="Emotion presets">
  <button type="button" class="emotion active" data-emotion="neutral">Neutral</button>
  <button type="button" class="emotion" data-emotion="happy">Happy</button>
  <button type="button" class="emotion" data-emotion="surprised">Surprised</button>
  <button type="button" class="emotion" data-emotion="sad">Sad</button>
  <button type="button" class="emotion" data-emotion="angry">Angry</button>
  <button type="button" class="emotion" data-emotion="fear">Fear</button>
</div>

<div class="sendbar">
<button id="send" type="button">Speak Now</button>
<div id="status"></div>
</div>
</section>

<aside class="side">
  <section class="panel tools">
    <div class="stat-title">Samples</div>
    <div class="quick">
      <button type="button" class="secondary" data-lang="ja" data-text="{sample_ja}">Japanese sample</button>
      <button type="button" class="secondary" data-lang="ru" data-text="{sample_ru}">Russian sample</button>
    </div>
  </section>
  <section class="panel stat">
    <div class="stat-title">Phone URL</div>
    <div id="phoneUrl" class="stat-value">このPCのIP:ポートで開いてください</div>
  </section>
  <section class="panel stat">
    <div class="stat-title">Recent sends</div>
    <div id="history" class="history-list"></div>
  </section>
</aside>
</div>
</main>
<script>
const text = document.getElementById('text');
const language = document.getElementById('language');
const emotion = document.getElementById('emotion');
const status = document.getElementById('status');
const server = document.getElementById('server');
const history = document.getElementById('history');
const send = document.getElementById('send');
const count = document.getElementById('count');
const phoneUrl = document.getElementById('phoneUrl');
const emotionButtons = Array.from(document.querySelectorAll('[data-emotion]'));
function setStatus(message, mode = '') {{
  status.className = mode;
  status.textContent = message;
}}
function updateCount() {{
  count.textContent = `${{text.value.length}} chars`;
}}
function setEmotion(value) {{
  emotion.value = value;
  emotionButtons.forEach((button) => button.classList.toggle('active', button.dataset.emotion === value));
}}
function addHistory(payload) {{
  const item = document.createElement('button');
  item.type = 'button';
  item.className = 'history-item';
  const when = new Date().toLocaleTimeString([], {{ hour: '2-digit', minute: '2-digit', second: '2-digit' }});
  item.innerHTML = `<div class="history-meta">${{when}} · ${{payload.language}} · ${{payload.emotion || 'neutral'}}</div><div>${{payload.text.replace(/[&<>]/g, (c) => ({{'&':'&amp;','<':'&lt;','>':'&gt;'}}[c]))}}</div>`;
  item.addEventListener('click', () => {{
    language.value = payload.language;
    setEmotion(payload.emotion || 'neutral');
    text.value = payload.text;
    updateCount();
    text.focus();
  }});
  history.prepend(item);
  while (history.children.length > 8) history.lastElementChild.remove();
}}
async function checkAvatar() {{
  try {{
    const res = await fetch('/health', {{ cache: 'no-store' }});
    const data = await res.json();
    server.textContent = data.ok ? 'Connected' : 'Control error';
  }} catch (err) {{
    server.textContent = 'Unreachable';
  }}
}}
phoneUrl.textContent = window.location.origin;
updateCount();
checkAvatar();
setInterval(checkAvatar, 5000);
text.addEventListener('input', updateCount);
emotion.addEventListener('change', () => setEmotion(emotion.value));
emotionButtons.forEach((button) => {{
  button.addEventListener('click', () => setEmotion(button.dataset.emotion));
}});
document.querySelectorAll('[data-text]').forEach((button) => {{
  button.addEventListener('click', () => {{
    language.value = button.dataset.lang;
    text.value = button.dataset.text;
    updateCount();
    text.focus();
  }});
}});
send.addEventListener('click', async () => {{
  const payload = {{ language: language.value, text: text.value.trim() }};
  if (emotion.value !== 'neutral') payload.emotion = emotion.value;
  if (!payload.text) {{
    setStatus('Text is empty', 'error');
    text.focus();
    return;
  }}
  send.disabled = true;
  send.textContent = 'Sending...';
  setStatus('Sending to avatar...', 'sending');
  try {{
    const res = await fetch('/speak', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify(payload),
    }});
    const data = await res.json();
    if (!res.ok || !data.ok) throw new Error(data.error || `request failed (${{res.status}})`);
    const queuedAt = new Date().toLocaleTimeString([], {{ hour: '2-digit', minute: '2-digit', second: '2-digit' }});
    setStatus(`Queued at ${{queuedAt}}`, 'ok');
    addHistory(payload);
    text.select();
  }} catch (err) {{
    setStatus(`Error: ${{err.message}}`, 'error');
  }} finally {{
    send.disabled = false;
    send.textContent = 'Speak Now';
  }}
}});
</script>
</body>
</html>"""


class LLMOutputReceiver(threading.Thread):
    TAG_PATTERN = re.compile(r"^\s*<([^>]+)>\s*(.*)$", re.DOTALL)

    def __init__(
        self,
        inbox: queue.Queue[str],
        server_url: str,
        interval_sec: float = DEFAULT_LLM_OUTPUT_INTERVAL,
    ) -> None:
        super().__init__(daemon=True)
        self.inbox = inbox
        self.server_url = server_url.rstrip("/")
        self.interval_sec = max(0.2, float(interval_sec))
        self.seq = 0
        self.running = True
        self.debug = DEFAULT_LLM_OUTPUT_DEBUG

    def run(self) -> None:
        try:
            import requests
        except Exception as exc:
            print(f"[LLM OUTPUT WARN] requests is not available: {exc}")
            return

        print(f"[LLM OUTPUT] polling {self.server_url}/api/output every {self.interval_sec:.1f}s")
        while self.running:
            try:
                response = requests.get(
                    f"{self.server_url}/api/output",
                    params={"since": self.seq},
                    timeout=min(25.0, max(2.0, self.interval_sec * 0.8)),
                )
                response.raise_for_status()
                data = response.json()
                outputs = data.get("outputs", [])
                latest_seq = int(data.get("latest_seq", self.seq))
                if self.debug:
                    print(f"[LLM OUTPUT] poll ok since={self.seq} count={len(outputs)} latest_seq={latest_seq}")
                for text in outputs:
                    if self.debug:
                        preview = str(text).replace("\n", " ")[:80]
                        print(f"[LLM OUTPUT] received: {preview}")
                    payload = self._payload_from_output(str(text))
                    if payload:
                        self.inbox.put(payload)
                self.seq = max(self.seq, latest_seq)
            except requests.exceptions.Timeout:
                print(f"[LLM OUTPUT WARN] timeout polling {self.server_url}/api/output")
            except Exception as exc:
                print(f"[LLM OUTPUT WARN] {exc}")

            slept = 0.0
            while self.running and slept < self.interval_sec:
                step = min(0.5, self.interval_sec - slept)
                time.sleep(step)
                slept += step

    def stop(self) -> None:
        self.running = False

    @classmethod
    def _payload_from_output(cls, output: str) -> str:
        text = output.strip()
        if not text:
            return ""

        parsed_segments = cls._segments_from_tagged_output(text)
        if not parsed_segments:
            return json.dumps({"source": "llm_output", "segments": [{"text": text}]}, ensure_ascii=False)

        return json.dumps({"source": "llm_output", "segments": parsed_segments}, ensure_ascii=False)

    @classmethod
    def _segments_from_tagged_output(cls, text: str) -> list[dict[str, str]]:
        segments: list[dict[str, str]] = []
        current: Optional[dict[str, str]] = None
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            match = cls.TAG_PATTERN.match(line)
            if not match:
                if current is not None:
                    current["text"] = f"{current['text']}\n{line}".strip()
                else:
                    segments.append({"text": line})
                continue

            tag = match.group(1).strip()
            body = match.group(2).strip()
            if not body:
                continue
            emotion, intensity = cls._split_emotion_tag(tag)
            current = {"text": body}
            if emotion:
                current["emotion"] = emotion
            if intensity:
                current["intensity"] = intensity
            segments.append(current)

        if segments:
            return segments

        match = cls.TAG_PATTERN.match(text)
        if not match:
            return []
        tag = match.group(1).strip()
        body = match.group(2).strip()
        if not body:
            return []

        emotion, intensity = cls._split_emotion_tag(tag)
        segment: dict[str, str] = {"text": body}
        if emotion:
            segment["emotion"] = emotion
        if intensity:
            segment["intensity"] = intensity
        return [segment]

    @staticmethod
    def _split_emotion_tag(tag: str) -> tuple[str, str]:
        normalized = tag.replace("-", "_").strip().lower()
        if "|" in normalized:
            emotion, intensity = normalized.split("|", 1)
            return emotion.strip(), intensity.strip()

        parts = [part for part in normalized.split("_") if part]
        if len(parts) >= 2 and parts[-1] in {"high", "normal", "mid", "low"}:
            return "_".join(parts[:-1]), parts[-1]
        return normalized, ""


class GLBAvatar:
    def __init__(self, glb_path: Path) -> None:
        if not glb_path.exists():
            raise FileNotFoundError(f"Missing GLB model: {glb_path}")
        self.data = glb_path.read_bytes()
        self.json_data, self.bin_chunk = self._parse_glb(self.data)
        self.nodes = self.json_data.get("nodes", [])
        self.meshes = self.json_data.get("meshes", [])
        self.materials = self.json_data.get("materials", [])
        self.textures = self.json_data.get("textures", [])
        self.images = self.json_data.get("images", [])
        self.scene_index = int(self.json_data.get("scene", 0))
        self.display_rotation = self._rotation_z_matrix(math.radians(DISPLAY_ROTATION_DEG))
        self.node_world_matrices = self._compute_node_world_matrices()
        self.parts = self._build_mesh_parts()
        self._center_and_scale_model()

    @staticmethod
    def _parse_glb(data: bytes) -> tuple[dict[str, Any], bytes]:
        magic, version, _length = struct.unpack_from("<III", data, 0)
        if magic != 0x46546C67 or version != 2:
            raise ValueError("Unsupported GLB file")
        offset = 12
        json_chunk = None
        bin_chunk = b""
        while offset < len(data):
            chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
            offset += 8
            chunk = data[offset : offset + chunk_length]
            offset += chunk_length
            if chunk_type == 0x4E4F534A:
                json_chunk = chunk
            elif chunk_type == 0x004E4942:
                bin_chunk = chunk
        if json_chunk is None:
            raise ValueError("GLB missing JSON chunk")
        return json.loads(json_chunk.decode("utf-8")), bin_chunk

    @staticmethod
    def _component_dtype(component_type: int) -> np.dtype[Any]:
        mapping = {
            5120: np.int8,
            5121: np.uint8,
            5122: np.int16,
            5123: np.uint16,
            5125: np.uint32,
            5126: np.float32,
        }
        return np.dtype(mapping[component_type]).newbyteorder("<")

    @staticmethod
    def _num_components(type_name: str) -> int:
        return {
            "SCALAR": 1,
            "VEC2": 2,
            "VEC3": 3,
            "VEC4": 4,
            "MAT4": 16,
        }[type_name]

    def _read_accessor(self, accessor_index: int) -> np.ndarray:
        accessor = self.json_data["accessors"][accessor_index]
        buffer_view = self.json_data["bufferViews"][accessor["bufferView"]]
        dtype = self._component_dtype(accessor["componentType"])
        count = int(accessor["count"])
        components = self._num_components(accessor["type"])
        base_offset = int(buffer_view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
        byte_stride = int(buffer_view.get("byteStride", components * dtype.itemsize))
        normalized = bool(accessor.get("normalized", False))

        if byte_stride == components * dtype.itemsize:
            arr = np.frombuffer(
                self.bin_chunk,
                dtype=dtype,
                count=count * components,
                offset=base_offset,
            ).reshape(count, components)
        else:
            arr = np.ndarray(
                shape=(count, components),
                dtype=dtype,
                buffer=self.bin_chunk,
                offset=base_offset,
                strides=(byte_stride, dtype.itemsize),
            )
        arr = np.array(arr, copy=True)

        if normalized and arr.dtype.kind in {"i", "u"}:
            if arr.dtype.kind == "u":
                arr = arr.astype(np.float32) / np.iinfo(arr.dtype).max
            else:
                info = np.iinfo(arr.dtype)
                arr = np.maximum(arr.astype(np.float32) / max(abs(info.min), info.max), -1.0)
        return arr

    def _decode_image(self, image_index: int) -> Image.Image:
        image = self.images[image_index]
        if "bufferView" in image:
            view = self.json_data["bufferViews"][image["bufferView"]]
            start = int(view.get("byteOffset", 0))
            end = start + int(view["byteLength"])
            payload = self.bin_chunk[start:end]
        elif "uri" in image:
            uri = image["uri"]
            if uri.startswith("data:"):
                payload = base64.b64decode(uri.split(",", 1)[1])
            else:
                payload = Path(uri).read_bytes()
        else:
            raise ValueError("Unsupported image source")
        return Image.open(io.BytesIO(payload)).convert("RGBA")

    def _load_texture(self, image_index: Optional[int]) -> Optional[Any]:
        if image_index is None:
            return None
        pil_image = self._decode_image(image_index)
        raw = pil_image.tobytes()
        texture = pyglet.image.ImageData(
            pil_image.width,
            pil_image.height,
            "RGBA",
            raw,
            pitch=-pil_image.width * 4,
        ).get_texture()
        glBindTexture(GL_TEXTURE_2D, texture.id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        return texture

    @staticmethod
    def _translation_matrix(t: np.ndarray) -> np.ndarray:
        m = np.eye(4, dtype=np.float32)
        m[:3, 3] = t[:3]
        return m

    @staticmethod
    def _scale_matrix(s: np.ndarray) -> np.ndarray:
        m = np.eye(4, dtype=np.float32)
        m[0, 0], m[1, 1], m[2, 2] = s[:3]
        return m

    @staticmethod
    def _rotation_z_matrix(angle: float) -> np.ndarray:
        c = math.cos(angle)
        s = math.sin(angle)
        return np.array(
            [
                [c, -s, 0.0, 0.0],
                [s, c, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _quat_matrix(q: np.ndarray) -> np.ndarray:
        x, y, z, w = q
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        return np.array(
            [
                [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy), 0],
                [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx), 0],
                [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy), 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

    def _local_matrix(self, node: dict[str, Any]) -> np.ndarray:
        if "matrix" in node:
            return np.array(node["matrix"], dtype=np.float32).reshape(4, 4).T
        matrix = np.eye(4, dtype=np.float32)
        if "translation" in node:
            matrix = matrix @ self._translation_matrix(np.array(node["translation"], dtype=np.float32))
        if "rotation" in node:
            matrix = matrix @ self._quat_matrix(np.array(node["rotation"], dtype=np.float32))
        if "scale" in node:
            matrix = matrix @ self._scale_matrix(np.array(node["scale"], dtype=np.float32))
        return matrix

    def _compute_node_world_matrices(self) -> list[np.ndarray]:
        world = [np.eye(4, dtype=np.float32) for _ in self.nodes]

        def visit(index: int, parent: np.ndarray) -> None:
            local = self._local_matrix(self.nodes[index])
            world[index] = parent @ local
            for child in self.nodes[index].get("children", []):
                visit(child, world[index])

        scene = self.json_data.get("scenes", [])[self.scene_index]
        for root in scene.get("nodes", []):
            visit(root, np.eye(4, dtype=np.float32))
        return world

    def _material_texture(self, material_index: Optional[int]) -> tuple[Optional[Any], str]:
        if material_index is None:
            return None, "OPAQUE"
        material = self.materials[material_index]
        alpha_mode = material.get("alphaMode", "OPAQUE")
        pbr = material.get("pbrMetallicRoughness", {})
        tex_index = None
        if "baseColorTexture" in pbr:
            tex_index = pbr["baseColorTexture"].get("index")
        texture = None
        if tex_index is not None:
            image_index = self.textures[tex_index].get("source")
            texture = self._load_texture(image_index)
        return texture, alpha_mode

    def _build_mesh_parts(self) -> list[MeshPart]:
        vertex_shader = """
            #version 330 core
            in vec3 position;
            in vec3 normal;
            in vec2 texcoord;
            uniform mat4 model;
            uniform mat4 view;
            uniform mat4 projection;
            out vec3 v_normal;
            out vec2 v_texcoord;

            void main() {
                vec4 world_pos = model * vec4(position, 1.0);
                v_normal = mat3(transpose(inverse(model))) * normal;
                v_texcoord = vec2(texcoord.x, 1.0 - texcoord.y);
                gl_Position = projection * view * world_pos;
            }
        """
        fragment_shader = """
            #version 330 core
            in vec3 v_normal;
            in vec2 v_texcoord;
            uniform sampler2D tex0;
            uniform vec3 light_dir;
            uniform int use_texture;
            uniform vec4 color_tint;
            out vec4 fragColor;
            void main() {
                vec3 n = normalize(v_normal);
                float diff = max(dot(n, normalize(-light_dir)), 0.2);
                vec4 base = color_tint;
                if (use_texture == 1) {
                    base *= texture(tex0, v_texcoord);
                }
                float brightness = 0.82 + 0.38 * diff;
                fragColor = vec4(base.rgb * brightness, base.a);
            }
        """
        self.program = ShaderProgram(Shader(vertex_shader, "vertex"), Shader(fragment_shader, "fragment"))
        self.program.use()
        self.program["tex0"] = 0
        self.program["light_dir"] = (0.3, 0.7, 0.8)
        self.program["use_texture"] = 0
        self.program["color_tint"] = (1.0, 1.0, 1.0, 1.0)

        parts: list[MeshPart] = []
        for node_index, node in enumerate(self.nodes):
            mesh_index = node.get("mesh")
            if mesh_index is None:
                continue
            mesh = self.meshes[int(mesh_index)]
            mesh_name = node.get("name") or mesh.get("name") or f"mesh_{mesh_index}"
            if mesh_name not in FACE_MESHES:
                continue
            primitive = mesh["primitives"][0]
            attributes = primitive["attributes"]
            base_positions = self._read_accessor(attributes["POSITION"]).astype(np.float32)
            base_normals = self._read_accessor(attributes["NORMAL"]).astype(np.float32)
            if "TEXCOORD_0" in attributes:
                base_texcoords = self._read_accessor(attributes["TEXCOORD_0"]).astype(np.float32)
            else:
                base_texcoords = np.zeros((base_positions.shape[0], 2), dtype=np.float32)
            indices = self._read_accessor(int(primitive["indices"]))[:, 0].astype(np.uint32)

            morph_names = [name.lower() for name in mesh.get("extras", {}).get("targetNames", [])]
            morph_positions = []
            morph_normals = []
            for target in primitive.get("targets", []):
                if "POSITION" in target:
                    morph_positions.append(self._read_accessor(target["POSITION"]).astype(np.float32))
                else:
                    morph_positions.append(np.zeros_like(base_positions))
                if "NORMAL" in target:
                    morph_normals.append(self._read_accessor(target["NORMAL"]).astype(np.float32))
                else:
                    morph_normals.append(np.zeros_like(base_normals))
            if morph_positions:
                morph_positions_arr = np.stack(morph_positions, axis=0)
                morph_normals_arr = np.stack(morph_normals, axis=0)
            else:
                morph_positions_arr = np.zeros((0, *base_positions.shape), dtype=np.float32)
                morph_normals_arr = np.zeros((0, *base_normals.shape), dtype=np.float32)

            texture, alpha_mode = self._material_texture(primitive.get("material"))
            mesh_transform = self.node_world_matrices[node_index].astype(np.float32)
            normal_transform = np.linalg.inv(mesh_transform).T.astype(np.float32)

            vlist = self.program.vertex_list_indexed(
                int(base_positions.shape[0]),
                pyglet.gl.GL_TRIANGLES,
                indices.tolist(),
                position=("f", base_positions.reshape(-1)),
                normal=("f", base_normals.reshape(-1)),
                texcoord=("f", base_texcoords.reshape(-1)),
            )

            parts.append(
                MeshPart(
                    name=mesh_name,
                    vertex_list=vlist,
                    texture=texture,
                    alpha_mode=alpha_mode,
                    base_positions=base_positions,
                    base_normals=base_normals,
                    base_texcoords=base_texcoords,
                    indices=indices,
                    morph_names=morph_names,
                    morph_positions=morph_positions_arr,
                    morph_normals=morph_normals_arr,
                    mesh_transform=mesh_transform,
                    normal_transform=normal_transform,
                )
            )
        return parts

    def _center_and_scale_model(self) -> None:
        points = []
        for part in self.parts:
            pts = np.c_[part.base_positions, np.ones((part.base_positions.shape[0], 1), dtype=np.float32)]
            pts = (part.mesh_transform @ pts.T).T[:, :3]
            points.append(pts)
        if not points:
            self.model_matrix = np.eye(4, dtype=np.float32)
            return
        cloud = np.concatenate(points, axis=0)
        mins = cloud.min(axis=0)
        maxs = cloud.max(axis=0)
        center = (mins + maxs) * 0.5
        extent = float(np.max(maxs - mins))
        scale = 2.25 / extent if extent > 1e-6 else 1.0
        translate = np.eye(4, dtype=np.float32)
        translate[:3, 3] = -center
        scale_m = np.eye(4, dtype=np.float32)
        scale_m[:3, :3] *= scale
        self.model_matrix = self.display_rotation @ scale_m @ translate

    @staticmethod
    def _update_vertex_list(part: MeshPart, positions: np.ndarray, normals: np.ndarray) -> None:
        part.vertex_list.set_attribute_data("position", positions.astype(np.float32).reshape(-1))
        part.vertex_list.set_attribute_data("normal", normals.astype(np.float32).reshape(-1))

    def draw(
        self,
        weights: dict[str, float],
        view: np.ndarray,
        projection: np.ndarray,
    ) -> None:
        self.program.use()
        self.program["model"] = tuple(self.model_matrix.T.reshape(-1))
        self.program["view"] = tuple(view.T.reshape(-1))
        self.program["projection"] = tuple(projection.T.reshape(-1))
        self.program["light_dir"] = (0.3, 0.7, 0.8)
        for part in self.parts:
            self._update_part_for_draw(part, weights)
            self.program["use_texture"] = 1 if part.texture is not None else 0
            self.program["color_tint"] = (1.0, 1.0, 1.0, 1.0)
            if part.texture is not None:
                glActiveTexture(GL_TEXTURE0)
                glBindTexture(part.texture.target, part.texture.id)
            part.vertex_list.draw(pyglet.gl.GL_TRIANGLES)

    def _update_part_for_draw(self, part: MeshPart, weights: dict[str, float]) -> None:
        vertices = part.base_positions.copy()
        normals = part.base_normals.copy()
        if part.morph_names and part.morph_positions.size:
            for index, morph_name in enumerate(part.morph_names):
                weight = weights.get(morph_name, 0.0)
                if weight > 1e-4:
                    vertices += part.morph_positions[index] * weight
                    normals += part.morph_normals[index] * weight
        vertex_h = np.c_[vertices, np.ones((vertices.shape[0], 1), dtype=np.float32)]
        vertex_h = (part.mesh_transform @ vertex_h.T).T[:, :3]
        normal_matrix = part.normal_transform[:3, :3]
        normal_h = (normal_matrix @ normals.T).T
        normal_h = normal_h / np.linalg.norm(normal_h, axis=1, keepdims=True).clip(min=1e-6)
        self._update_vertex_list(part, vertex_h, normal_h)

class AvatarApp(pyglet.window.Window):
    def __init__(
        self,
        launch_terminal: bool = True,
        control_host: str = DEFAULT_CONTROL_HOST,
        control_port: int = DEFAULT_CONTROL_PORT,
        start_fullscreen: Optional[bool] = None,
        static_avatar: bool = False,
        llm_output_server: Optional[str] = DEFAULT_LLM_OUTPUT_SERVER,
        llm_output_interval: float = DEFAULT_LLM_OUTPUT_INTERVAL,
    ) -> None:
        state = self._load_window_state()
        fullscreen = bool(state.get("fullscreen", False)) if start_fullscreen is None else start_fullscreen
        config = pyglet.gl.Config(double_buffer=True, depth_size=24)
        super().__init__(
            caption=WINDOW_TITLE,
            width=int(state.get("width", 1280)),
            height=int(state.get("height", 720)),
            resizable=True,
            fullscreen=fullscreen,
            config=config,
            vsync=True,
        )
        self.set_mouse_visible(True)
        if "x" in state and "y" in state and not self.fullscreen:
            try:
                self.set_location(int(state["x"]), int(state["y"]))
            except Exception:
                pass

        self.avatar = GLBAvatar(MODEL_PATH)
        self.avatar_base_model_matrix = self.avatar.model_matrix.copy()
        self.mirror_display = False
        
        # モードと記録データの追加 ("IDLE", "TRACK", "RECORD", "PLAY")
        self.mode = "IDLE" 
        self.record_data = []
        self.record_start_time = 0.0
        self.play_data = []
        self.play_start_time = 0.0
        self.play_pause_elapsed = 0.0
        self.play_paused_for_speech = False
        self.speech_face_reset = 0.0
        self.speech_emotion_blend = 0.0
        self.speech_emotion_started_at = 0.0
        self.speech_emotion: Optional[str] = None
        self.animation_duration = 0.0  # アニメーション全体の長さ
        self.loop_transition_duration = 0.3  # ループ時の遷移時間（秒）
        self.last_frame_data: dict[str, Any] = {}  # ループ時のブレンド用
        self.expression_data = self._load_expression_recordings()
        self.static_avatar = static_avatar

        self.smoothed_weights: dict[str, float] = {}
        self.smoothed_gaze_weights: dict[str, float] = {}
        self.next_blink_time = self._schedule_next_blink(time.time())
        self.blink_started_at: Optional[float] = None
        self.tts_server = DEFAULT_TTS_SERVER
        self.tts_output = DEFAULT_TTS_OUTPUT
        self.audio_device = DEFAULT_AUDIO_DEVICE
        self.tts_ref_id = DEFAULT_TTS_REF_ID
        self.state_lock = threading.Lock()
        self.speech_queue: queue.Queue[Optional[list[SpeechSegment]]] = queue.Queue()
        self.control_inbox: queue.Queue[str] = queue.Queue()
        self.llm_receiver: Optional[LLMOutputReceiver] = None
        
        self.is_speaking = False
        self.speech_motion = 0.0
        
        # 姿勢情報
        self.head_yaw = 0.0
        self.head_pitch = 0.0
        self.head_roll = 0.0
        self.view_rotation_index = 0

        self.lipsync_active = False
        self.lipsync_start_time = 0.0
        self.lipsync_index = 0
        self.lipsync_timeline: list[tuple[float, dict[str, float]]] = []
        self.app_start = time.time()
        
        self.speech_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.speech_thread.start()

        if llm_output_server:
            self.llm_receiver = LLMOutputReceiver(
                self.control_inbox,
                llm_output_server,
                llm_output_interval,
            )
            self.llm_receiver.start()

        if not self.static_avatar and self._load_recording():
            self.mode = "PLAY"
            self.play_start_time = time.time()
            print(f"[MODE] PLAY auto-started from {ANIMATION_DATA_PATH.name}.")
        
        self.control_window: Optional[ControlWindow] = None
        if launch_terminal:
            self.control_window = ControlWindow(self.control_inbox.put, host=control_host, port=control_port)
            self.control_window.start()
            
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        pyglet.clock.schedule_interval(self.update, 1 / 60.0)

        print("\n--- 操作方法 ---")
        print("T キー: トラッキング機能は無効")
        print("R キー: 録画機能は無効")
        print("P キー: 再生 (Play) モードの 開始/停止")
        print("F / F11 キー: フルスクリーン切り替え")
        print(f"TTS provider: {DEFAULT_TTS_PROVIDER}")
        print(f"TTS server: {self.tts_server}")
        if self.llm_receiver is not None:
            print(f"LLM output: {self.llm_receiver.server_url}/api/output")
        print(f"control web: http://127.0.0.1:{control_port}/")
        print("control web/API: 入力文またはJSONをTTSで読み上げ")
        print("control API: 'ref happy_normal' や 'ref happy_high' のように参照音声を切り替え")
        print("----------------\n")

    def update(self, dt: float) -> None:
        now = time.time()
        target_weights: dict[str, float] = {}

        with self.state_lock:
            speaking_now = self.is_speaking
            speech_emotion = self.speech_emotion
            speech_emotion_started_at = self.speech_emotion_started_at
        target_motion = 1.0 if speaking_now else 0.0
        self.speech_motion += (target_motion - self.speech_motion) * 0.12
        if speaking_now:
            face_target = min(1.0, max(0.0, (now - speech_emotion_started_at) / SPEECH_FACE_BLEND_IN_SEC))
            face_gain = 0.16
        else:
            face_target = 0.0
            face_gain = 0.10
        self.speech_face_reset += (face_target - self.speech_face_reset) * face_gain

        has_expression = speech_emotion in self.expression_data
        target_expression_blend = 1.0 if speaking_now and has_expression else 0.0
        expression_gain = 0.11 if target_expression_blend > self.speech_emotion_blend else 0.08
        self.speech_emotion_blend += (target_expression_blend - self.speech_emotion_blend) * expression_gain
        if not speaking_now and self.speech_emotion_blend < 0.01:
            self.speech_emotion_blend = 0.0
            with self.state_lock:
                if not self.is_speaking:
                    self.speech_emotion = None

        target_yaw, target_pitch, target_roll = 0.0, 0.0, 0.0

        # 再生モード
        if self.mode == "PLAY":
            if not self.play_data:
                self.mode = "IDLE"
            else:
                if speaking_now:
                    if not self.play_paused_for_speech:
                        self.play_pause_elapsed = max(0.0, now - self.play_start_time)
                        self.play_paused_for_speech = True
                    elapsed = self.play_pause_elapsed
                else:
                    if self.play_paused_for_speech:
                        self.play_start_time = now - self.play_pause_elapsed
                        self.play_paused_for_speech = False
                    elapsed = now - self.play_start_time
                
                # ループ処理（自然な遷移付き）
                if self.animation_duration > 0 and elapsed > self.animation_duration:
                    # ループ時の遷移ゾーン内かチェック
                    transition_zone_start = self.animation_duration - self.loop_transition_duration
                    
                    if elapsed - self.animation_duration < self.loop_transition_duration:
                        # 遷移ゾーン内：最後のフレームから最初のフレームへ自然にブレンド
                        blend_progress = (elapsed - self.animation_duration) / self.loop_transition_duration
                        frame_data = self._blend_frame_data(
                            self.last_frame_data,
                            self.play_data[0],
                            blend_progress
                        )
                    else:
                        # 遷移ゾーン外：ループをリセット
                        self.play_start_time = now
                        elapsed = 0.0
                        frame_data = self._get_interpolated_frame(elapsed)
                else:
                    frame_data = self._get_interpolated_frame(elapsed)
                
                for k, v in frame_data["weights"].items():
                    if speaking_now and self._is_base_speech_mouth_key(k):
                        continue
                    if k in GAZE_KEYS:
                        self._set_layer_weight(target_weights, k, self._filtered_gaze_value(k, v))
                    else:
                        self._set_layer_weight(target_weights, k, self._base_motion_weight(k, v, speaking_now))

                if speaking_now:
                    self._soft_release_layer(target_weights, self.speech_face_reset)

                target_yaw = frame_data["pose"]["yaw"]
                target_pitch = frame_data["pose"]["pitch"]
                target_roll = frame_data["pose"]["roll"]

                if speaking_now:
                    pose_release = float(np.clip(self.speech_face_reset, 0.0, 1.0))
                    target_yaw *= 1.0 - pose_release
                    target_pitch *= 1.0 - pose_release
                    target_roll *= 1.0 - pose_release

        expression_pose = None
        if speech_emotion and self.speech_emotion_blend > 0.0:
            expression_elapsed = max(0.0, now - speech_emotion_started_at)
            expression_pose = self._apply_emotion_expression(
                speech_emotion,
                expression_elapsed,
                self.speech_emotion_blend,
                target_weights,
            )
        if expression_pose is not None:
            pose_blend = float(np.clip(self.speech_emotion_blend, 0.0, 1.0))
            target_yaw += (float(expression_pose.get("yaw", 0.0)) - target_yaw) * pose_blend
            target_pitch += (float(expression_pose.get("pitch", 0.0)) - target_pitch) * pose_blend
            target_roll += (float(expression_pose.get("roll", 0.0)) - target_roll) * pose_blend

        if speaking_now:
            lipsync_weights = self._get_lipsync_weights_from_timeline()
            if lipsync_weights:
                for key, value in lipsync_weights.items():
                    self._set_layer_weight(target_weights, key, value, replace=True)
            else:
                talk_wave = 0.30 + 0.55 * (0.5 + 0.5 * math.sin((now - self.app_start) * 12.0))
                self._set_layer_weight(target_weights, "jawopen", talk_wave, replace=True)
                self._set_layer_weight(target_weights, "mouthopen", talk_wave * 0.88, replace=True)

        if not self.static_avatar:
            for key, value in self._blink_weights(now, speaking_now).items():
                self._set_layer_weight(target_weights, key, value, replace=True)
            self._suppress_eye_conflicts_for_blink(target_weights)

        self.smoothed_weights = self._smooth_face_weights(self.smoothed_weights, target_weights, dt)

        # 姿勢をスムージングしながら適用
        head_gain = 0.2
        self.head_yaw += (target_yaw - self.head_yaw) * head_gain
        self.head_pitch += (target_pitch - self.head_pitch) * head_gain
        self.head_roll += (target_roll - self.head_roll) * head_gain

        self.avatar.model_matrix = self._make_head_neck_pose_matrix() @ self.avatar_base_model_matrix
        self._drain_external_commands()

    @staticmethod
    def _set_layer_weight(
        weights: dict[str, float],
        key: str,
        value: float,
        replace: bool = False,
    ) -> None:
        value = float(np.clip(value, 0.0, 1.0))
        if value <= 0.001:
            return
        if replace:
            weights[key] = value
        else:
            weights[key] = max(weights.get(key, 0.0), value)

    @staticmethod
    def _base_motion_weight(key: str, value: float, speaking_now: bool) -> float:
        value = float(np.clip(value, 0.0, 1.0))
        if key in GAZE_KEYS:
            return value
        if key in BLINK_KEYS:
            return min(value, 0.25)
        if key in MOUTH_KEYS:
            return value * (0.18 if speaking_now else 0.55)
        return value * (0.42 if speaking_now else 0.72)

    @staticmethod
    def _soft_release_layer(weights: dict[str, float], strength: float) -> None:
        strength = float(np.clip(strength, 0.0, 1.0))
        if strength <= 0.0:
            return
        keep_factor = 1.0 - 0.55 * strength
        for key in list(weights.keys()):
            if key in SPEECH_LIPSYNC_KEYS or key in BLINK_KEYS:
                continue
            if key in GAZE_KEYS:
                weights[key] *= 1.0 - 0.20 * strength
            else:
                weights[key] *= keep_factor
            if weights[key] < 0.005:
                weights.pop(key, None)

    @staticmethod
    def _suppress_eye_conflicts_for_blink(weights: dict[str, float]) -> None:
        blink = max(float(weights.get(key, 0.0)) for key in BLINK_KEYS)
        if blink <= 0.05:
            return

        if blink >= 0.72:
            for key in BLINK_KEYS:
                weights[key] = max(float(weights.get(key, 0.0)), 1.0)

        conflict_keep = max(0.0, 1.0 - blink * 1.25)
        for key in EYE_CONFLICT_KEYS:
            if key not in weights:
                continue
            weights[key] *= conflict_keep
            if weights[key] < 0.005:
                weights.pop(key, None)

    def _filtered_gaze_value(self, key: str, value: float) -> float:
        target = float(np.clip(value, 0.0, 1.0)) * GAZE_MAX_WEIGHT
        current = float(self.smoothed_gaze_weights.get(key, 0.0))
        gain = GAZE_SMOOTH_GAIN if target > current else GAZE_DECAY_GAIN
        filtered = current + (target - current) * gain
        if filtered < 0.005:
            self.smoothed_gaze_weights.pop(key, None)
            return 0.0
        self.smoothed_gaze_weights[key] = filtered
        return filtered

    @staticmethod
    def _smooth_face_weights(
        current: dict[str, float],
        target: dict[str, float],
        dt: float,
    ) -> dict[str, float]:
        dt_scale = max(0.25, min(2.5, dt * 60.0))
        next_weights: dict[str, float] = {}
        for key in set(current).union(target):
            cur = float(current.get(key, 0.0))
            tgt = float(target.get(key, 0.0))
            if key in BLINK_KEYS:
                gain_up, gain_down = 0.90, 0.55
            elif key in SPEECH_LIPSYNC_KEYS:
                gain_up, gain_down = 0.42, 0.30
            elif key in MOUTH_KEYS:
                gain_up, gain_down = 0.22, 0.14
            elif key in GAZE_KEYS:
                gain_up, gain_down = 0.16, 0.12
            else:
                gain_up, gain_down = 0.18, 0.10
            gain = gain_up if tgt > cur else gain_down
            alpha = 1.0 - (1.0 - gain) ** dt_scale
            value = cur + (tgt - cur) * alpha
            if value > 0.008:
                next_weights[key] = float(np.clip(value, 0.0, 1.0))
        return next_weights

    def _blink_weights(self, now: float, speaking_now: bool) -> dict[str, float]:
        if self.blink_started_at is None and now >= self.next_blink_time:
            self.blink_started_at = now

        if self.blink_started_at is None:
            return {}

        elapsed = now - self.blink_started_at
        close_sec = BLINK_CLOSE_SEC * (1.15 if speaking_now else 1.0)
        hold_sec = BLINK_HOLD_SEC
        open_sec = BLINK_OPEN_SEC * (1.10 if speaking_now else 1.0)
        total = close_sec + hold_sec + open_sec
        if elapsed >= total:
            self.blink_started_at = None
            self.next_blink_time = self._schedule_next_blink(now)
            return {}

        if elapsed <= close_sec:
            amount = elapsed / max(close_sec, 1e-6)
        elif elapsed <= close_sec + hold_sec:
            amount = 1.0
        else:
            amount = 1.0 - ((elapsed - close_sec - hold_sec) / max(open_sec, 1e-6))
        amount = float(np.clip(math.sin(amount * math.pi * 0.5), 0.0, 1.0))
        return {"eyesclosed": amount, "eyeblinkleft": amount, "eyeblinkright": amount}

    @staticmethod
    def _schedule_next_blink(now: float) -> float:
        return now + float(np.random.uniform(BLINK_INTERVAL_MIN_SEC, BLINK_INTERVAL_MAX_SEC))

    def _soft_reset_speech_face(self, strength: float) -> None:
        """話している間、ループ中断時の顔を自然な正面に戻す。"""
        strength = float(np.clip(strength, 0.0, 1.0))
        if strength <= 0.0:
            return

        keep = SPEECH_LIPSYNC_KEYS
        keep_factor = 1.0 - (1.0 - SPEECH_BASE_FACE_KEEP) * strength
        for key in list(self.smoothed_weights.keys()):
            if key in keep:
                continue
            self.smoothed_weights[key] *= keep_factor
            if self.smoothed_weights[key] < 0.01:
                self.smoothed_weights.pop(key, None)

    @staticmethod
    def _is_base_speech_mouth_key(key: str) -> bool:
        return key == "jawopen" or key.startswith("mouth") or key.startswith("tongue")

    def _suppress_base_mouth_motion(self, strength: float) -> None:
        strength = float(np.clip(strength, 0.0, 1.0))
        if strength <= 0.0:
            return
        keep_factor = 1.0 - (1.0 - SPEECH_BASE_MOUTH_KEEP) * strength
        for key in list(self.smoothed_weights.keys()):
            if not self._is_base_speech_mouth_key(key):
                continue
            self.smoothed_weights[key] *= keep_factor
            if self.smoothed_weights[key] < 0.01:
                self.smoothed_weights.pop(key, None)

    def _get_interpolated_frame(self, elapsed: float) -> dict[str, Any]:
        """再生データから経過時間に応じたフレームを補間して取得"""
        if not self.play_data:
            return {"weights": {}, "pose": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}}
        if len(self.play_data) == 1:
            return self.play_data[0]
            
        times = [d["time"] for d in self.play_data]
        idx = bisect.bisect_right(times, elapsed)
        
        if idx == 0:
            return self.play_data[0]
        if idx >= len(self.play_data):
            return self.play_data[-1]
            
        f0 = self.play_data[idx - 1]
        f1 = self.play_data[idx]
        
        t0, t1 = f0["time"], f1["time"]
        ratio = (elapsed - t0) / max((t1 - t0), 1e-6)
        
        interp_weights = {}
        for k in set(f0["weights"].keys()).union(f1["weights"].keys()):
            v0 = f0["weights"].get(k, 0.0)
            v1 = f1["weights"].get(k, 0.0)
            interp_weights[k] = v0 + (v1 - v0) * ratio
            
        interp_pose = {}
        for k in ["yaw", "pitch", "roll"]:
            v0 = f0["pose"][k]
            v1 = f1["pose"][k]
            interp_pose[k] = v0 + (v1 - v0) * ratio
            
        return {"weights": interp_weights, "pose": interp_pose}

    def _blend_frame_data(self, frame1: dict[str, Any], frame2: dict[str, Any], blend: float) -> dict[str, Any]:
        """２つのフレームデータをブレンド（blend=0で frame1、blend=1で frame2）"""
        blend = max(0.0, min(1.0, blend))
        
        # ウェイトをブレンド
        blended_weights = {}
        all_keys = set(frame1.get("weights", {}).keys()).union(frame2.get("weights", {}).keys())
        for k in all_keys:
            v1 = frame1.get("weights", {}).get(k, 0.0)
            v2 = frame2.get("weights", {}).get(k, 0.0)
            blended_weights[k] = v1 + (v2 - v1) * blend
        
        # ポーズをブレンド
        blended_pose = {}
        for k in ["yaw", "pitch", "roll"]:
            v1 = frame1.get("pose", {}).get(k, 0.0)
            v2 = frame2.get("pose", {}).get(k, 0.0)
            blended_pose[k] = v1 + (v2 - v1) * blend
        
        return {"weights": blended_weights, "pose": blended_pose}

    def _load_expression_recordings(self) -> dict[str, dict[str, Any]]:
        expression_data: dict[str, dict[str, Any]] = {}
        legacy_dir = BASE_DIR / "runtime" / "motion_record"
        candidate_dirs = [MOTION_RECORD_DIR, legacy_dir]
        for motion_dir in candidate_dirs:
            if not motion_dir.exists():
                continue
            for target_path in sorted(motion_dir.glob("*.json")):
                emotion = target_path.stem.lower()
                if emotion == "animation_data" or emotion in expression_data:
                    continue
                self._load_one_expression_recording(expression_data, emotion, target_path)

        for alias, target in EMOTION_MOTION_ALIASES.items():
            if alias not in expression_data and target in expression_data:
                expression_data[alias] = expression_data[target]
        return expression_data

    @staticmethod
    def _load_one_expression_recording(
        expression_data: dict[str, dict[str, Any]],
        emotion: str,
        target_path: Path,
    ) -> None:
        try:
            data = json.loads(target_path.read_text(encoding="utf-8"))
            frames = data.get("frames", [])
            if not frames:
                print(f"[MOTION WARN] Empty expression motion: {target_path}")
                return
            expression_data[emotion] = {
                "frames": frames,
                "duration": float(frames[-1].get("time", 0.0)),
            }
            print(f"[MOTION] Loaded {emotion} expression: {target_path.name}")
        except Exception as exc:
            print(f"[MOTION WARN] Failed to load {target_path}: {exc}")

    @staticmethod
    def _motion_key_from_ref_id(ref_id: Optional[str]) -> Optional[str]:
        ref = (ref_id or DEFAULT_TTS_REF_ID or "happy_high").strip().lower()
        if not ref:
            return None
        if ref.endswith(".wav"):
            ref = ref[:-4]
        emotion = ref.split("_", 1)[0]
        return EMOTION_MOTION_ALIASES.get(emotion, emotion)

    def _get_interpolated_motion_frame(
        self,
        frames: list[dict[str, Any]],
        elapsed: float,
        duration: float,
        loop: bool = True,
    ) -> dict[str, Any]:
        if not frames:
            return {"weights": {}, "pose": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}}
        if len(frames) == 1 or duration <= 0.0:
            return frames[0]

        if loop:
            transition = min(self.loop_transition_duration, max(duration * 0.2, 0.05))
            phase = elapsed % (duration + transition)
            if phase > duration:
                return self._blend_frame_data(
                    frames[-1],
                    frames[0],
                    (phase - duration) / max(transition, 1e-6),
                )
            elapsed = phase
        else:
            elapsed = float(np.clip(elapsed, 0.0, duration))

        times = [float(d.get("time", 0.0)) for d in frames]
        idx = bisect.bisect_right(times, elapsed)
        if idx == 0:
            return frames[0]
        if idx >= len(frames):
            return frames[-1]

        f0 = frames[idx - 1]
        f1 = frames[idx]
        t0, t1 = float(f0.get("time", 0.0)), float(f1.get("time", 0.0))
        ratio = (elapsed - t0) / max((t1 - t0), 1e-6)
        return self._blend_frame_data(f0, f1, ratio)

    def _apply_emotion_expression(
        self,
        emotion: str,
        elapsed: float,
        blend: float,
        target_weights: dict[str, float],
    ) -> Optional[dict[str, float]]:
        data = self.expression_data.get(emotion)
        if not data:
            return None
        frame = self._get_interpolated_motion_frame(
            data["frames"],
            elapsed,
            float(data.get("duration", 0.0)),
            loop=True,
        )
        for key, value in frame.get("weights", {}).items():
            if key in EXPRESSION_LIPSYNC_OVERRIDE_KEYS:
                continue
            expression_value = float(value) * float(np.clip(blend, 0.0, 1.0))
            if key in GAZE_KEYS:
                self._set_layer_weight(
                    target_weights,
                    key,
                    self._filtered_gaze_value(key, expression_value),
                )
            else:
                self._set_layer_weight(target_weights, key, expression_value)
        pose = frame.get("pose")
        if not isinstance(pose, dict):
            return None
        return {
            "yaw": float(pose.get("yaw", 0.0)),
            "pitch": float(pose.get("pitch", 0.0)),
            "roll": float(pose.get("roll", 0.0)),
        }

    def _emotion_from_ref_id(self, ref_id: Optional[str]) -> Optional[str]:
        return self._motion_key_from_ref_id(ref_id)

    def _get_lipsync_weights_from_timeline(self) -> Optional[dict[str, float]]:
        with self.state_lock:
            active = self.lipsync_active
            start_time = self.lipsync_start_time
            index = self.lipsync_index
            timeline = self.lipsync_timeline

        if not active or not timeline:
            return None

        elapsed = max(0.0, time.time() - start_time)
        while index + 1 < len(timeline) and timeline[index + 1][0] <= elapsed:
            index += 1
        index = min(index, len(timeline) - 1)
        current = timeline[index][1]

        with self.state_lock:
            self.lipsync_index = index
        return {key: float(value) for key, value in current.items()}

    def on_draw(self) -> None:
        glClearColor(0.0, 0.0, 0.0, 1.0)
        self.clear()
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        projection = self._make_projection_matrix()
        view = self._make_view_matrix()
        original_model_matrix = self.avatar.model_matrix
        if self.mirror_display:
            mirror = np.eye(4, dtype=np.float32)
            mirror[0, 0] = -1.0
            self.avatar.model_matrix = mirror @ original_model_matrix
        try:
            self.avatar.draw(
                self.smoothed_weights,
                view,
                projection,
            )
        finally:
            self.avatar.model_matrix = original_model_matrix

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        if symbol == pyglet.window.key.ESCAPE:
            self.close()
            return
        if symbol in {pyglet.window.key.F, pyglet.window.key.F11}:
            self.set_fullscreen(not self.fullscreen)
            self._save_window_state()
            print(f"[WINDOW] fullscreen {'on' if self.fullscreen else 'off'}")
            return
        if symbol == pyglet.window.key.V:
            self.view_rotation_index = (self.view_rotation_index + 1) % 4
            return
        
        if symbol == pyglet.window.key.T:
            self.mirror_display = not self.mirror_display
            print(f"[DISPLAY] mirror {'on' if self.mirror_display else 'off'}")
            return
            
        if symbol == pyglet.window.key.R:
            print("[MODE] RECORD is disabled.")
                
        if symbol == pyglet.window.key.P:
            if self.mode == "PLAY":
                self.mode = "IDLE"
                print("[MODE] PLAY stopped.")
            else:
                if self._load_recording():
                    self.mode = "PLAY"
                    self.play_start_time = time.time()
                    self.play_pause_elapsed = 0.0
                    self.play_paused_for_speech = False
                    print("[MODE] PLAY started (Looping).")
                else:
                    print(f"Failed to load {ANIMATION_DATA_PATH.name}. Please record (R key) first.")

    def _save_recording(self) -> None:
        """記録したデータをJSONファイルに保存"""
        if not self.record_data:
            return
        try:
            data_to_save = {"frames": self.record_data}
            ANIMATION_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
            ANIMATION_DATA_PATH.write_text(json.dumps(data_to_save, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"Error saving recording: {e}")

    def _load_recording(self) -> bool:
        """JSONファイルからアニメーションデータを読み込み"""
        candidate_paths = [ANIMATION_DATA_PATH, LEGACY_ANIMATION_DATA_PATH]
        target_path = next((p for p in candidate_paths if p.exists()), None)
        if target_path is None:
            return False
        try:
            data = json.loads(target_path.read_text(encoding="utf-8"))
            self.play_data = data.get("frames", [])
            # アニメーション全体の長さを計算
            if self.play_data:
                self.animation_duration = self.play_data[-1]["time"]
                self.last_frame_data = {
                    "weights": self.play_data[-1].get("weights", {}),
                    "pose": self.play_data[-1].get("pose", {"yaw": 0.0, "pitch": 0.0, "roll": 0.0})
                }
            return len(self.play_data) > 0
        except Exception as e:
            print(f"Error loading recording: {e}")
            return False

    def on_close(self) -> None:
        self._save_window_state()
        if self.llm_receiver is not None:
            self.llm_receiver.stop()
            self.llm_receiver = None
        if self.control_window is not None:
            self.control_window.close_from_app()
            self.control_window = None
        self.speech_queue.put(None)
        if self.speech_thread.is_alive():
            self.speech_thread.join(timeout=1.5)
        super().on_close()

    def on_move(self, x: int, y: int) -> None:
        if not self.fullscreen:
            self._save_window_state()

    def on_resize(self, width: int, height: int) -> None:
        if not self.fullscreen:
            self._save_window_state()

    def _save_window_state(self) -> None:
        WINDOW_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            x, y = self.get_location()
        except Exception:
            x, y = 0, 0
        state = {
            "x": int(x),
            "y": int(y),
            "width": int(self.width),
            "height": int(self.height),
            "fullscreen": bool(self.fullscreen),
        }
        WINDOW_STATE_PATH.write_text(json.dumps(state), encoding="utf-8")

    @staticmethod
    def _load_window_state() -> dict[str, Any]:
        if not WINDOW_STATE_PATH.exists():
            return {}
        try:
            return json.loads(WINDOW_STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _drain_external_commands(self) -> None:
        for _ in range(4):
            try:
                raw = self.control_inbox.get_nowait()
            except queue.Empty:
                return

            command, command_ref_id, command_segments = self._extract_command_payload(raw)
            if not command:
                continue
            if command.lower() in {"quit", "exit"}:
                pyglet.clock.schedule_once(lambda _dt: self.close(), 0.0)
                continue

            lower = command.lower()
            if lower in {"tts on", "tts start", "tts enable", "tts", "tts off", "tts stop", "tts disable"}:
                print("[TTS] control mode is always on")
                continue
            if lower.startswith("ref "):
                ref_id = command.split(maxsplit=1)[1].strip()
                self.tts_ref_id = None if ref_id.lower() in {"default", "none", "off"} else ref_id
                print(f"[TTS REF] {self.tts_ref_id or 'server default'}")
                continue

            print(f"[You] {command}")
            ref_id = command_ref_id or self.tts_ref_id
            if command_segments:
                segments = [
                    SpeechSegment(segment.text, segment.ref_id or ref_id, segment.language)
                    for segment in command_segments
                    if segment.text.strip()
                ]
                languages = sorted({segment.language for segment in segments})
                print(f"[TTS] queued {len(segments)} segment(s), language={','.join(languages)}")
                self.speech_queue.put(segments)
            else:
                print(f"[TTS] {command}")
                self.speech_queue.put([SpeechSegment(command, ref_id, DEFAULT_TTS_LANGUAGE)])

    @classmethod
    def _extract_command_payload(cls, raw: str) -> tuple[str, Optional[str], list[SpeechSegment]]:
        text = raw.strip()
        if not text:
            return "", None, []
        if text.startswith("{"):
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                return text, None, []
            ref_id = cls._normalize_ref_id(obj.get("ref_id") or obj.get("ref") or obj.get("emotion"))
            segments = cls._segments_from_payload(obj, ref_id)
            command_text = str(obj.get("text") or obj.get("message") or "").strip()
            if not command_text and segments:
                command_text = " ".join(segment.text for segment in segments)
            return command_text, ref_id, segments
        return text, None, []

    @classmethod
    def _segments_from_payload(cls, obj: dict[str, Any], fallback_ref_id: Optional[str]) -> list[SpeechSegment]:
        raw_segments = None
        for key in ("utterances", "sentences", "segments"):
            if isinstance(obj.get(key), list):
                raw_segments = obj[key]
                break

        if raw_segments is None:
            text = str(obj.get("text") or obj.get("message") or "").strip()
            if not text:
                return []
            ref_id = cls._normalize_ref_id(
                obj.get("ref_id") or obj.get("ref") or obj.get("emotion"),
                fallback_ref_id,
                obj.get("intensity") or obj.get("level"),
            )
            language = cls._normalize_language(obj.get("language") or obj.get("lang"))
            return [SpeechSegment(text, ref_id, language)]

        segments: list[SpeechSegment] = []
        for item in raw_segments:
            if isinstance(item, str):
                segment_text = item.strip()
                ref_id = fallback_ref_id
                language = cls._normalize_language(obj.get("language") or obj.get("lang"))
            elif isinstance(item, dict):
                segment_text = str(item.get("text") or item.get("message") or "").strip()
                ref_id = cls._normalize_ref_id(
                    item.get("ref_id") or item.get("ref") or item.get("emotion"),
                    fallback_ref_id,
                    item.get("intensity") or item.get("level"),
                )
                language = cls._normalize_language(
                    item.get("language") or item.get("lang") or obj.get("language") or obj.get("lang")
                )
            else:
                continue
            if segment_text:
                segments.append(SpeechSegment(segment_text, ref_id, language))
        return segments

    @staticmethod
    def _normalize_language(value: Any) -> str:
        language = str(value or DEFAULT_TTS_LANGUAGE).strip().lower()
        return language or DEFAULT_TTS_LANGUAGE

    @staticmethod
    def _normalize_ref_id(
        value: Any,
        fallback: Optional[str] = None,
        intensity: Any = None,
    ) -> Optional[str]:
        raw = str(value or "").strip().lower()
        if not raw:
            return fallback
        if raw.endswith(".wav"):
            raw = raw[:-4]
        if raw.endswith("_01"):
            raw = raw[:-3]
        parts = raw.replace("-", "_").split("_")
        emotion = EMOTION_REF_ALIASES.get(parts[0], parts[0])
        if emotion not in EMOTION_REF_PREFIXES:
            return raw
        explicit_intensity = next((part for part in parts[1:] if part in {"high", "normal", "mid", "low"}), None)
        level = str(intensity or explicit_intensity or DEFAULT_EMOTION_INTENSITY).strip().lower()
        level = EMOTION_LEVEL_ALIASES.get(level, level)
        if level not in {"high", "normal"}:
            level = DEFAULT_EMOTION_INTENSITY
        return f"{emotion}_{level}"

    def _build_lipsync_timeline(self, wav_path: Path) -> list[tuple[float, dict[str, float]]]:
        try:
            audio, sr = sf.read(str(wav_path), dtype="float32")
        except Exception:
            return []

        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if audio.size < 128:
            return []

        frame_len = max(int(sr * LIPSYNC_FRAME_SEC), 256)
        hop_len = max(int(sr * LIPSYNC_HOP_SEC), 80)
        if audio.size <= frame_len:
            return []

        win = np.hanning(frame_len).astype(np.float32)
        amp_ref = float(np.percentile(np.abs(audio), 95)) + 1e-6
        timeline: list[tuple[float, dict[str, float]]] = []

        for start in range(0, audio.size - frame_len, hop_len):
            frame = audio[start : start + frame_len] * win
            rms = float(np.sqrt(np.mean(np.square(frame))) + 1e-9)
            energy = float(np.clip(rms / amp_ref, 0.0, 1.0))

            if energy < 0.03:
                weights = {
                    "jawopen": 0.0,
                    "mouthopen": 0.0,
                    "mouthpucker": 0.0,
                }
                timeline.append((start / sr, weights))
                continue

            spec = np.abs(np.fft.rfft(frame))
            freqs = np.fft.rfftfreq(frame_len, d=1.0 / sr)
            band = (freqs >= 200.0) & (freqs <= 3000.0)
            mag = spec[band]
            f = freqs[band]
            if mag.size < 8:
                continue

            mag_smooth = np.convolve(mag, np.ones(5, dtype=np.float32) / 5.0, mode="same")
            top_idx = np.argpartition(mag_smooth, -6)[-6:]
            ranked = top_idx[np.argsort(mag_smooth[top_idx])[::-1]]
            cand = [float(f[i]) for i in ranked]
            if not cand:
                continue

            f1 = cand[0]
            f2 = cand[0]
            for val in cand[1:]:
                if abs(val - f1) >= 120.0:
                    f2 = val
                    break
            if f2 < f1:
                f1, f2 = f2, f1

            probs = self._estimate_vowel_probabilities(f1, f2)
            weights = self._vowel_probs_to_weights(probs, energy)
            timeline.append((start / sr, weights))

        return timeline

    @staticmethod
    def _estimate_vowel_probabilities(f1: float, f2: float) -> dict[str, float]:
        prototypes = {
            "a": (800.0, 1200.0),
            "i": (320.0, 2300.0),
            "u": (350.0, 1300.0),
            "e": (500.0, 2000.0),
            "o": (520.0, 900.0),
        }
        sigma_f1 = 220.0
        sigma_f2 = 420.0
        scores: dict[str, float] = {}
        for key, (pf1, pf2) in prototypes.items():
            d1 = (f1 - pf1) / sigma_f1
            d2 = (f2 - pf2) / sigma_f2
            scores[key] = math.exp(-0.5 * (d1 * d1 + d2 * d2))

        total = sum(scores.values()) + 1e-9
        return {k: v / total for k, v in scores.items()}

    @staticmethod
    def _vowel_probs_to_weights(probs: dict[str, float], energy: float) -> dict[str, float]:
        a = probs.get("a", 0.0)
        u = probs.get("u", 0.0)
        e = probs.get("e", 0.0)
        o = probs.get("o", 0.0)

        jaw = energy * (0.18 + 0.78 * a + 0.32 * o + 0.28 * e)
        mouth_open = energy * (0.12 + 0.84 * a + 0.42 * o + 0.34 * e)
        pucker = energy * (0.76 * u + 0.56 * o)

        return {
            "jawopen": float(np.clip(jaw, 0.0, 1.0)),
            "mouthopen": float(np.clip(mouth_open, 0.0, 1.0)),
            "mouthpucker": float(np.clip(pucker, 0.0, 1.0)),
        }

    def _speech_worker(self) -> None:
        while True:
            segments = self.speech_queue.get()
            if segments is None:
                return
            segments = [segment for segment in segments if segment.text.strip()]
            segments = self._split_speech_segments(segments)
            if not segments:
                continue

            batch_id = int(time.time() * 1000)
            TTS_SEGMENT_DIR.mkdir(parents=True, exist_ok=True)
            first_success_path: Optional[Path] = None
            failed_count = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                next_index = 0

                def submit_next() -> Optional[concurrent.futures.Future]:
                    nonlocal next_index
                    if next_index >= len(segments):
                        return None
                    segment = segments[next_index]
                    out_path = TTS_SEGMENT_DIR / f"tts_{batch_id}_{next_index:02d}.wav"
                    print(
                        f"[TTS] synth {next_index + 1}/{len(segments)} "
                        f"language={segment.language} server={self.tts_server}: {segment.text[:36]}"
                    )
                    future = executor.submit(self._synthesize_speech_segment, segment, out_path)
                    next_index += 1
                    return future

                future = submit_next()
                current_index = 0
                while future is not None:
                    ok, reason, synthesized_parts = future.result()
                    future = submit_next()
                    current_index += 1
                    if not ok:
                        print(f"[TTS ERROR] {reason[:110]}")
                        failed_count += 1
                        continue
                    for spoken_segment, spoken_path in synthesized_parts:
                        if first_success_path is None:
                            first_success_path = spoken_path
                        print(f"[TTS] play {current_index}/{len(segments)}")
                        self._play_speech_segment(spoken_segment, spoken_path)
                        if TTS_SEGMENT_GAP_SEC > 0:
                            time.sleep(TTS_SEGMENT_GAP_SEC)

            if failed_count:
                print(f"[TTS WARN] skipped {failed_count}/{len(segments)} chunk(s) after all client fallbacks")

            if first_success_path is not None:
                try:
                    shutil.copyfile(str(first_success_path), str(self.tts_output))
                except Exception:
                    pass

    @classmethod
    def _split_speech_segments(cls, segments: list[SpeechSegment]) -> list[SpeechSegment]:
        split_segments: list[SpeechSegment] = []
        for segment in segments:
            for text in cls._split_text_for_tts(segment.text, TTS_SEGMENT_MAX_CHARS):
                split_segments.append(SpeechSegment(text, segment.ref_id, segment.language))
        split_segments = cls._merge_short_speech_segments(split_segments, TTS_SEGMENT_MAX_CHARS)
        if len(split_segments) > len(segments):
            print(
                f"[TTS] split {len(segments)} segment(s) into {len(split_segments)} chunk(s) "
                f"(max {TTS_SEGMENT_MAX_CHARS} chars)"
            )
        return split_segments

    @staticmethod
    def _merge_short_speech_segments(
        segments: list[SpeechSegment],
        max_chars: int,
    ) -> list[SpeechSegment]:
        if not segments:
            return []
        merged: list[SpeechSegment] = []
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue
            if (
                merged
                and merged[-1].ref_id == segment.ref_id
                and merged[-1].language == segment.language
                and len(merged[-1].text) + len(text) + 1 <= max_chars
                and (len(merged[-1].text) < TTS_FALLBACK_MIN_CHARS or len(text) < TTS_FALLBACK_MIN_CHARS)
            ):
                merged[-1] = SpeechSegment(f"{merged[-1].text} {text}", segment.ref_id, segment.language)
            else:
                merged.append(SpeechSegment(text, segment.ref_id, segment.language))
        return merged

    @staticmethod
    def _split_text_for_tts(text: str, max_chars: int) -> list[str]:
        normalized = AvatarApp._normalize_text_for_tts_retry(text)
        if len(normalized) <= max_chars:
            return [normalized] if normalized else []

        def split_long_sentence(sentence: str) -> list[str]:
            pieces = [piece.strip() for piece in re.findall(r"[^、,，;；:：]+[、,，;；:：]?", sentence) if piece.strip()]
            if not pieces:
                pieces = [sentence]
            result: list[str] = []
            current_piece = ""

            def split_oversize_piece(value: str) -> list[str]:
                words = value.split()
                if len(words) > 1:
                    word_chunks: list[str] = []
                    current_words = ""
                    for word in words:
                        if len(word) > max_chars:
                            if current_words:
                                word_chunks.append(current_words)
                                current_words = ""
                            word_chunks.extend(word[start : start + max_chars] for start in range(0, len(word), max_chars))
                            continue
                        candidate = f"{current_words} {word}".strip()
                        if current_words and len(candidate) > max_chars:
                            word_chunks.append(current_words)
                            current_words = word
                        else:
                            current_words = candidate
                    if current_words:
                        word_chunks.append(current_words)
                    return word_chunks
                return [value[start : start + max_chars] for start in range(0, len(value), max_chars)]

            for piece in pieces:
                if len(piece) > max_chars:
                    if current_piece:
                        result.append(current_piece)
                        current_piece = ""
                    result.extend(split_oversize_piece(piece))
                    continue
                if current_piece and len(current_piece) + len(piece) > max_chars:
                    result.append(current_piece)
                    current_piece = ""
                current_piece += piece
            if current_piece:
                result.append(current_piece)
            return result

        sentence_parts = [
            part.strip()
            for part in re.findall(r"[^。．.!！？?؟]+[。．.!！？?؟]*", normalized)
            if part.strip()
        ]
        chunks: list[str] = []
        current = ""

        def flush_current() -> None:
            nonlocal current
            if current:
                chunks.append(current)
                current = ""

        for part in sentence_parts:
            if len(part) > max_chars:
                flush_current()
                chunks.extend(split_long_sentence(part))
                continue
            if current and len(current) + len(part) + 1 > max_chars:
                flush_current()
            current = f"{current} {part}".strip() if current else part
        flush_current()
        return chunks

    def _synthesize_speech_segment(
        self,
        segment: SpeechSegment,
        out_path: Path,
    ) -> tuple[bool, str, list[tuple[SpeechSegment, Path]]]:
        return self._synthesize_speech_segment_with_fallback(segment, out_path)

    def _synthesize_safe_batch(
        self,
        segments: list[SpeechSegment],
        batch_id: int,
    ) -> list[tuple[SpeechSegment, Path]]:
        safe_text = " ".join(self._normalize_text_for_tts_retry(segment.text) for segment in segments)
        safe_chunks = self._split_text_for_tts(safe_text, TTS_FALLBACK_MIN_CHARS)
        synthesized_parts: list[tuple[SpeechSegment, Path]] = []
        for index, chunk in enumerate(safe_chunks):
            if not chunk.strip():
                continue
            language = segments[0].language if segments else DEFAULT_TTS_LANGUAGE
            safe_segment = SpeechSegment(chunk, TTS_STABLE_REF_ID, language)
            safe_path = TTS_SEGMENT_DIR / f"tts_{batch_id}_safe_{index:02d}.wav"
            ok, reason = self._try_synthesize_variants(safe_segment, safe_path)
            if not ok:
                print(f"[TTS SAFE ERROR] {index + 1}/{len(safe_chunks)} {reason[:90]}")
                return []
            synthesized_parts.append((safe_segment, safe_path))
        return synthesized_parts

    def _synthesize_speech_segment_with_fallback(
        self,
        segment: SpeechSegment,
        out_path: Path,
        depth: int = 0,
    ) -> tuple[bool, str, list[tuple[SpeechSegment, Path]]]:
        ok, reason = self._try_synthesize_variants(segment, out_path, retry_max=TTS_RETRY_MAX)
        if ok:
            return True, "OK", [(segment, out_path)]

        normalized_text = self._normalize_text_for_tts_retry(segment.text)
        if normalized_text and normalized_text != segment.text:
            normalized_path = out_path.with_name(f"{out_path.stem}_clean{out_path.suffix}")
            normalized_segment = SpeechSegment(normalized_text, segment.ref_id, segment.language)
            ok_clean, reason_clean = self._try_synthesize_variants(normalized_segment, normalized_path)
            if ok_clean:
                print(f"[TTS RECOVER] normalized text: {segment.text[:28]}")
                return True, "OK", [(normalized_segment, normalized_path)]
            reason = f"{reason}; clean={reason_clean}"

        if segment.ref_id:
            default_path = out_path.with_name(f"{out_path.stem}_default{out_path.suffix}")
            default_segment = SpeechSegment(normalized_text or segment.text, None, segment.language)
            ok_default, reason_default = self._try_synthesize_variants(default_segment, default_path)
            if ok_default:
                print(f"[TTS RECOVER] default voice: {segment.text[:28]}")
                return True, "OK", [(default_segment, default_path)]
            reason = f"{reason}; default_ref={reason_default}"

        retry_chunks = self._split_text_for_tts(normalized_text or segment.text, self._fallback_chunk_size(segment.text))
        if len(retry_chunks) <= 1:
            return False, reason, []

        print(
            f"[TTS RECOVER] split failed chunk into {len(retry_chunks)} smaller chunk(s): "
            f"{segment.text[:36]}"
        )
        synthesized_parts: list[tuple[SpeechSegment, Path]] = []
        failures: list[str] = []
        for index, chunk_text in enumerate(retry_chunks):
            chunk_path = out_path.with_name(f"{out_path.stem}_r{depth}_{index:02d}{out_path.suffix}")
            chunk_segment = SpeechSegment(chunk_text, segment.ref_id, segment.language)
            ok_chunk, reason_chunk, chunk_parts = self._synthesize_speech_segment_with_fallback(
                chunk_segment,
                chunk_path,
                depth + 1,
            )
            if ok_chunk:
                synthesized_parts.extend(chunk_parts)
            else:
                failures.append(f"{index + 1}/{len(retry_chunks)}: {reason_chunk[:80]}")

        if failures:
            return False, "; ".join(failures), synthesized_parts
        return True, "OK", synthesized_parts

    def _try_synthesize_variants(
        self,
        segment: SpeechSegment,
        out_path: Path,
        retry_max: int = TTS_FALLBACK_RETRY_MAX,
    ) -> tuple[bool, str]:
        text = segment.text.strip()
        if not text:
            return False, "Text is empty"

        attempts: list[tuple[str, Optional[str]]] = []
        if TTS_STABLE_REF_ID:
            attempts.append((text, TTS_STABLE_REF_ID))
        attempts.append((text, segment.ref_id))
        soft_text = self._soften_text_for_tts(text)
        if soft_text != text:
            if TTS_STABLE_REF_ID:
                attempts.append((soft_text, TTS_STABLE_REF_ID))
            attempts.append((soft_text, segment.ref_id))

        for ref_id in self._fallback_ref_ids(segment.ref_id):
            attempts.append((soft_text, ref_id))

        seen: set[tuple[str, Optional[str]]] = set()
        reasons: list[str] = []
        for attempt_index, (attempt_text, ref_id) in enumerate(attempts):
            key = (attempt_text, ref_id)
            if key in seen:
                continue
            seen.add(key)
            attempt_path = out_path if attempt_index == 0 else out_path.with_name(
                f"{out_path.stem}_v{attempt_index}{out_path.suffix}"
            )
            ok, reason = synthesize_audio(
                server_url=self.tts_server,
                text=attempt_text,
                out_path=attempt_path,
                ref_id=self._tts_api_ref_id(ref_id),
                language=segment.language,
                retry_max=retry_max,
            )
            if ok:
                if attempt_path != out_path:
                    try:
                        shutil.copyfile(str(attempt_path), str(out_path))
                    except Exception:
                        return True, "OK"
                return True, "OK"
            reasons.append(f"ref={ref_id or 'default'} {reason[:70]}")
        return False, "; ".join(reasons)

    @staticmethod
    def _tts_api_ref_id(ref_id: Optional[str]) -> Optional[str]:
        if not TTS_SEND_REF_ID:
            return None
        return ref_id

    @staticmethod
    def _fallback_ref_ids(ref_id: Optional[str]) -> list[Optional[str]]:
        refs: list[Optional[str]] = []
        if ref_id and "_" in ref_id:
            emotion = ref_id.split("_", 1)[0]
            refs.append(f"{emotion}_normal")
        refs.extend([TTS_STABLE_REF_ID, None])
        result: list[Optional[str]] = []
        for ref in refs:
            if ref == ref_id or ref in result:
                continue
            result.append(ref)
        return result

    @staticmethod
    def _soften_text_for_tts(text: str) -> str:
        softened = text.replace("！", "。").replace("!", "。").replace("？", "。").replace("?", "。")
        softened = re.sub(r"[、,]+", "、", softened)
        softened = re.sub(r"[。]{2,}", "。", softened)
        return softened.strip()

    @staticmethod
    def _normalize_text_for_tts_retry(text: str) -> str:
        normalized = re.sub(r"<[^>\n]{1,32}>", "", text)
        normalized = re.sub(r"[\u200b-\u200f\u202a-\u202e]", "", normalized)
        normalized = re.sub(r"(?<![A-Za-z0-9])KIT(?![A-Za-z0-9])", "ケーアイティー", normalized)
        normalized = re.sub(r"(?<![A-Za-z0-9])AI(?![A-Za-z0-9])", "エーアイ", normalized)
        normalized = normalized.replace("…", "。").replace("―", "-").replace("〜", "ー")
        normalized = re.sub(r"[!?！？]{3,}", "！", normalized)
        normalized = re.sub(r"[。]{3,}", "。", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    @staticmethod
    def _fallback_chunk_size(text: str) -> int:
        current_len = len(text.strip())
        if current_len <= TTS_FALLBACK_MIN_CHARS * 2:
            return current_len
        return max(TTS_FALLBACK_MIN_CHARS, min(TTS_SEGMENT_MAX_CHARS - 1, math.ceil(current_len / 2)))

    def _play_speech_segment(self, segment: SpeechSegment, out_path: Path) -> None:
        timeline = self._build_lipsync_timeline(out_path)
        speech_emotion = self._emotion_from_ref_id(segment.ref_id)

        with self.state_lock:
            self.is_speaking = True
            self.speech_emotion = speech_emotion
            self.speech_emotion_started_at = time.time()
            self.lipsync_timeline = timeline
            self.lipsync_index = 0
            self.lipsync_start_time = self.speech_emotion_started_at
            self.lipsync_active = len(timeline) > 0

        ok_play, reason_play = play_audio(out_path, audio_device=self.audio_device)

        with self.state_lock:
            self.is_speaking = False
            self.lipsync_active = False
            self.lipsync_index = 0
            self.lipsync_timeline = []
        if not ok_play:
            print(f"[AUDIO ERROR] {reason_play[:110]}")

    def _make_head_neck_pose_matrix(self) -> np.ndarray:
        speak = float(self.speech_motion)

        mouth_open = float(np.clip(self.smoothed_weights.get("mouthopen", 0.0), 0.0, 1.0))
        speaking_nod = mouth_open * 0.06 * speak

        yaw = self.head_yaw
        pitch = self.head_pitch + speaking_nod
        roll = self.head_roll

        # Keep pose behavior consistent with face_motion_avatar.py.
        yaw_m = self._rotation_x_matrix(math.radians(-yaw * 0.6))
        pitch_m = self._rotation_y_matrix(math.radians(pitch * 0.6))
        roll_m = self._rotation_z_matrix(math.radians(roll * 0.4))
        head_rot = pitch_m @ yaw_m @ roll_m

        pivot = np.array([0.0, -0.06, 0.0], dtype=np.float32)
        return (
            self._translation_matrix(pivot)
            @ head_rot
            @ self._translation_matrix(-pivot)
        )

    @staticmethod
    def _translation_matrix(offset: np.ndarray) -> np.ndarray:
        m = np.eye(4, dtype=np.float32)
        m[:3, 3] = offset[:3]
        return m

    @staticmethod
    def _rotation_x_matrix(angle: float) -> np.ndarray:
        c = math.cos(angle)
        s = math.sin(angle)
        return np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, c, -s, 0.0],
                [0.0, s, c, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _rotation_y_matrix(angle: float) -> np.ndarray:
        c = math.cos(angle)
        s = math.sin(angle)
        return np.array(
            [
                [c, 0.0, s, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-s, 0.0, c, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _rotation_z_matrix(angle: float) -> np.ndarray:
        c = math.cos(angle)
        s = math.sin(angle)
        return np.array(
            [
                [c, -s, 0.0, 0.0],
                [s, c, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _decay_weights(current: dict[str, float]) -> dict[str, float]:
        next_weights: dict[str, float] = {}
        for key, value in current.items():
            value *= 0.93
            if value > 0.01:
                next_weights[key] = value
        return next_weights

    def _make_projection_matrix(self) -> np.ndarray:
        aspect = max(self.width, 1) / max(self.height, 1)
        fov = math.radians(26.0)
        near = 0.01
        far = 100.0
        f = 1.0 / math.tan(fov / 2.0)
        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = f / aspect
        proj[1, 1] = f
        proj[2, 2] = (far + near) / (near - far)
        proj[2, 3] = (2 * far * near) / (near - far)
        proj[3, 2] = -1.0
        return proj

    def _make_view_matrix(self) -> np.ndarray:
        base_eye = np.array([0.0, 0.10, 3.45], dtype=np.float32)
        target = np.array([0.0, 0.05, 0.0], dtype=np.float32)
        base_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        orbit_rad = math.radians(90.0 * float(self.view_rotation_index))
        rotation = self._rotation_z_matrix(orbit_rad)

        eye_offset = np.array(
            [
                base_eye[0] - target[0],
                base_eye[1] - target[1],
                base_eye[2] - target[2],
                1.0,
            ],
            dtype=np.float32,
        )

        # Portrait rotations need a little more distance so the full head stays in frame.
        if self.view_rotation_index % 2 == 1:
            eye_offset[:3] *= 1.22

        rotated_offset = (rotation @ eye_offset)[:3]
        rotated_up = (rotation @ np.array([base_up[0], base_up[1], base_up[2], 0.0], dtype=np.float32))[:3]
        eye = target + rotated_offset
        up = rotated_up / max(np.linalg.norm(rotated_up), 1e-6)
        forward = target - eye
        forward = forward / np.linalg.norm(forward)
        side = np.cross(forward, up)
        side = side / np.linalg.norm(side)
        up = np.cross(side, forward)

        view = np.eye(4, dtype=np.float32)
        view[0, :3] = side
        view[1, :3] = up
        view[2, :3] = -forward
        view[:3, 3] = -view[:3, :3] @ eye
        return view

def main() -> None:
    parser = argparse.ArgumentParser(description="DANYA avatar viewer and control endpoint")
    parser.add_argument(
        "--no-control-window",
        action="store_true",
        help="Do not auto-open the TTS control window",
    )
    parser.add_argument(
        "--no-control-terminal",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--control-host",
        default=DEFAULT_CONTROL_HOST,
        help="Web control bind host. Use 0.0.0.0 for phones on the same LAN.",
    )
    parser.add_argument(
        "--control-port",
        type=int,
        default=DEFAULT_CONTROL_PORT,
        help="Web control port",
    )
    parser.add_argument(
        "--fullscreen",
        action="store_true",
        help="Start the avatar window in fullscreen mode",
    )
    parser.add_argument(
        "--windowed",
        action="store_true",
        help="Start the avatar window in windowed mode, ignoring the saved fullscreen state",
    )
    parser.add_argument(
        "--static",
        action="store_true",
        help="Show a still avatar: disable recorded motion auto-play and idle blinking",
    )
    parser.add_argument(
        "--llm-output-server",
        default=DEFAULT_LLM_OUTPUT_SERVER,
        help="LLM output server URL for /api/output polling",
    )
    parser.add_argument(
        "--llm-output-interval",
        type=float,
        default=DEFAULT_LLM_OUTPUT_INTERVAL,
        help="Seconds between LLM output polls",
    )
    parser.add_argument("--no-llm-output", action="store_true", help="Disable LLM output polling")
    args = parser.parse_args()
    if args.fullscreen and args.windowed:
        parser.error("--fullscreen and --windowed cannot be used together")
    start_fullscreen = True if args.fullscreen else False if args.windowed else None
    llm_output_server = None if args.no_llm_output else args.llm_output_server

    AvatarApp(
        launch_terminal=not (args.no_control_window or args.no_control_terminal),
        control_host=args.control_host,
        control_port=args.control_port,
        start_fullscreen=start_fullscreen,
        static_avatar=args.static,
        llm_output_server=llm_output_server,
        llm_output_interval=args.llm_output_interval,
    )
    pyglet.app.run()

if __name__ == "__main__":
    main()




