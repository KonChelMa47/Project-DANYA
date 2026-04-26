"""conversation_agent 設定。"""

import os
from pathlib import Path

from dotenv import load_dotenv

_config_dir = Path(__file__).resolve().parent
load_dotenv(dotenv_path=_config_dir / ".env", override=False)
load_dotenv(dotenv_path=_config_dir.parent / "visitor_tracker" / ".env", override=False)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default

# ===== パス設定 =====
base_dir = _config_dir
tracker_log_dir = (base_dir / "../visitor_tracker/logs").resolve()
agent_log_dir = (base_dir / "dynamic_rag/event_logs").resolve()
static_rag_dir = base_dir / "static_rag"
dynamic_rag_dir = base_dir / "dynamic_rag"
self_improvement_log_dir = dynamic_rag_dir / "event_logs"

# ===== 動作設定 =====
global_vlm_timeout_sec = 30
active_speech_interval_sec = 3
idle_speech_interval_sec = 10
person_event_ttl_sec = _env_float("DANYA_PERSON_EVENT_TTL_SEC", 15.0)
camera_scene_event_ttl_sec = _env_float("DANYA_CAMERA_SCENE_EVENT_TTL_SEC", 35.0)
idle_short_min_sec = 20
idle_short_max_sec = 100
idle_long_interval_sec = 180
min_speech_busy_sec = _env_float("DANYA_MIN_SPEECH_BUSY_SEC", 3.0)
speech_char_sec = _env_float("DANYA_SPEECH_CHAR_SEC", 0.13)
speech_fixed_overhead_sec = _env_float("DANYA_SPEECH_FIXED_OVERHEAD_SEC", 1.5)
speech_wait_scale = _env_float("DANYA_SPEECH_WAIT_SCALE", 1.0)
speech_wait_max_sec = _env_float("DANYA_SPEECH_WAIT_MAX_SEC", 60.0)
long_speech_max_chars = 460
short_speech_max_chars = 150
max_recent_speeches = 10
max_recent_topics = 10
max_recent_openings = 12
max_visitor_topics = 12
special_speech_interval = _env_int("DANYA_SPECIAL_SPEECH_INTERVAL", 3)
rag_enabled = True
use_openai = True
debug = False

# ===== 発話配信用HTTPサーバー =====
speech_server_enabled = _env_bool("DANYA_SPEECH_SERVER", True)
speech_server_host = os.getenv("DANYA_SPEECH_SERVER_HOST", "127.0.0.1")
speech_server_port = _env_int("DANYA_SPEECH_SERVER_PORT", 8765)

# ===== LLM設定 =====
use_llm = os.getenv("DANYA_FORCE_TEMPLATE", "").lower() not in {"1", "true", "yes"}
llm_provider = "openai"
# gpt-5-mini is the first choice for "latest, cheap, good" short MC speech.
# If the API/account does not support it, llm_client falls back to these.
llm_model = os.getenv("DANYA_LLM_MODEL", "gpt-5-mini")
llm_fallback_models = [
    model.strip()
    for model in os.getenv("DANYA_LLM_FALLBACK_MODELS", "gpt-4.1-mini,gpt-4o-mini").split(",")
    if model.strip()
]
llm_timeout_sec = 10
llm_max_retries = 1
llm_temperature = 0.9
llm_max_output_chars = 1200
fallback_to_template = True
llm_debug = False

# ===== 状況判定しきい値 =====
people_count_crowd_threshold = 4
intro_dwell_sec = 10
deepen_dwell_sec = 30
leaving_risk_threshold = 0.7
