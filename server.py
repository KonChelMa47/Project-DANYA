import io
import base64
import concurrent.futures
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, Form, Request
from fastapi.responses import JSONResponse, Response

# =========================================================
# 固定パス設定
# =========================================================
BASE_DIR = Path("/home/i2lab/daniil_ws/GPT-SoVITS")
GPT_MODEL_PATH = str(BASE_DIR / "GPT_weights_v2Pro" / "daniil_ja_20260423-e15.ckpt")
SOVITS_MODEL_PATH = str(BASE_DIR / "SoVITS_weights_v2Pro" / "daniil_ja_20260423_e8_s216.pth")
REF_AUDIO_PATH = str(Path("/home/i2lab/daniil_ws/dataset/emotion_wavs/happy_high_01.wav"))
REF_TEXT = "わあ、すごい！まじで嬉しい、ありがとう！"
REF_AUDIO_DIR = Path("/home/i2lab/daniil_ws/dataset/emotion_wavs")
DEFAULT_REF_ID = "happy_high"
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
REF_PRESETS = {
    "happy_normal": {
        "audio": "../wavs/recitation_008.wav",
        "text": "中国の外交団にアタッシェとして派遣された。",
    },
    "happy_high": {
        "audio": "happy_high.wav",
        "text": "やばい、本当に嬉しすぎて今ちょっとテンション上がりすぎてる！",
    },
    "angry_normal": {
        "audio": "angry_normal.wav",
        "text": "それはちょっとやり方が違うと思うから、一度ちゃんと話したい。",
    },
    "angry_high": {
        "audio": "angry_high.wav",
        "text": "なんでそんなことになるんだよ、ちゃんと考えてから行動してくれ！",
    },
    "sad_normal": {
        "audio": "sad_normal.wav",
        "text": "今日はなんだかうまくいかなくて、少し気持ちが沈んでるんだ。",
    },
    "sad_high": {
        "audio": "sad_high.wav",
        "text": "もうどうしたらいいのか分からなくて、本当に心が苦しいよ。",
    },
    "fear_normal": {
        "audio": "fear_normal.wav",
        "text": "ちょっと嫌な予感がしてて、なんだか落ち着かない気分なんだ。",
    },
    "fear_high": {
        "audio": "fear_high.wav",
        "text": "ちょっと待って、これ本当にやばいかもしれない、どうしよう。",
    },
    "surprised_normal": {
        "audio": "surprised_normal.wav",
        "text": "今のちょっと意外だったね、そんな展開になると思わなかった。",
    },
    "surprised_high": {
        "audio": "surprised_high.wav",
        "text": "え、ちょっと待って今の何！？全然予想してなかったんだけど！",
    },
}

# GPT-SoVITS の依存モデルパス
os.environ["bert_path"] = str(BASE_DIR / "GPT_SoVITS" / "pretrained_models" / "chinese-roberta-wwm-ext-large")
os.environ["cnhubert_base_path"] = str(BASE_DIR / "GPT_SoVITS" / "pretrained_models" / "chinese-hubert-base")
os.environ.setdefault("version", "v2")

os.chdir(BASE_DIR)
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav

app = FastAPI(title="GPT-SoVITS TTS Server")
i18n = I18nAuto()

REF_LANGUAGE = "日文"
TARGET_LANGUAGE = "日英混合"

# 直列化
INFER_CONCURRENCY = max(1, int(os.environ.get("DANYA_TTS_INFER_CONCURRENCY", "1")))
INFER_SEMAPHORE = threading.Semaphore(INFER_CONCURRENCY)
BATCH_WORKERS = max(1, int(os.environ.get("DANYA_TTS_BATCH_WORKERS", str(INFER_CONCURRENCY))))
MAX_BATCH_ITEMS = max(1, int(os.environ.get("DANYA_TTS_MAX_BATCH_ITEMS", "16")))

# 安全判定
MIN_SAMPLES = 8000          # 約0.25秒@32k
MIN_PEAK = 1e-4
MIN_RMS = 1e-5
MAX_RETRY_ON_SERVER = 2     # サーバー内部での再実行回数


def _ensure_exists(path: str, kind: str):
    if not Path(path).exists():
        raise FileNotFoundError(f"{kind} not found: {path}")


def _normalize_ref_id(value: Any, fallback: str | None = None, intensity: Any = None) -> str | None:
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


def _resolve_ref(ref_id: str | None) -> tuple[str, str, str]:
    resolved_ref_id = (_normalize_ref_id(ref_id, DEFAULT_REF_ID) or DEFAULT_REF_ID).strip()
    preset = REF_PRESETS.get(resolved_ref_id)
    if preset is None:
        for candidate_id, candidate in REF_PRESETS.items():
            audio_name = candidate["audio"]
            if resolved_ref_id in {audio_name, Path(audio_name).stem}:
                resolved_ref_id = candidate_id
                preset = candidate
                break
    if preset is None:
        available = ", ".join(sorted(REF_PRESETS))
        raise ValueError(f"Unknown ref_id '{resolved_ref_id}'. Available refs: {available}")

    audio_path = REF_AUDIO_DIR / preset["audio"]
    _ensure_exists(str(audio_path), "reference audio")
    return resolved_ref_id, str(audio_path), preset["text"]


def _refs_payload():
    refs = []
    for ref_id, preset in sorted(REF_PRESETS.items()):
        audio_path = REF_AUDIO_DIR / preset["audio"]
        refs.append(
            {
                "id": ref_id,
                "audio": preset["audio"],
                "path": str(audio_path),
                "text": preset["text"],
                "exists": audio_path.exists(),
            }
        )
    return {
        "default_ref_id": DEFAULT_REF_ID,
        "infer_concurrency": INFER_CONCURRENCY,
        "batch_workers": BATCH_WORKERS,
        "max_batch_items": MAX_BATCH_ITEMS,
        "refs": refs,
    }


def _load_models_once():
    _ensure_exists(GPT_MODEL_PATH, "GPT model")
    _ensure_exists(SOVITS_MODEL_PATH, "SoVITS model")
    _resolve_ref(DEFAULT_REF_ID)

    change_gpt_weights(gpt_path=GPT_MODEL_PATH)
    change_sovits_weights(sovits_path=SOVITS_MODEL_PATH)


def _is_silent_audio(audio: np.ndarray) -> bool:
    audio = np.asarray(audio)
    if audio.size == 0:
        return True
    if audio.ndim > 1:
        audio = audio.reshape(-1)
    if audio.size < MIN_SAMPLES:
        return True

    audio = audio.astype(np.float32, copy=False)
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(np.square(audio))))
    if peak < MIN_PEAK:
        return True
    if rms < MIN_RMS:
        return True
    return False


def _to_numpy(audio):
    try:
        import torch
        if isinstance(audio, torch.Tensor):
            return audio.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(audio)


def _run_one_infer(text: str, ref_id: str | None = None):
    resolved_ref_id, ref_audio_path, ref_text = _resolve_ref(ref_id)
    synthesis_result = get_tts_wav(
        ref_wav_path=ref_audio_path,
        prompt_text=ref_text,
        prompt_language=i18n(REF_LANGUAGE),
        text=text,
        text_language=i18n(TARGET_LANGUAGE),
        top_p=1,
        temperature=1,
    )
    result_list = list(synthesis_result)
    if not result_list:
        raise RuntimeError("No synthesis result returned from get_tts_wav")

    last_sampling_rate, last_audio_data = result_list[-1]
    last_audio_data = _to_numpy(last_audio_data)

    if last_sampling_rate is None or int(last_sampling_rate) <= 0:
        raise RuntimeError(f"Invalid sampling rate: {last_sampling_rate}")

    if _is_silent_audio(last_audio_data):
        raise RuntimeError(f"Generated silent audio with ref_id '{resolved_ref_id}'")

    return int(last_sampling_rate), last_audio_data


def _synthesize_to_wav_bytes(text: str, ref_id: str | None = None) -> bytes:
    last_err = None
    with INFER_SEMAPHORE:
        for attempt in range(1, MAX_RETRY_ON_SERVER + 1):
            try:
                sr, audio = _run_one_infer(text, ref_id=ref_id)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_path = tmp.name
                try:
                    sf.write(tmp_path, audio, sr)
                    data = Path(tmp_path).read_bytes()
                    if len(data) < 1000:
                        raise RuntimeError("WAV too small after write")
                    return data
                finally:
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
            except Exception as e:
                last_err = e
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                if attempt < MAX_RETRY_ON_SERVER:
                    continue
                raise last_err


def _extract_batch_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        raw_items = payload
        fallback_ref_id = None
    elif isinstance(payload, dict):
        raw_items = None
        for key in ("utterances", "sentences", "segments", "items"):
            if isinstance(payload.get(key), list):
                raw_items = payload[key]
                break
        fallback_ref_id = _normalize_ref_id(
            payload.get("ref_id") or payload.get("ref") or payload.get("emotion"),
            None,
            payload.get("intensity") or payload.get("level"),
        )
        if raw_items is None:
            text = str(payload.get("text") or payload.get("message") or "").strip()
            raw_items = [{"text": text}] if text else []
    else:
        raise ValueError("Batch payload must be a JSON object or array")

    items = []
    for index, item in enumerate(raw_items):
        if isinstance(item, str):
            text = item.strip()
            ref_id = fallback_ref_id
        elif isinstance(item, dict):
            text = str(item.get("text") or item.get("message") or "").strip()
            ref_id = _normalize_ref_id(
                item.get("ref_id") or item.get("ref") or item.get("emotion"),
                fallback_ref_id,
                item.get("intensity") or item.get("level"),
            )
        else:
            continue
        if not text:
            continue
        items.append(
            {
                "index": index,
                "text": text,
                "ref_id": ref_id or DEFAULT_REF_ID,
            }
        )

    if len(items) > MAX_BATCH_ITEMS:
        raise ValueError(f"Too many batch items: {len(items)} > {MAX_BATCH_ITEMS}")
    return items


def _synthesize_batch_item(item: dict[str, Any]) -> dict[str, Any]:
    resolved_ref_id, _, _ = _resolve_ref(item["ref_id"])
    wav_bytes = _synthesize_to_wav_bytes(item["text"], ref_id=resolved_ref_id)
    return {
        "ok": True,
        "index": item["index"],
        "text": item["text"],
        "ref_id": resolved_ref_id,
        "audio_format": "wav",
        "audio_base64": base64.b64encode(wav_bytes).decode("ascii"),
        "byte_length": len(wav_bytes),
    }


@app.on_event("startup")
def startup_event():
    _load_models_once()


@app.get("/health")
def health():
    return {
        "ok": True,
        "gpt_model": GPT_MODEL_PATH,
        "sovits_model": SOVITS_MODEL_PATH,
        "default_ref_id": DEFAULT_REF_ID,
        "refs": _refs_payload()["refs"],
        "infer_concurrency": INFER_CONCURRENCY,
        "batch_workers": BATCH_WORKERS,
        "max_batch_items": MAX_BATCH_ITEMS,
        "ref_language": REF_LANGUAGE,
        "target_language": TARGET_LANGUAGE,
    }


@app.get("/refs")
def refs():
    return _refs_payload()


@app.post("/tts")
async def tts(text: str = Form(...), ref_id: str | None = Form(None)):
    try:
        wav_bytes = _synthesize_to_wav_bytes(text, ref_id=ref_id)
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": 'inline; filename="tts.wav"'},
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


@app.post("/tts_batch")
async def tts_batch(request: Request):
    try:
        payload = await request.json()
        items = _extract_batch_items(payload)
        if not items:
            return JSONResponse(status_code=400, content={"ok": False, "error": "No text items"})

        results_by_index = {}
        worker_count = min(BATCH_WORKERS, len(items))
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {executor.submit(_synthesize_batch_item, item): item for item in items}
            for future in concurrent.futures.as_completed(future_map):
                item = future_map[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "ok": False,
                        "index": item["index"],
                        "text": item["text"],
                        "ref_id": item["ref_id"],
                        "error": str(exc),
                    }
                results_by_index[item["index"]] = result

        results = [results_by_index[item["index"]] for item in items]
        return {
            "ok": all(result.get("ok") for result in results),
            "count": len(results),
            "results": results,
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False, workers=1)
