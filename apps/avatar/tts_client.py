import argparse
import io
import os
import subprocess
import time
from pathlib import Path
from shutil import which

import numpy as np
import requests
import soundfile as sf

# --- 判定基準を現実的な数値に引き上げ ---
RETRY_MAX = int(os.environ.get("DANYA_TTS_RETRY_MAX", "2"))
RETRY_WAIT_SEC = float(os.environ.get("DANYA_TTS_RETRY_WAIT_SEC", "0.3"))
MIN_DURATION_SEC = float(os.environ.get("DANYA_TTS_MIN_DURATION_SEC", "0.35"))
MIN_PEAK = float(os.environ.get("DANYA_TTS_MIN_PEAK", "0.003"))
MIN_RMS = float(os.environ.get("DANYA_TTS_MIN_RMS", "0.0005"))
REQUEST_TIMEOUT = float(os.environ.get("DANYA_TTS_REQUEST_TIMEOUT", "45"))
DEFAULT_LANGUAGE = os.environ.get("DANYA_TTS_LANGUAGE", "ja").strip() or "ja"
OUTPUT_GAIN = max(0.1, float(os.environ.get("DANYA_TTS_GAIN", "8.0")))


def _run_player(cmd: list[str]) -> tuple[bool, str]:
    try:
        result = subprocess.run(cmd, check=False)
    except Exception as exc:
        return False, str(exc)
    if result.returncode != 0:
        return False, f"player exit code {result.returncode}"
    return True, "OK"


def play_audio(filepath: Path, audio_device: str | None = None) -> tuple[bool, str]:
    if not filepath.exists():
        raise FileNotFoundError(f"audio file not found: {filepath}")

    # Respect explicit argument first, then environment variable.
    device = (audio_device or os.environ.get("DANYA_AUDIO_DEVICE", "")).strip() or None

    # PulseAudio / PipeWire path (Ubuntu default).
    paplay = which("paplay")
    if paplay:
        cmd = [paplay]
        if device:
            cmd.extend(["--device", device])
        cmd.append(str(filepath))
        return _run_player(cmd)

    # ALSA fallback.
    aplay = which("aplay")
    if aplay:
        cmd = [aplay]
        if device:
            cmd.extend(["-D", device])
        cmd.append(str(filepath))
        return _run_player(cmd)

    # Last fallback.
    ffplay = which("ffplay")
    if ffplay:
        return _run_player([ffplay, "-autoexit", "-nodisp", "-loglevel", "quiet", str(filepath)])

    return False, "No audio player found (paplay/aplay/ffplay)"

def server_health(server_url: str) -> bool:
    try:
        r = requests.get(f"{server_url.rstrip('/')}/health", timeout=5)
        payload = r.json()
        return r.status_code == 200 and (payload.get("ok") is True or payload.get("status") == "ok")
    except Exception:
        return False

def tts_request(
    server_url: str,
    text: str,
    ref_id: str | None = None,
    language: str | None = None,
) -> bytes:
    data = {
        "language": (language or DEFAULT_LANGUAGE).strip() or DEFAULT_LANGUAGE,
        "text": text,
    }
    if ref_id:
        data["ref_id"] = ref_id
    r = requests.post(f"{server_url.rstrip('/')}/tts", json=data, timeout=REQUEST_TIMEOUT)
    if r.status_code != 200:
        raise RuntimeError(f"server error {r.status_code}: {r.text}")
    return r.content

def is_valid_audio(content: bytes) -> tuple[bool, str]:
    """音声が有効かどうかを判定し、無効なら理由を返す"""
    try:
        data, sr = sf.read(io.BytesIO(content), dtype="float32")
        if data.size == 0:
            return False, "Empty data"
        if data.ndim > 1:
            data = data.mean(axis=1)

        duration = len(data) / float(sr)
        peak = float(np.max(np.abs(data)))
        rms = float(np.sqrt(np.mean(np.square(data))))

        # デバッグ用に出力
        # print(f"DEBUG: Dur={duration:.2f}s, Peak={peak:.4f}, RMS={rms:.4f}")

        if duration < MIN_DURATION_SEC:
            return False, f"Too short ({duration:.2f}s)"
        if peak < MIN_PEAK:
            return False, f"Too quiet (Peak={peak:.4f})"
        if rms < MIN_RMS:
            return False, f"Low energy (RMS={rms:.4f})"
            
        return True, "OK"
    except Exception as e:
        return False, f"Parse error: {e}"

def save_content(content: bytes, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if abs(OUTPUT_GAIN - 1.0) < 1e-6:
        out_path.write_bytes(content)
        return
    try:
        data, sr = sf.read(io.BytesIO(content), dtype="float32")
        boosted = np.clip(data * OUTPUT_GAIN, -0.98, 0.98)
        sf.write(out_path, boosted, sr, subtype="PCM_16")
    except Exception:
        out_path.write_bytes(content)


def synthesize_audio(
    server_url: str,
    text: str,
    out_path: Path,
    ref_id: str | None = None,
    language: str | None = None,
    retry_max: int = RETRY_MAX,
    retry_wait_sec: float = RETRY_WAIT_SEC,
) -> tuple[bool, str]:
    """Synthesize speech and save as a wav file without playback."""
    normalized_text = (text or "").strip()
    if not normalized_text:
        return False, "Text is empty"

    for attempt in range(1, retry_max + 1):
        try:
            content = tts_request(
                server_url=server_url,
                text=normalized_text,
                ref_id=ref_id,
                language=language,
            )
            valid, reason = is_valid_audio(content)
            if not valid:
                if attempt < retry_max:
                    time.sleep(retry_wait_sec)
                else:
                    return False, reason
                continue

            save_content(content, out_path)
            return True, "OK"
        except Exception as exc:
            if attempt < retry_max:
                time.sleep(retry_wait_sec)
            else:
                return False, str(exc)

    return False, "Failed to generate valid audio"


def synthesize_and_play(
    server_url: str,
    text: str,
    out_path: Path,
    ref_id: str | None = None,
    language: str | None = None,
    retry_max: int = RETRY_MAX,
    retry_wait_sec: float = RETRY_WAIT_SEC,
    audio_device: str | None = None,
) -> tuple[bool, str]:
    """Synthesize speech from text and play it.

    Returns:
        (True, "OK") on success, or (False, reason) on failure.
    """
    ok, reason = synthesize_audio(
        server_url=server_url,
        text=text,
        out_path=out_path,
        ref_id=ref_id,
        language=language,
        retry_max=retry_max,
        retry_wait_sec=retry_wait_sec,
    )
    if not ok:
        return False, reason
    return play_audio(out_path, audio_device=audio_device)

def main():
    parser = argparse.ArgumentParser(description="GPT-SoVITS CLI client")
    parser.add_argument("--server", required=True, help="Server URL")
    parser.add_argument("--out", default="runtime/output.wav", help="Output wav path")
    parser.add_argument("--ref", default="", help="Reference voice id, e.g. happy_high or sad_mid")
    parser.add_argument("--language", default=DEFAULT_LANGUAGE, help="TTS language, e.g. ja or ru")
    parser.add_argument("--audio-device", default="", help="Output device/sink name")
    args = parser.parse_args()

    out_path = Path(args.out)

    print(f"GPT-SoVITS CLI client | Target: {args.server}")
    print("Enter text. Type 'exit' to quit.\n")

    while True:
        try:
            text = input("> ").strip()
            if not text: continue
            if text.lower() in {"exit", "quit"}: break

            success = False
            for attempt in range(1, RETRY_MAX + 1):
                print(f"Sending... (attempt {attempt}/{RETRY_MAX})", end="\r")
                
                try:
                    content = tts_request(
                        server_url=args.server,
                        text=text,
                        ref_id=args.ref or None,
                        language=args.language,
                    )
                    valid, reason = is_valid_audio(content)
                    
                    if valid:
                        save_content(content, out_path)
                        print(f"Success! Saved to {out_path}          ") # 空白は上書き消去用
                        ok_play, reason_play = play_audio(out_path, audio_device=args.audio_device or None)
                        if ok_play:
                            success = True
                            break
                        print(f"Playback failed: {reason_play}")
                    else:
                        print(f"Attempt {attempt} failed: {reason}      ")
                        time.sleep(RETRY_WAIT_SEC)
                        
                except Exception as e:
                    print(f"Attempt {attempt} error: {e}")
                    time.sleep(RETRY_WAIT_SEC)

            if not success:
                print("❌ Failed to generate valid audio after maximum retries.")

        except KeyboardInterrupt:
            print("\nbye")
            break

if __name__ == "__main__":
    main()
