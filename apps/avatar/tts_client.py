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
MIN_DURATION_SEC = 0.5  # 0.5秒未満は短すぎると判定
MIN_PEAK = 0.05         # 0.008程度だと「ほぼ無音」なので、0.05（約-26dB）を境界に設定
MIN_RMS = 0.01          # 全体の平均音量もチェック
REQUEST_TIMEOUT = float(os.environ.get("DANYA_TTS_REQUEST_TIMEOUT", "45"))


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
        return r.status_code == 200 and r.json().get("ok") is True
    except Exception:
        return False

def tts_request(server_url: str, text: str, ref_id: str | None = None) -> bytes:
    # 多くのGPT-soVITS APIで採用されているパラメータを追加
    # 推論が不安定な場合は top_p や temp を調整できるように設計
    data = {
        "text": text,
        "text_language": "ja", # 必要に応じて変更
        "top_p": 1, 
        "temperature": 1,
        "speed": 1.0
    }
    if ref_id:
        data["ref_id"] = ref_id
    r = requests.post(f"{server_url.rstrip('/')}/tts", data=data, timeout=REQUEST_TIMEOUT)
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
    out_path.write_bytes(content)


def synthesize_audio(
    server_url: str,
    text: str,
    out_path: Path,
    ref_id: str | None = None,
    retry_max: int = RETRY_MAX,
    retry_wait_sec: float = RETRY_WAIT_SEC,
) -> tuple[bool, str]:
    """Synthesize speech and save as a wav file without playback."""
    normalized_text = (text or "").strip()
    if not normalized_text:
        return False, "Text is empty"

    for attempt in range(1, retry_max + 1):
        try:
            content = tts_request(server_url=server_url, text=normalized_text, ref_id=ref_id)
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
                    content = tts_request(server_url=args.server, text=text, ref_id=args.ref or None)
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
