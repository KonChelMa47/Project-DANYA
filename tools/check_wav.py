import numpy as np
import librosa
from pathlib import Path

def analyze_audio(file_path):
    try:
        # 音声ファイルの読み込み（y: 波形データ, sr: サンプリングレート）
        y, sr = librosa.load(file_path, sr=None)
        
        duration = librosa.get_duration(y=y, sr=sr)
        max_amplitude = np.max(np.abs(y))
        mean_amplitude = np.mean(np.abs(y))
        
        print(f"--- Analysis for: {file_path} ---")
        print(f"Sampling Rate: {sr} Hz")
        print(f"Duration: {duration:.3f} seconds")
        print(f"Max Amplitude: {max_amplitude:.6f}")
        print(f"Mean Amplitude: {mean_amplitude:.6f}")

        # 無音判定のロジック
        if duration == 0:
            print("❌ Error: ファイルが空（0秒）です。")
        elif max_amplitude < 1e-5:
            print("❌ Status: 実質的に無音です（データはありますが振幅がほぼ0です）。")
        else:
            print("✅ Status: 音声データが含まれています。")
            
        # NaN（非数）が含まれていないかチェック
        if np.isnan(y).any():
            print("⚠️ Warning: データ内にNaN（数値エラー）が含まれています。モデルの計算が壊れた可能性があります。")

    except Exception as e:
        print(f"❌ Error while loading file: {e}")

# 調べたいファイルパスを指定して実行
PROJECT_ROOT = Path(__file__).resolve().parent.parent
analyze_audio(str(PROJECT_ROOT / "runtime" / "output.wav"))