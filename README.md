# DANYA (Project DANYA)

> **"Finding the essence of 'naturalness' to bridge the gap between humans and robots."**

## 🌐 Overview
DANYAは、人間とロボットの間に「信頼」と「親しみ」を築くためのHRI（Human-Robot Interaction）研究プロジェクトです。人間同士のコミュニケーションにおける「自然さ」の本質を解明し、それをデジタルおよび物理的な存在として再現することを目指しています。

本プロジェクトは2段階のフェーズで進行します：
1. **Digital Phase**: 自身のAIアバターの開発とコミュニケーションモデルの構築
2. **Physical Phase**: アバターを現実世界へ具現化する「頭部コミュニケーションロボット」の開発

## 🎯 Research Objectives
- 人間がロボットに対して抱く「信頼感」と「親近感」の要因特定
- 非言語コミュニケーション（視線、表情、間）の最適化
- 物理的実体（ロボットヘッド）がコミュニケーションの質に与える影響の調査

## 🛠 Tech Stack (Planned)
これまでのプロジェクト経験（RoboCup@Home等）を活かし、以下の技術を中心に構築します。

- **Software**: Python, ROS 2, Android (Kotlin)
- **AI/NLP**: LLM (OpenAI/Gemini API), Text-to-Speech, Speech-to-Text
- **Hardware**: 3D Printing (Flashforge Adventurer 5M), Arduino / ESP32
- **Design**: Blender / Unity (Avatar Design)

## 🚀 Roadmap
- [ ] Phase 1: AIアバターのプロトタイプ作成と対話エンジンの統合
- [ ] Phase 2: アバターの表情・視線制御の実装
- [ ] Phase 3: 3Dプリンターを用いた頭部ロボット筐体の設計・製作
- [ ] Phase 4: 実機を用いたHRI実験の実施

## 🖥 Python Projection Mapping Face App

Pythonで動くリアルタイム顔プロジェクションマッピングアプリです。`assets/models/avatar.glb` の顔だけを全画面で表示します。

### 特徴
- カメラ映像から顔を検知
- MediaPipe の表情推定を使って、`avatar.glb` の表情をリアルタイムで変化
- 顔の位置は固定で、首の動きは固定のまま
- 顔は 90 度回転表示され、顎が右・頭が左になる
- 黒背景、周囲のUIなし
- フルスクリーン表示
- 外側の四隅ハンドルで顔全体の拡大・台形補正ができる（従来ワープ）
- 内側の5x5グリッドハンドルで顔パーツを局所的にメッシュ変形できる
- E でハンドルの表示・非表示を切り替えられる
- R で外側四隅と内側グリッドの両方を初期状態に戻せる

### 起動方法
```bash
/home/daniil/Project-DANYA/.venv/bin/python main.py
```

### 必要ファイル
- `main.py`
- `assets/models/avatar.glb`

### ディレクトリ構成
```text
Project-DANYA/
├── assets/
│   └── models/
│       └── avatar.glb
├── data/
│   └── animation_data.json
├── main.py
├── conversation_with_danya.py
├── tts_client.py
├── runtime/
│   └── output.wav
├── tools/
│   └── check_wav.py
├── README.md
└── requirements.txt
```

初回起動時に MediaPipe の顔ランドマーカー用モデルを自動ダウンロードします。

## 👤 Author
**Daniil Malchenko (ダニール マルチェンコ)**
- Kanazawa Institute of Technology (KIT)
- Department of Information and Computer Engineering
- Robotics & HRI Researcher

---
© 2026
