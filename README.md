<p align="center">
  <img src="./logo.png" alt="DANYA logo" width="880">
</p>

# Project DANYA

DANYAは、人間とロボットの間にある「自然な対話」を研究するためのアバター/音声/LLM連携プロジェクトです。

## Directory Map

```text
Project-DANYA/
├── apps/
│   ├── avatar/          # アバター表示、表情、口パク、TTSクライアント
│   ├── tts/             # GPT-SoVITS HTTPサーバー
│   └── demo_servers/    # LLM出力モック、Open Campus Demo連携ブリッジ
├── open-campus-demo-v1/ # オープンキャンパスで使ったデモ本体
│   ├── launchers/       # デモ起動スクリプト
│   └── services/
│       ├── visitor_tracker/
│       └── conversation_agent/
├── assets/              # GLBモデルなど
├── data/                # 旧互換データ
├── runtime/             # 実行時生成物
├── scripts/             # 複数アプリをまとめて起動する補助スクリプト
└── tools/
```

## Main Apps

| Path | Role |
| --- | --- |
| `apps/avatar/face_motion_avatar.py` | MediaPipe Face Landmarkerから表情・顔向きデータを取り、`avatar.glb` に反映する単体アバター |
| `apps/avatar/conversation_avatar.py` | LLM出力APIを受信し、TTS音声・表情・口パク・YOLO視線制御を行う会話アバター |
| `apps/avatar/tts_client.py` | TTSサーバーへリクエストし、WAV保存・再生を行うクライアント |
| `apps/tts/gpt_sovits_server.py` | GPT-SoVITSをHTTP APIとして動かすTTSサーバー |
| `apps/demo_servers/llm_output_demo_server.py` | `<emotion_tag>本文` を20秒ごとに返すデモ用LLM出力サーバー |
| `apps/demo_servers/open_campus_speech_bridge.py` | Open Campus Demo v1の発話HTTPサーバーを会話アバター用 `/api/output` に変換するブリッジ |
| `apps/demo_servers/browser_face_demo.html` | Three.jsでGLBの顔表示と口パクを確認するブラウザデモ |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Open Campus Demo v1も使う場合:

```bash
python3 -m pip install -r open-campus-demo-v1/services/conversation_agent/requirements.txt
python3 -m pip install -r open-campus-demo-v1/services/visitor_tracker/requirements.txt
```

## Run

MediaPipe単体アバター:

```bash
.venv/bin/python apps/avatar/face_motion_avatar.py
```

会話アバター:

```bash
.venv/bin/python apps/avatar/conversation_avatar.py
```

デモ用LLMサーバー付きで会話アバターを起動:

```bash
scripts/start_conversation_demo.sh
```

Open Campus Demo v1の自律発話をアバターへ接続:

```bash
scripts/start_open_campus_avatar.sh
```

このコマンドは、visitor tracker、conversation agent、発話ブリッジ、3Dアバター画面をまとめて起動します。

Open Campus Demo v1の発話だけを確認:

```bash
cd open-campus-demo-v1
python3 launchers/run_camera_terminal_demo.py --debug-mode
```

GPT-SoVITS TTSサーバー:

```bash
.venv/bin/python apps/tts/gpt_sovits_server.py
```

## Runtime Notes

`runtime/` と `.cache/` は実行時に作られる作業ディレクトリです。ログ、TTS音声、ウィンドウ状態、MediaPipeモデルキャッシュなどが入ります。

Open Campus Demo v1の環境変数は `open-campus-demo-v1/services/visitor_tracker/.env` に置きます。`.env.example` をコピーして使ってください。

## Author

Daniil Malchenko  
Kanazawa Institute of Technology  
Department of Information and Computer Engineering
