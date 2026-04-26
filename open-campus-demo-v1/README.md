# Open Campus Demo v1

オープンキャンパスで使ったDANYAの自律型展示MCデモです。カメラで来場者を観察しながら、会話エージェントが状況判断、話題選択、感情タグ付き発話を生成します。

## Directory Map

```text
open-campus-demo-v1/
├── launchers/
│   ├── run_open_campus_demo.py      # tracker + agent 同時起動
│   └── run_camera_terminal_demo.py  # カメラ/端末確認用ランチャー
└── services/
    ├── visitor_tracker/             # カメラ入力、人物検出、追跡、画像AI解析
    └── conversation_agent/          # 状況判断、RAG、発話生成、発話HTTPサーバー
```

## Setup

```bash
cd ..
python3 -m venv .venv
source .venv/bin/activate

cd open-campus-demo-v1
python3 -m pip install -r services/conversation_agent/requirements.txt
python3 -m pip install -r services/visitor_tracker/requirements.txt
```

OpenAIを使う場合は、`services/visitor_tracker/.env` に `OPENAI_API_KEY` を設定します。

```bash
cp services/visitor_tracker/.env.example services/visitor_tracker/.env
```

`launchers/` のスクリプトは、プロジェクトルートの `.venv/bin/python` があれば自動でそれを使います。`python3 launchers/...` で起動しても、内部の tracker / agent は `.venv` 側で動きます。

## Run

3Dアバター画面まで含めて起動する場合は、プロジェクトルートから統合ランチャーを使います。

```bash
cd ..
scripts/start_open_campus_avatar.sh
```

このコマンドは、visitor tracker、conversation agent、発話ブリッジ、3Dアバター画面をまとめて起動します。

カメラあり、TTSなしで確認:

```bash
python3 launchers/run_camera_terminal_demo.py --real-vision --agent-llm
```

このコマンドはターミナル確認用なので、3Dアバター画面は出ません。

発話だけ確認するデバッグモード:

```bash
python3 launchers/run_camera_terminal_demo.py --debug-mode
```

tracker と agent の基本同時起動:

```bash
python3 launchers/run_open_campus_demo.py
```

このコマンドもOpen Campus Demo v1本体だけを起動するため、3Dアバター画面は出ません。

trackerなしでagentだけ試す:

```bash
python3 launchers/run_open_campus_demo.py --agent-only
```

## Speech API

起動すると、conversation agent側で発話配信用HTTPサーバーが立ちます。

```text
http://127.0.0.1:8765
```

最新発話:

```bash
curl http://127.0.0.1:8765/latest.txt
curl http://127.0.0.1:8765/latest_plain.txt
curl http://127.0.0.1:8765/latest.json
```

SSEで発話更新ごとに受け取る:

```bash
curl -N http://127.0.0.1:8765/stream
curl -N http://127.0.0.1:8765/stream.json
```

## Notes

`services/visitor_tracker/logs/` と `services/conversation_agent/dynamic_rag/event_logs/` は実行ログです。YOLOの重み（`*.pt`）は `.gitignore` で除外しています。
