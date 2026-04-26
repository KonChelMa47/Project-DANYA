# visitor_tracker

Open Campus Demo v1の来場者観察サービスです。カメラ入力、YOLO人物検出、人物追跡、画像AI解析、日次JSONログ出力を担当します。

## Setup

```bash
cd open-campus-demo-v1/services/visitor_tracker
python3 -m pip install -r requirements.txt
cp .env.example .env
```

OpenAI画像解析を使う場合は `.env` に `OPENAI_API_KEY` を設定します。

## Run

```bash
python3 main.py
```

端末ログだけで確認:

```bash
python3 main.py --terminal-only
```

OpenAI画像解析を使わないモック:

```bash
python3 main.py --mock-vision
```

## Output

日次イベントログは `logs/YYYY-MM-DD.json` に出力されます。conversation agentはこのログを読んで発話内容を決めます。
