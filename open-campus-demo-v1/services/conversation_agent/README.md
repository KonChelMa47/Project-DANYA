# conversation_agent

Open Campus Demo v1の発話生成サービスです。`visitor_tracker` の日次JSONログを読み、状況判断、話題選択、RAG検索、感情タグ付き発話生成、発話HTTP配信を行います。

## Run

```bash
cd open-campus-demo-v1/services/conversation_agent
python3 main.py
```

デバッグモード:

```bash
python3 main.py --debug-mode
```

## Related Paths

- `../visitor_tracker/logs/`: trackerが出力するイベントログ
- `static_rag/`: 展示説明、観客タイプ、戦略、ダーニャの人格設定
- `dynamic_rag/`: 実行中に更新される戦略メモ
- `prompts/`: LLMプロンプト
- `self_improvement/`: 発話評価と戦略メモ更新

## Speech API

既定では `http://127.0.0.1:8765` で配信します。

```bash
curl http://127.0.0.1:8765/latest.txt
curl http://127.0.0.1:8765/latest.json
curl -N http://127.0.0.1:8765/stream
```
