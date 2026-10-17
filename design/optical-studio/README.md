# DANYA 頭部・外装・光学設計

ローカルの旧作業フォルダを整理するため、2026-10-17に最新版と必要なデータを集約しました。全履歴ではなく、現在のモデル・実行に必要なデータを残す方針です。

## ダウンロード

[最新版保存セット（GitHub Release）](https://github.com/KonChelMa47/Project-DANYA/releases/tag/danya-design-20261017)

`DANYA-current-20261017.tar.gz` と `CURRENT-SHA256SUMS` を同じフォルダに保存して実行します。

```sh
shasum -a 256 -c CURRENT-SHA256SUMS
mkdir restored
 tar -xzf DANYA-current-20261017.tar.gz -C restored
```

## 保存したもの

- `models/`：unified-head-v5の全印刷部品セット。頭部4部品は旧作業フォルダの最新版STLとSHA-256が一致。ZIP内に3MF、検証結果、組立資料も含みます。
- `guides/`：試作機・鏡の組立説明。頭部形状はHEAD-V5.mdを優先してください。
- `simulator/`：光学シミュレーターのメイン画面、対応するシェル形状、投影機形状、既存JavaScript依存パッケージ。`python3 server.py --no-avatar` で起動できます。既存のProject-DANYAと連携する場合は `--project /path/to/Project-DANYA` を指定します。
- `projector/`：ブラウザー投影アプリ。`python3 server.py` で起動できます。
- Git内の `source/`：旧作業フォルダのソース・生成スクリプト・調査資料。1MiB未満の対象テキストを保存した参照用コピーです。生成されたwork/assetsのJSONは省略しています。過去のスクリプトが必要とする旧版のバイナリ中間データは含みません。

## 確認と制約

- `current-manifest.json`：今回保存したセットの各ファイルのSHA-256とサイズ。
- `current-archive.json`：配布アーカイブのSHA-256とサイズ。
- `original-workspace-inventory.json`：削除前の全ファイル・リンクの目録。これは全ファイルの保存を意味しません。
- シミュレーターの既存テスト22件と投影アプリの既存テスト1件は通過。
- 古い試作・計算キャッシュ・重複モデルは保存対象から除外。シミュレーターの過去の個別検討ページ用アセットは含みません。メイン画面の光学シミュレーターを対象としています。
- 実機での光学性能や製造性の検証は未完了です。元の資料に記載された制約を引き継ぎます。
- 既存のアバター本体と未コミットの実機調整変更は変更していません。
