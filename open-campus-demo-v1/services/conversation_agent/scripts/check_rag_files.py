"""RAG対象ファイル確認。"""

from __future__ import annotations

import sys
from pathlib import Path

base = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(base))

import config
from rag_manager import RagManager

r = RagManager(config.static_rag_dir, config.dynamic_rag_dir)
r.ensure_placeholders()
for p in r.list_search_targets():
    print(p)
