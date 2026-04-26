"""dynamic_ragの要約更新。"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path

from self_improvement.rag_write_guard import RagWriteGuard


class StrategyMemoryWriter:
    def __init__(self, dynamic_rag_dir: Path, guard: RagWriteGuard, batch_size: int = 10) -> None:
        self.dynamic_rag_dir = dynamic_rag_dir
        self.guard = guard
        self.batch_size = batch_size
        self.buffer: list[dict] = []
        self.last_summary = ""

    def append(self, event: dict) -> None:
        self.buffer.append(event)

    def _append_if_allowed(self, rel: str, text: str) -> None:
        p = self.dynamic_rag_dir / rel
        if not self.guard.allow_path(p):
            return
        if not self.guard.allow_content(text):
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(text)

    def maybe_flush(self) -> None:
        if len(self.buffer) < self.batch_size:
            return
        modes = Counter(e.get("strategy", {}).get("mode", "unknown") for e in self.buffer)
        evals = Counter(e.get("evaluation", {}).get("result", "neutral") for e in self.buffer)
        llm_success = sum(1 for e in self.buffer if e.get("llm_success"))
        fallback_used = sum(1 for e in self.buffer if e.get("fallback_used"))
        now = datetime.now().isoformat(timespec="seconds")
        summary = (
            f"\n## {now}\n"
            f"- mode: {dict(modes)}\n"
            f"- eval: {dict(evals)}\n"
            f"- llm_success_count: {llm_success}\n"
            f"- fallback_count: {fallback_used}\n"
            f"- count: {len(self.buffer)}\n"
        )
        if summary == self.last_summary:
            self.buffer.clear()
            return
        self._append_if_allowed("current_strategy.md", summary)
        self._append_if_allowed("strategy_memory.md", summary)
        self._append_if_allowed("visitor_reaction_notes.md", summary)
        if evals.get("success", 0) >= evals.get("failure", 0):
            self._append_if_allowed("successful_patterns.md", summary)
        else:
            self._append_if_allowed("failed_patterns.md", summary)
        self.last_summary = summary
        self.buffer.clear()

