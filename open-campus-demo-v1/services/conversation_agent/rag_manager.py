"""RAG資料ロードと簡易検索。本文が空でも安全。"""

from __future__ import annotations

from pathlib import Path


class RagManager:
    def __init__(self, static_dir: Path, dynamic_dir: Path) -> None:
        self.static_dir = static_dir
        self.dynamic_dir = dynamic_dir
        self.dynamic_allowed = [
            dynamic_dir / "current_strategy.md",
            dynamic_dir / "strategy_memory.md",
            dynamic_dir / "successful_patterns.md",
            dynamic_dir / "failed_patterns.md",
            dynamic_dir / "visitor_reaction_notes.md",
        ]

    def ensure_placeholders(self) -> None:
        self.static_dir.mkdir(parents=True, exist_ok=True)
        self.dynamic_dir.mkdir(parents=True, exist_ok=True)
        (self.dynamic_dir / "event_logs").mkdir(parents=True, exist_ok=True)
        for p in self.dynamic_allowed:
            if not p.exists():
                p.write_text("# TODO: ここにRAG本文を書く\n", encoding="utf-8")

    def _read(self, path: Path) -> str:
        if not path.exists():
            return ""
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            return ""

    def list_search_targets(self) -> list[Path]:
        targets: list[Path] = []
        if self.static_dir.exists():
            targets.extend(sorted(self.static_dir.rglob("*.md")))
        targets.extend([p for p in self.dynamic_allowed if p.exists()])
        return targets

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        targets = self.list_search_targets()
        if not targets:
            return []
        words = [w for w in query.lower().split() if w]
        if not words:
            return []

        results = []
        for p in targets:
            text = self._read(p)
            if not text.strip():
                continue
            score = sum(text.lower().count(w) for w in words)
            if score <= 0:
                continue
            heading = ""
            for line in text.splitlines():
                if line.startswith("#"):
                    heading = line.strip()
                    break
            excerpt = text[:200].replace("\n", " ")
            results.append({"file": str(p), "heading": heading, "excerpt": excerpt, "score": score})
        results.sort(key=lambda x: x["score"], reverse=True)
        # 各チャンクを短くして返す（LLM入力過多防止）
        trimmed = []
        for r in results[:top_k]:
            rr = dict(r)
            rr["excerpt"] = rr["excerpt"][:300]
            trimmed.append(rr)
        return trimmed

    def build_query(self, mode: str, audience: str, returning: bool = False) -> str:
        """modeと対象層から検索クエリを生成する。"""
        mode_map = {
            "hook": "hook_patterns persona appeal_points",
            "intro": "intro_patterns overview high_school_student",
            "deepen": "deepen_patterns expert high_school_student appeal_points",
            "returning": "returning_visitor_patterns closing_messages",
            "quiz": "quiz_patterns child high_school_student",
            "crowd": "crowd_patterns overview",
            "closing": "recovery_patterns closing_messages",
            "idle": "hook_patterns general",
        }
        if returning:
            return "returning_visitor_patterns current_strategy " + mode_map.get(mode, "")
        return f"{mode_map.get(mode, '')} {audience}".strip()

