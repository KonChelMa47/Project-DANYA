"""dynamic_rag書き込みガード。"""

from __future__ import annotations

from pathlib import Path


class RagWriteGuard:
    def __init__(self, dynamic_rag_dir: Path, static_rag_dir: Path) -> None:
        self.dynamic_dir = dynamic_rag_dir.resolve()
        self.static_dir = static_rag_dir.resolve()

    def allow_path(self, path: Path) -> bool:
        rp = path.resolve()
        if str(rp).startswith(str(self.static_dir)):
            return False
        return str(rp).startswith(str(self.dynamic_dir))

    def allow_content(self, text: str) -> bool:
        lowered = text.lower()
        blocked = ["顔画像", "face image", ".jpg", ".png", "/users/", "性別ベース", "gender-based"]
        return not any(k in lowered for k in blocked)

