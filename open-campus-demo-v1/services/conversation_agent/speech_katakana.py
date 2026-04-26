"""発話本文中のラテン字略語をカタカナ読みへ寄せる（後処理用）。"""

from __future__ import annotations

# 長いパターンを先に置換（部分一致の誤変換を減らす）
_LATIN_TO_KATAKANA: tuple[tuple[str, str], ...] = (
    ("RoboCup@Home", "ロボカップアットホーム"),
    ("RoboCup", "ロボカップ"),
    ("生成AI", "生成エーアイ"),
    ("画像AI", "画像エーアイ"),
    ("AI技術", "エーアイ技術"),
    ("AIロボット", "エーアイロボット"),
    ("JPHACKS", "ジェイピーハックス"),
    ("4D@HOME", "フォーディーアットホーム"),
    ("LLM", "エルエルエム"),
    ("IoT", "アイオーティー"),
    ("API", "エーピーアイ"),
    ("GPU", "ジーピーユー"),
    ("GPT", "ジーピーティー"),
    ("GUI", "ジーユーアイ"),
    ("XR", "エックスアール"),
    ("3D", "スリーディー"),
    ("VR", "ブイアール"),
    ("KIT", "ケーアイティー"),
    ("AI", "エーアイ"),
)


def latin_abbrev_to_katakana(text: str) -> str:
    out = text
    for old, new in _LATIN_TO_KATAKANA:
        out = out.replace(old, new)
    return out
