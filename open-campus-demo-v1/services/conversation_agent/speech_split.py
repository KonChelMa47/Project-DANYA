"""発話セグメントを、全文のほぼ中央に最も近い『。』で二分割する。"""

from __future__ import annotations

from schemas import SpeechSegment


def split_segments_at_nearest_period(
    segments: list[SpeechSegment],
    *,
    min_total_chars: int = 20,
    min_part_chars: int = 5,
) -> tuple[list[SpeechSegment], list[SpeechSegment]] | None:
    """
    連結した本文中の『。』（U+3002）のうち、文字位置が全文長の中央に最も近いもので分割。
    分割できない場合は None。
    """
    if not segments:
        return None
    full = "".join(seg.text for seg in segments)
    n = len(full)
    if n < min_total_chars:
        return None
    period_positions = [i for i, ch in enumerate(full) if ch == "。"]
    if not period_positions:
        return None
    mid = (n - 1) / 2.0

    def sort_key(i: int) -> tuple[float, float]:
        dist_center = abs(i - mid)
        left_len = i + 1
        right_len = n - left_len
        balance = abs(left_len - right_len)
        return (dist_center, balance)

    split_end = min(period_positions, key=sort_key) + 1  # 左片に『。』を含める
    if split_end < min_part_chars or n - split_end < min_part_chars:
        return None

    left: list[SpeechSegment] = []
    right: list[SpeechSegment] = []
    pos = 0
    for seg in segments:
        t = seg.text
        start = pos
        end = pos + len(t)
        if end <= split_end:
            if t.strip():
                left.append(SpeechSegment(emotion=seg.emotion, text=t))
        elif start >= split_end:
            if t.strip():
                right.append(SpeechSegment(emotion=seg.emotion, text=t))
        else:
            lo = t[: split_end - start]
            ro = t[split_end - start :]
            if lo.strip():
                left.append(SpeechSegment(emotion=seg.emotion, text=lo))
            if ro.strip():
                right.append(SpeechSegment(emotion=seg.emotion, text=ro))
        pos = end

    if not left or not right:
        return None
    return (left, right)
