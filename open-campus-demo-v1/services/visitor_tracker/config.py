"""設定値をまとめるモジュール。

ここを編集するだけで、しきい値やタイムアウトを調整できます。
環境変数や .env がある場合は、そちらを優先します。
"""

import os

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_int_list(name: str, default: list[int]) -> list[int]:
    raw = os.getenv(name)
    if raw is None:
        return default
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(int(part))
        except ValueError:
            continue
    return values or default

# ====== モデル / カメラ設定 ======
YOLO_MODEL_NAME = os.getenv("DANYA_TRACKER_YOLO_MODEL", "yolov8n.pt")
# MacでiPhone(Continuity Camera)が0番になりやすいため、
# 内蔵カメラを優先したい場合は 1 を先に試す。
CAMERA_INDEX = _env_int("DANYA_TRACKER_CAMERA_INDEX", _env_int("DANYA_CAMERA_INDEX", 1))
# 起動時に順に試す候補（先頭が優先）
CAMERA_CANDIDATES = _env_int_list("DANYA_TRACKER_CAMERA_CANDIDATES", [1, 0, 2, 3])

# ====== 復帰判定の基本設定 ======
# 人物を見失ってから何秒まで「同じ人物として復帰」させるか
LOST_TIMEOUT_SEC = 30.0

# 同一人物スコアがこの値以上なら、同じvisitorとして復帰させる
SAME_PERSON_THRESHOLD = 0.75

# 一瞬画面から外れた程度なら「戻ってきた」ではなく継続滞在扱いにする
CONTINUITY_GRACE_SEC = 8.0

# 同じ人物に複数bboxが出た時の重複削除しきい値
DUPLICATE_BBOX_IOU_THRESHOLD = 0.55

# ====== スコア重み ======
# score = W_CLOTHING * clothing_similarity + W_POSITION * position_similarity + W_TIME * time_similarity
W_CLOTHING = 0.60
W_POSITION = 0.25
W_TIME = 0.15

# ====== 特徴量抽出設定 ======
# HSVヒストグラムのビン数（H, S, V）
HIST_BINS = (30, 32, 32)

# 上半身領域の割合
# バウンディングボックス上側何%を「服装特徴」として使うか
UPPER_BODY_RATIO = 0.60

# 位置類似度で使うスケール係数（値が大きいほど位置変化に寛容）
POSITION_SIGMA_RATIO = 0.25

# ====== 表示 / 出力設定 ======
WINDOW_NAME = "Danya Visitor Tracker"
PRINT_JSON_EVERY_FRAME = _env_bool("DANYA_TRACKER_PRINT_JSON_EVERY_FRAME", True)

# ====== 長時間滞在 / VLM設定 ======
# 何秒以上滞在したら「長時間滞在」とみなすか
LONG_STAY_THRESHOLD_SEC = 10.0

# 長時間滞在者のbbox色（BGR）
LONG_STAY_BBOX_COLOR = (255, 0, 255)

# LangChain + OpenAI VLM設定
OPENAI_MODEL_NAME = os.getenv("DANYA_TRACKER_OPENAI_MODEL", os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"))
ENABLE_VLM_ANALYSIS = _env_bool("DANYA_TRACKER_ENABLE_VLM", _env_bool("DANYA_ENABLE_VLM_ANALYSIS", True))
