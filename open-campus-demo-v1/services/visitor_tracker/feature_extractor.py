"""人物特徴量（服装色・位置・時間）を扱うモジュール。"""

from __future__ import annotations

from typing import Tuple
import math

import cv2
import numpy as np

import config


def _clip_bbox_to_frame(
    bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, int, int]
) -> Tuple[int, int, int, int]:
    """バウンディングボックスをフレーム内に収める。"""
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2


def extract_upper_body_histogram(
    frame: np.ndarray, bbox: Tuple[int, int, int, int]
) -> np.ndarray:
    """人物bboxから上半身領域を切り出し、HSVヒストグラムを返す。"""
    x1, y1, x2, y2 = _clip_bbox_to_frame(bbox, frame.shape)
    height = y2 - y1
    upper_y2 = y1 + int(height * config.UPPER_BODY_RATIO)
    upper_y2 = max(y1 + 1, min(upper_y2, y2))

    roi = frame[y1:upper_y2, x1:x2]
    if roi.size == 0:
        # ROIが取れない場合はゼロベクトル
        bins = config.HIST_BINS[0] * config.HIST_BINS[1] * config.HIST_BINS[2]
        return np.zeros((bins,), dtype=np.float32)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv],
        [0, 1, 2],
        None,
        list(config.HIST_BINS),
        [0, 180, 0, 256, 0, 256],
    )
    # 正規化して比較しやすくする
    hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
    return hist


def clothing_similarity(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
    """ヒストグラムの類似度を0.0-1.0で返す（相関ベース）。"""
    if hist_a is None or hist_b is None:
        return 0.0
    if len(hist_a) == 0 or len(hist_b) == 0:
        return 0.0
    score = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)
    # 相関値は[-1, 1]なので[0, 1]へ線形変換
    score_01 = (score + 1.0) / 2.0
    return float(np.clip(score_01, 0.0, 1.0))


def bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """bbox中心座標を返す。"""
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def position_similarity(
    prev_bbox: Tuple[int, int, int, int],
    curr_bbox: Tuple[int, int, int, int],
    frame_shape: Tuple[int, int, int],
) -> float:
    """位置の近さを0.0-1.0で返す（距離が近いほど高い）。"""
    prev_cx, prev_cy = bbox_center(prev_bbox)
    curr_cx, curr_cy = bbox_center(curr_bbox)
    dist = math.hypot(curr_cx - prev_cx, curr_cy - prev_cy)

    h, w = frame_shape[:2]
    diag = math.hypot(w, h)
    sigma = max(1e-6, diag * config.POSITION_SIGMA_RATIO)

    # ガウス形で距離を類似度化
    sim = math.exp(-(dist**2) / (2.0 * (sigma**2)))
    return float(np.clip(sim, 0.0, 1.0))


def time_similarity(lost_duration_sec: float, lost_timeout_sec: float) -> float:
    """見失い時間の短さを0.0-1.0で返す。"""
    if lost_timeout_sec <= 0:
        return 0.0
    ratio = 1.0 - (lost_duration_sec / lost_timeout_sec)
    return float(np.clip(ratio, 0.0, 1.0))


def same_person_score(
    clothing_sim: float, position_sim: float, time_sim: float
) -> float:
    """重み付き同一人物スコアを返す。"""
    score = (
        config.W_CLOTHING * clothing_sim
        + config.W_POSITION * position_sim
        + config.W_TIME * time_sim
    )
    return float(np.clip(score, 0.0, 1.0))
