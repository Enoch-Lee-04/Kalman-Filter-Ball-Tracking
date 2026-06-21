"""Draws the tracking HUD: match box, +horizon prediction box, residual bar,
latency table, and status banner. Styled to echo the reference clips.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

WHITE = (245, 245, 245)
BLUE = (255, 150, 40)      # BGR -> bright blue
GREEN = (120, 230, 130)
RED = (90, 90, 235)
PINK = (170, 120, 245)
GREY = (150, 150, 150)
DARK = (40, 40, 40)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _box(img, bbox, color, thickness=2):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)


def _label(img, text, org, color, scale=0.45, thickness=1):
    cv2.putText(img, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def draw(
    frame: np.ndarray,
    match_bbox: Optional[Tuple] = None,
    predicted_center: Optional[np.ndarray] = None,
    predicted_bbox: Optional[Tuple] = None,
    match_center: Optional[np.ndarray] = None,
    residual_px: float = float("nan"),
    residual_threshold: float = 9.0,
    horizon_ms: int = 180,
    latency: Optional[dict] = None,
    confidence: float = 0.0,
    status: str = "TRACKING",
    striking: bool = False,
) -> np.ndarray:
    h, w = frame.shape[:2]
    img = frame

    # current detection ("match")
    if match_bbox is not None:
        _box(img, match_bbox, WHITE, 2)
        x1, y1, _, _ = [int(v) for v in match_bbox]
        _label(img, "match", (x1, y1 - 6), WHITE)

    # forward prediction box (+horizon)
    if predicted_center is not None:
        px, py = float(predicted_center[0]), float(predicted_center[1])
        color = GREEN if striking else BLUE
        if predicted_bbox is not None:
            _box(img, predicted_bbox, color, 2)
            lx, ly = int(predicted_bbox[0]), int(predicted_bbox[1])
        else:
            s = 50.0
            _box(img, (px - s / 2, py - s / 2, px + s / 2, py + s / 2), color, 2)
            lx, ly = int(px - s / 2), int(py - s / 2)
        _label(img, f"+{horizon_ms} ms", (lx, ly - 6), color)

        # overshoot vector from current detection to prediction
        if match_center is not None:
            cv2.line(img, (int(match_center[0]), int(match_center[1])),
                     (int(px), int(py)), PINK, 1, cv2.LINE_AA)

    _draw_residual_bar(img, residual_px, residual_threshold)
    if latency:
        _draw_latency(img, latency)
    _draw_banner(img, status, confidence, horizon_ms)
    return img


def _draw_residual_bar(img, residual_px, threshold):
    h, w = img.shape[:2]
    bx, by, bw, bh = 20, h - 34, w - 200, 10
    cv2.putText(img, "RESIDUAL", (bx, by - 6), FONT, 0.4, GREY, 1, cv2.LINE_AA)
    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), DARK, -1)
    if not np.isnan(residual_px):
        # full bar at 60px error, empty near 0
        frac = max(0.02, min(1.0, residual_px / 60.0))
        good = residual_px <= threshold
        color = GREEN if good else BLUE
        cv2.rectangle(img, (bx, by), (bx + int(bw * frac), by + bh), color, -1)
        cv2.putText(img, f"{residual_px:5.1f} px", (bx + bw + 12, by + bh),
                    FONT, 0.5, color, 1, cv2.LINE_AA)


def _draw_latency(img, latency):
    x, y = 20, 24
    total = latency.get("sense", 0) + latency.get("think", 0) + latency.get("move", 0)
    rows = [("sense", latency.get("sense", 0)),
            ("think", latency.get("think", 0)),
            ("move", latency.get("move", 0))]
    cv2.putText(img, "ROUND TRIP", (x, y), FONT, 0.4, GREY, 1, cv2.LINE_AA)
    for i, (name, ms) in enumerate(rows):
        yy = y + 18 * (i + 1)
        cv2.putText(img, f"{name:<6}{ms:>4d} ms", (x, yy), FONT, 0.45, WHITE, 1, cv2.LINE_AA)
    cv2.putText(img, f"total {total:>4d} ms", (x, y + 18 * 4), FONT, 0.45, BLUE, 1, cv2.LINE_AA)


def _draw_banner(img, status, confidence, horizon_ms):
    h, w = img.shape[:2]
    text = f"{status}  |  +{horizon_ms} ms ahead"
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.55, 1)
    x = (w - tw) // 2
    cv2.putText(img, text, (x, 28), FONT, 0.55, BLUE, 1, cv2.LINE_AA)

    # confidence meter top-right
    bx, by, bw, bh = w - 130, 40, 110, 8
    cv2.putText(img, "confidence", (bx, by - 5), FONT, 0.38, GREY, 1, cv2.LINE_AA)
    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), DARK, -1)
    cv2.rectangle(img, (bx, by), (bx + int(bw * confidence), by + bh), GREEN, -1)
