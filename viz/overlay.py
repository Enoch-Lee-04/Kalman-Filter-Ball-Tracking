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
    ui_scale: float = 1.0,
    tracking_scale: float = 1.0,
    show_tracking: bool = True,
    view_label: str | None = None,
) -> np.ndarray:
    h, w = frame.shape[:2]
    img = frame
    sx = ui_scale
    ts = tracking_scale
    thick = max(1, int(round(2 * ts)))

    # ball tracking overlays (robot camera only; coords in display pixel space)
    if show_tracking:
        if match_bbox is not None:
            _box(img, _scale_bbox(match_bbox, ts), WHITE, thick)
            x1, y1, _, _ = [int(v) for v in _scale_bbox(match_bbox, ts)]
            _label(img, "match", (x1, y1 - int(6 * sx)), WHITE, 0.45 * sx, max(1, int(round(sx))))

        if predicted_center is not None:
            px, py = float(predicted_center[0]) * ts, float(predicted_center[1]) * ts
            color = GREEN if striking else BLUE
            if predicted_bbox is not None:
                _box(img, _scale_bbox(predicted_bbox, ts), color, thick)
                lx, ly = int(predicted_bbox[0] * ts), int(predicted_bbox[1] * ts)
            else:
                s = 50.0 * ts
                _box(img, (px - s / 2, py - s / 2, px + s / 2, py + s / 2), color, thick)
                lx, ly = int(px - s / 2), int(py - s / 2)
            _label(img, f"+{horizon_ms} ms", (lx, ly - int(6 * sx)), color, 0.45 * sx,
                   max(1, int(round(sx))))

            if match_center is not None:
                cv2.line(img, (int(match_center[0] * ts), int(match_center[1] * ts)),
                         (int(px), int(py)), PINK, max(1, int(round(ts))), cv2.LINE_AA)

    _draw_hud_panels(img, residual_px, residual_threshold, latency, status,
                     confidence, horizon_ms, sx)
    if view_label:
        scale = 0.38 * sx
        thick = max(1, int(round(sx)))
        (tw, _), _ = cv2.getTextSize(view_label, FONT, scale, thick)
        _label(img, view_label, (w - tw - int(20 * sx), h - int(8 * sx)), GREY, scale, thick)
    return img


def _draw_hud_panels(img, residual_px, residual_threshold, latency, status,
                     confidence, horizon_ms, sx):
    _draw_residual_bar(img, residual_px, residual_threshold, sx)
    if latency:
        _draw_latency(img, latency, sx)
    _draw_banner(img, status, confidence, horizon_ms, sx)


def _scale_bbox(bbox, sx: float):
    return tuple(v * sx for v in bbox)


def _draw_residual_bar(img, residual_px, threshold, sx=1.0):
    h, w = img.shape[:2]
    bx, by = int(20 * sx), h - int(34 * sx)
    bw, bh = w - int(200 * sx), max(6, int(10 * sx))
    cv2.putText(img, "RESIDUAL", (bx, by - int(6 * sx)), FONT, 0.4 * sx, GREY,
                max(1, int(round(sx))), cv2.LINE_AA)
    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), DARK, -1)
    if not np.isnan(residual_px):
        # full bar at 60px error, empty near 0
        frac = max(0.02, min(1.0, residual_px / 60.0))
        good = residual_px <= threshold
        color = GREEN if good else BLUE
        cv2.rectangle(img, (bx, by), (bx + int(bw * frac), by + bh), color, -1)
        cv2.putText(img, f"{residual_px:5.1f} px", (bx + bw + int(12 * sx), by + bh),
                    FONT, 0.5 * sx, color, max(1, int(round(sx))), cv2.LINE_AA)


def _draw_latency(img, latency, sx=1.0):
    x, y = int(20 * sx), int(24 * sx)
    total = latency.get("sense", 0) + latency.get("think", 0) + latency.get("move", 0)
    rows = [("sense", latency.get("sense", 0)),
            ("think", latency.get("think", 0)),
            ("move", latency.get("move", 0))]
    cv2.putText(img, "ROUND TRIP", (x, y), FONT, 0.4 * sx, GREY, max(1, int(round(sx))), cv2.LINE_AA)
    for i, (name, ms) in enumerate(rows):
        yy = y + int(18 * sx) * (i + 1)
        cv2.putText(img, f"{name:<6}{ms:>4d} ms", (x, yy), FONT, 0.45 * sx, WHITE,
                    max(1, int(round(sx))), cv2.LINE_AA)
    cv2.putText(img, f"total {total:>4d} ms", (x, y + int(18 * sx) * 4), FONT, 0.45 * sx, BLUE,
                max(1, int(round(sx))), cv2.LINE_AA)


def _draw_banner(img, status, confidence, horizon_ms, sx=1.0):
    h, w = img.shape[:2]
    text = f"{status}  |  +{horizon_ms} ms ahead"
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.55 * sx, max(1, int(round(sx))))
    x = (w - tw) // 2
    cv2.putText(img, text, (x, int(28 * sx)), FONT, 0.55 * sx, BLUE,
                max(1, int(round(sx))), cv2.LINE_AA)

    # confidence meter top-right
    bx, by = w - int(130 * sx), int(40 * sx)
    bw, bh = int(110 * sx), max(6, int(8 * sx))
    cv2.putText(img, "confidence", (bx, by - int(5 * sx)), FONT, 0.38 * sx, GREY,
                max(1, int(round(sx))), cv2.LINE_AA)
    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), DARK, -1)
    cv2.rectangle(img, (bx, by), (bx + int(bw * confidence), by + bh), GREEN, -1)
