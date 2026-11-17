"""Length of a detected item along its main axis, and pixel → cm scale."""

from __future__ import annotations

import math

import cv2
import numpy as np


def item_axis(frame, box) -> tuple[tuple[int, int], tuple[int, int]]:
    """End points of the item's long axis inside its detection box.

    Segments the item from its background (Otsu threshold, polarity picked
    from the box border) and fits a rotated rectangle to the largest blob.
    Falls back to the box's diagonal / long side if segmentation fails.
    """
    x1, y1, x2, y2 = (int(v) for v in box)
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w - 1), min(y2, h - 1)
    bw, bh = x2 - x1, y2 - y1
    if bw < 4 or bh < 4:
        return (x1, y1), (x2, y2)

    crop = cv2.GaussianBlur(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, mask = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    long_side, diag = max(bw, bh), math.hypot(bw, bh)

    # Try both polarities (dark item on light, light item on dark) and keep the
    # biggest blob that does not run along the box edge — that one is background.
    best = None
    for m in (mask, 255 - mask):
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        c = max(contours, key=cv2.contourArea)
        blob = np.zeros_like(m)
        cv2.drawContours(blob, [c], -1, 255, -1)
        rim = (np.concatenate([blob[0], blob[-1], blob[:, 0], blob[:, -1]]) > 0).mean()
        if rim < 0.2 and cv2.contourArea(c) > 0.05 * bw * bh:
            if best is None or cv2.contourArea(c) > cv2.contourArea(best):
                best = c
    if best is not None:
        (cx, cy), (rw, rh), ang = cv2.minAreaRect(best)
        length, width = max(rw, rh), min(rw, rh)
        # Trust the blob only if it is elongated and plausibly sized; a hand
        # gripping the item merges with it into a fat blob.
        if width < 0.5 * length and 0.6 * long_side <= length <= 1.1 * diag:
            theta = math.radians(ang if rw >= rh else ang + 90)
            # The blob gives the direction; parts of the item (a pen cap, a
            # shiny tip) often split off, so stretch the line through the box
            # centre to the box edges — YOLO boxes are tight around the item.
            return _chord((x1, y1, x2, y2), math.cos(theta), math.sin(theta))

    # Fallback: elongated box → long side; squarish box → diagonal.
    if bw > 2.5 * bh:
        return (x1, (y1 + y2) // 2), (x2, (y1 + y2) // 2)
    if bh > 2.5 * bw:
        return ((x1 + x2) // 2, y1), ((x1 + x2) // 2, y2)
    # Squarish box: pick the diagonal that runs along more edge pixels.
    edges = cv2.Canny(crop, 50, 150)
    d1 = np.zeros_like(edges)
    d2 = np.zeros_like(edges)
    t = max(3, min(bw, bh) // 12)
    cv2.line(d1, (0, 0), (bw - 1, bh - 1), 255, t)
    cv2.line(d2, (bw - 1, 0), (0, bh - 1), 255, t)
    if cv2.countNonZero(edges & d2) > cv2.countNonZero(edges & d1):
        return (x2, y1), (x1, y2)
    return (x1, y1), (x2, y2)


def _chord(box, ux, uy):
    """Segment through the box centre along (ux, uy), clipped to the box."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    ts = []
    if abs(ux) > 1e-6:
        ts.append((x2 - x1) / 2 / abs(ux))
    if abs(uy) > 1e-6:
        ts.append((y2 - y1) / 2 / abs(uy))
    t = min(ts)
    return ((int(cx - ux * t), int(cy - uy * t)), (int(cx + ux * t), int(cy + uy * t)))


def fmt_len(px: float, cm_per_px: float | None) -> str:
    if cm_per_px is None:
        return f"{px:.0f}px"
    return f"{px * cm_per_px:.1f}cm"
