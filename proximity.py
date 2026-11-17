"""How close is the hand to a sharp item, how fast is it closing in, and how dangerous is that?

The hand is modelled as its 20 bone segments (from MediaPipe's 21 landmarks)
and each sharp item as its long-axis segment. The gap is the shortest
segment-to-segment distance in the image, converted to cm with the current
scale. Approach speed comes from a least-squares fit over the last ~0.5 s,
and time-to-contact = gap / closing speed.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass

from hands import FINGERS, SKELETON, Hand

# Danger levels, least to most severe.
SAFE, CAUTION, DANGER, CONTACT = "SAFE", "CAUTION", "DANGER", "CONTACT"
LEVEL_RANK = {SAFE: 0, CAUTION: 1, DANGER: 2, CONTACT: 3}


@dataclass
class Thresholds:
    caution_cm: float = 10.0   # closer than this → CAUTION
    danger_cm: float = 4.0     # closer than this → DANGER
    contact_cm: float = 1.0    # closer than this → CONTACT
    ttc_danger_s: float = 0.8  # contact expected sooner than this → at least DANGER
    ttc_caution_s: float = 2.0  # … sooner than this → at least CAUTION


@dataclass
class Gap:
    item: int                     # index into detections
    hand: int                     # index into hands
    px: float
    cm: float | None
    part: str                     # e.g. "index finger", "palm"
    p_hand: tuple[float, float]   # closest point on the hand
    p_item: tuple[float, float]   # closest point on the item


@dataclass
class Assessment:
    level: str
    closest: Gap | None
    speed_cm_s: float | None      # positive = approaching
    ttc_s: float | None           # seconds until contact, if approaching
    per_item: dict[int, Gap]      # closest gap for each item


def _bone_part(a: int, b: int) -> str:
    for name, chain in FINGERS.items():
        if a in chain and b in chain:
            return "thumb" if name == "thumb" else f"{name} finger"
    return "palm"


BONES = [(a, b, _bone_part(a, b)) for a, b in SKELETON]


def _closest_on_seg(p, a, b):
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / L2))
    return (ax + t * dx, ay + t * dy)


def _intersect(a, b, c, d):
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)


def seg_seg(a, b, c, d):
    """Shortest distance between segments ab and cd → (dist, point on ab, point on cd)."""
    if _intersect(a, b, c, d):
        # Report the crossing point for both.
        x1, y1 = a
        x2, y2 = b
        x3, y3 = c
        x4, y4 = d
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        p = (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
        return 0.0, p, p
    best = None
    for p, (s, e), on_ab in ((a, (c, d), True), (b, (c, d), True),
                             (c, (a, b), False), (d, (a, b), False)):
        q = _closest_on_seg(p, s, e)
        dist = math.dist(p, q)
        if best is None or dist < best[0]:
            best = (dist, p, q) if on_ab else (dist, q, p)
    return best


def measure_gaps(dets, hands: list[Hand], cm_per_px: float | None) -> list[Gap]:
    """Closest hand ↔ item gap for every (item, hand) pair."""
    gaps = []
    for i, d in enumerate(dets):
        ia, ib = d.axis
        for j, h in enumerate(hands):
            best = None
            for a, b, part in BONES:
                dist, ph, pi = seg_seg(h.pts[a], h.pts[b], ia, ib)
                if best is None or dist < best[0]:
                    best = (dist, ph, pi, part)
            if best is None:
                continue
            dist, ph, pi, part = best
            gaps.append(Gap(i, j, dist, dist * cm_per_px if cm_per_px else None, part, ph, pi))
    return gaps


class ProximityMonitor:
    """Tracks the closest gap over time and grades the danger."""

    def __init__(self, thresholds: Thresholds | None = None, window_s: float = 0.5):
        self.th = thresholds or Thresholds()
        self.window_s = window_s
        self.hist: deque[tuple[float, float]] = deque()

    def update(self, dets, hands, cm_per_px) -> Assessment:
        now = time.monotonic()
        gaps = measure_gaps(dets, hands, cm_per_px)
        per_item: dict[int, Gap] = {}
        for g in gaps:
            if g.item not in per_item or g.px < per_item[g.item].px:
                per_item[g.item] = g
        closest = min(gaps, key=lambda g: g.px) if gaps else None

        if closest is None or closest.cm is None:
            self.hist.clear()
            return Assessment(SAFE, closest, None, None, per_item)

        self.hist.append((now, closest.cm))
        while self.hist and now - self.hist[0][0] > self.window_s:
            self.hist.popleft()
        speed = self._closing_speed()
        ttc = closest.cm / speed if speed and speed > 2.0 else None  # ignore jitter < 2 cm/s

        th, d = self.th, closest.cm
        level = SAFE
        if d <= th.contact_cm:
            level = CONTACT
        elif d <= th.danger_cm or (ttc is not None and ttc <= th.ttc_danger_s):
            level = DANGER
        elif d <= th.caution_cm or (ttc is not None and ttc <= th.ttc_caution_s):
            level = CAUTION
        return Assessment(level, closest, speed, ttc, per_item)

    def _closing_speed(self) -> float | None:
        """cm/s, positive when the gap is shrinking (least-squares slope)."""
        if len(self.hist) < 4:
            return None
        ts = [t for t, _ in self.hist]
        ds = [d for _, d in self.hist]
        tm, dm = sum(ts) / len(ts), sum(ds) / len(ds)
        var = sum((t - tm) ** 2 for t in ts)
        if var < 1e-6:
            return None
        slope = sum((t - tm) * (d - dm) for t, d in zip(ts, ds)) / var
        return -slope
