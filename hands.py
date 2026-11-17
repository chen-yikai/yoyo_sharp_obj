"""Hand + finger tracking with MediaPipe Hand Landmarker (21 points per hand)."""

from __future__ import annotations

import math
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")
MODEL_PATH = Path(__file__).parent / "models" / "hand_landmarker.task"

# Landmark chains, base → tip. Thumb starts at the CMC joint, others at the knuckle (MCP).
FINGERS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
TIPS = (4, 8, 12, 16, 20)
PALM = (0, 9)  # wrist → middle knuckle, the most stable span for scale
SKELETON = [(0, 1), (0, 5), (0, 17), (5, 9), (9, 13), (13, 17)] + [
    (c[i], c[i + 1]) for c in FINGERS.values() for i in range(len(c) - 1)
]


@dataclass
class Hand:
    side: str                                  # "Left" / "Right" (as seen in the image)
    pts: list[tuple[float, float]]             # 21 landmarks in pixels
    world: list[tuple[float, float, float]]    # 21 landmarks in metres (MediaPipe estimate)
    box: tuple[int, int, int, int] = (0, 0, 0, 0)
    finger_px: dict[str, float] = field(default_factory=dict)
    finger_cm_est: dict[str, float] = field(default_factory=dict)  # from world landmarks

    def __post_init__(self):
        xs = [p[0] for p in self.pts]
        ys = [p[1] for p in self.pts]
        self.box = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
        for name, chain in FINGERS.items():
            self.finger_px[name] = _chain_len(self.pts, chain)
            self.finger_cm_est[name] = _chain_len(self.world, chain) * 100

    def palm_scale_cm_per_px(self) -> float | None:
        """cm per pixel at the depth of this hand, from MediaPipe's metric estimate."""
        px = math.dist(self.pts[PALM[0]], self.pts[PALM[1]])
        cm = math.dist(self.world[PALM[0]], self.world[PALM[1]]) * 100
        return cm / px if px > 20 else None


def _chain_len(pts, chain) -> float:
    return sum(math.dist(pts[a], pts[b]) for a, b in zip(chain, chain[1:]))


def ensure_model() -> Path:
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(exist_ok=True)
        tmp = MODEL_PATH.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.rename(MODEL_PATH)
    return MODEL_PATH


class HandTracker:
    def __init__(self, max_hands: int = 2):
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model())),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        self._lm = vision.HandLandmarker.create_from_options(opts)
        self._last_ts = 0

    def detect(self, frame_bgr) -> list[Hand]:
        h, w = frame_bgr.shape[:2]
        ts = max(int(time.monotonic() * 1000), self._last_ts + 1)  # must strictly increase
        self._last_ts = ts
        img = mp.Image(image_format=mp.ImageFormat.SRGB,
                       data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        res = self._lm.detect_for_video(img, ts)
        hands = []
        for lms, wlms, side in zip(res.hand_landmarks, res.hand_world_landmarks, res.handedness):
            hands.append(Hand(
                side=side[0].category_name,
                pts=[(p.x * w, p.y * h) for p in lms],
                world=[(p.x, p.y, p.z) for p in wlms],
            ))
        return hands

    def close(self):
        self._lm.close()
