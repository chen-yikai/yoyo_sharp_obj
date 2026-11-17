"""Sharp object identifier — live YOLO detection from the desktop camera with a Tk UI.

  * Detects sharp items (YOLO-World open vocabulary, or COCO knife/scissors/fork).
  * Tracks hands and fingers (MediaPipe) and warns when a sharp item is near a hand.
  * Measures the length of every sharp item and every finger.

The camera stays live from Start until Stop: model reloads, class edits,
resolution changes and camera hiccups never close the feed.

Run:  uv run app.py                    (desktop camera)
      uv run app.py --source clip.mp4  (video file or image, for testing)
"""

from __future__ import annotations

import argparse
import math
import subprocess
import queue
import statistics
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from tkinter import simpledialog, ttk

import cv2
from PIL import Image, ImageTk

from hands import SKELETON, TIPS, FINGERS, Hand, HandTracker
from measure import fmt_len, item_axis
from proximity import (CAUTION, CONTACT, DANGER, LEVEL_RANK, SAFE, Assessment,
                       ProximityMonitor, Thresholds)

# Anything with an edge or point that can cut or puncture a hand.
DEFAULT_SHARP_CLASSES = [
    # blades
    "knife", "kitchen knife", "utility knife", "box cutter", "scalpel",
    "razor blade", "razor", "scissors", "saw", "saw blade", "axe", "chisel",
    # writing / office points
    "pen", "pen nib", "fountain pen", "pencil", "mechanical pencil",
    "drawing compass", "thumbtack", "push pin", "staple", "staple remover",
    # needles and spikes
    "needle", "sewing needle", "syringe", "safety pin", "nail", "screw",
    "screwdriver", "awl", "ice pick", "skewer", "toothpick", "fork", "fish hook",
    # shards and rough edges
    "broken glass", "glass shard", "metal shard", "can lid", "cheese grater",
    "wire cutter", "barbed wire", "thorn",
]
COCO_SHARP_CLASSES = {"knife", "scissors", "fork"}

# Harmless look-alikes. They are prompted so the model has a better label to
# give a spoon or a chopstick than "knife", then filtered out of the results.
NOT_SHARP_CLASSES = [
    "spoon", "chopsticks", "person", "face", "hand", "cup", "cell phone",
    "remote control", "marker", "comb", "ruler", "eraser", "paper",
]

MODELS = {
    "world-l": "yolov8l-worldv2.pt",  # open vocabulary, accurate (~33 FPS on M-series)
    "world-s": "yolov8s-worldv2.pt",  # open vocabulary, fast
    "coco": "yolo11n.pt",             # knife / scissors / fork only
}

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"
HAND_COLOR = (255, 160, 0)    # BGR blue
CAL_COLOR = (0, 220, 0)       # BGR green
LEVEL_BGR = {SAFE: (60, 160, 30), CAUTION: (0, 165, 255), DANGER: (0, 0, 230),
             CONTACT: (90, 0, 160)}
LEVEL_UI = {  # banner background colours
    SAFE: "#1e7f3c", CAUTION: "#e08a00", DANGER: "#c62828", CONTACT: "#7b1fa2",
}


def pick_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


@dataclass
class Detection:
    label: str
    conf: float
    box: tuple[int, int, int, int]
    axis: tuple[tuple[int, int], tuple[int, int]] = ((0, 0), (0, 0))
    level: str = SAFE             # danger level from the hand ↔ item gap

    @property
    def length_px(self) -> float:
        return math.dist(*self.axis)


@dataclass
class Result:
    frame: object                  # annotated BGR frame
    dets: list[Detection]
    hands: list[Hand]
    cm_per_px: float | None
    scale_src: str                 # "calibrated" | "hand estimate" | "none"
    model_ready: bool
    fps: float
    finger_cm: list[dict[str, float]] = field(default_factory=list)  # per hand
    assess: Assessment | None = None


@dataclass(frozen=True)
class ModelSpec:
    mode: str                 # key of MODELS
    classes: tuple[str, ...]  # sharp classes (world modes)


class Camera:
    """Camera (or video file) that reconnects itself instead of giving up."""

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, source):
        self.source = source
        self.cap = None
        self.still = None
        if isinstance(source, str) and Path(source).suffix.lower() in self.IMAGE_EXTS:
            self.still = cv2.imread(source)  # a still image acts as a frozen camera

    def _open(self):
        if isinstance(self.source, int):
            backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
            cap = cv2.VideoCapture(self.source, backend)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        else:
            cap = cv2.VideoCapture(self.source)
        return cap if cap.isOpened() else None

    def read(self):
        """Returns a frame, or None if the camera is (temporarily) unavailable."""
        if self.still is not None:
            time.sleep(1 / 30)
            return self.still.copy()
        if self.cap is None:
            self.cap = self._open()
            if self.cap is None:
                return None
        ok, frame = self.cap.read()
        if ok:
            return frame
        if not isinstance(self.source, int):  # video file: loop it
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
            if ok:
                return frame
        self.release()  # force a reopen next time
        return None

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class Worker(threading.Thread):
    """Owns the camera for the whole session. Models load in the background
    and are swapped in without interrupting the video feed."""

    def __init__(self, source, spec: ModelSpec, conf: float, imgsz: int,
                 out: queue.Queue, status_cb):
        super().__init__(daemon=True)
        self.source = source
        self.out = out
        self.status_cb = status_cb
        self.conf = conf
        self.imgsz = imgsz
        self.track_hands = True
        self.cm_per_px: float | None = None   # manual calibration, overrides hand estimate
        self.thresholds = Thresholds()
        self.monitor = ProximityMonitor(self.thresholds)
        self.mirror = isinstance(source, int)
        self.device = pick_device()
        self.stop_event = threading.Event()

        self._model = None
        self._wanted = None
        self._ignore: set[str] = set()
        self._lock = threading.Lock()
        self._load_gen = 0
        self.request_model(spec)

    # ---- live-adjustable settings ----
    def request_model(self, spec: ModelSpec):
        """Load a new model in the background; the current one keeps running."""
        with self._lock:
            self._load_gen += 1
            gen = self._load_gen
        threading.Thread(target=self._load, args=(spec, gen), daemon=True).start()

    def set_source(self, source):
        self.source = source
        self.mirror = isinstance(source, int)

    def _load(self, spec: ModelSpec, gen: int):
        from ultralytics import YOLO

        weights = MODELS[spec.mode]
        try:
            self.status_cb(f"Loading {weights} (first time downloads it) — camera stays live…")
            model = YOLO(weights)
            if spec.mode.startswith("world"):
                decoys = [c for c in NOT_SHARP_CLASSES if c not in spec.classes]
                model.set_classes(list(spec.classes) + decoys)
                wanted, ignore = None, set(decoys)
            else:
                wanted = [i for i, n in model.names.items() if n in COCO_SHARP_CLASSES]
                ignore = set()
        except Exception as e:  # noqa: BLE001
            self.status_cb(f"Model load failed (camera still live): {e}")
            return
        with self._lock:
            if gen != self._load_gen:  # a newer request superseded this one
                return
            self._model, self._wanted, self._ignore = model, wanted, ignore
        self.status_cb(f"Detecting on {self.device.upper()}")

    # ---- main loop ----
    def run(self):
        cam = Camera(self.source)
        current_source = self.source
        tracker = None
        last = time.perf_counter()
        fps = 0.0
        cam_down = False
        try:
            while not self.stop_event.is_set():
                if self.source != current_source:
                    cam.release()
                    cam = Camera(self.source)
                    current_source = self.source

                frame = cam.read()
                if frame is None:
                    if not cam_down:
                        cam_down = True
                        self.status_cb(
                            f"Camera {current_source} unavailable — retrying… "
                            "(macOS: allow Camera access for your terminal)"
                        )
                    self.stop_event.wait(1.0)
                    continue
                if cam_down:
                    cam_down = False
                    self.status_cb("Camera reconnected")
                if self.mirror:
                    frame = cv2.flip(frame, 1)

                hands: list[Hand] = []
                if self.track_hands:
                    try:
                        tracker = tracker or HandTracker()
                        hands = tracker.detect(frame)
                    except Exception as e:  # noqa: BLE001
                        self.status_cb(f"Hand tracking error: {e}")

                with self._lock:
                    model, wanted, ignore = self._model, self._wanted, self._ignore
                dets: list[Detection] = []
                if model is not None:
                    try:
                        dets = self._detect(model, wanted, ignore, frame)
                    except Exception as e:  # noqa: BLE001 — never let a bad frame kill the feed
                        self.status_cb(f"Inference error (skipping frame): {e}")

                cm_per_px, src = self._scale(hands)
                assess = self.monitor.update(dets, hands, cm_per_px)
                th = self.thresholds
                for i, d in enumerate(dets):
                    g = assess.per_item.get(i)
                    if g is None or g.cm is None:
                        continue
                    if assess.closest is g:
                        d.level = assess.level  # includes the time-to-contact rule
                    elif g.cm <= th.contact_cm:
                        d.level = CONTACT
                    elif g.cm <= th.danger_cm:
                        d.level = DANGER
                    elif g.cm <= th.caution_cm:
                        d.level = CAUTION
                finger_cm = [
                    {f: px * self.cm_per_px for f, px in h.finger_px.items()}
                    if self.cm_per_px else dict(h.finger_cm_est)
                    for h in hands
                ]
                annotated = draw(frame, dets, hands, cm_per_px, finger_cm, assess)

                now = time.perf_counter()
                fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
                last = now

                # Keep only the newest frame so the UI never lags behind.
                try:
                    self.out.get_nowait()
                except queue.Empty:
                    pass
                self.out.put(Result(annotated, dets, hands, cm_per_px, src,
                                    model is not None, fps, finger_cm, assess))
        finally:
            cam.release()
            if tracker:
                tracker.close()

    def _scale(self, hands: list[Hand]) -> tuple[float | None, str]:
        if self.cm_per_px:
            return self.cm_per_px, "calibrated"
        est = [s for h in hands if (s := h.palm_scale_cm_per_px())]
        if est:
            return statistics.median(est), "hand estimate"
        return None, "none"

    def _detect(self, model, wanted, ignore, frame):
        res = model.predict(frame, conf=self.conf, imgsz=self.imgsz,
                            classes=wanted, device=self.device, verbose=False,
                            agnostic_nms=True)[0]  # one label per object, so decoys win
        dets = []
        for b in res.boxes:
            label = res.names[int(b.cls)]
            if label in ignore:
                continue
            box = tuple(map(int, b.xyxy[0].tolist()))
            dets.append(Detection(label, float(b.conf), box, item_axis(frame, box)))
        return dets

    def stop(self):
        self.stop_event.set()


def _label(img, text, x, y, color, scale=0.7, thick=2):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    y = max(y, th + 10)
    cv2.rectangle(img, (x, y - th - 10), (x + tw + 10, y), color, -1)
    cv2.putText(img, text, (x + 5, y - 6), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick)


def gap_text(g, cm_per_px) -> str:
    return fmt_len(g.px, cm_per_px) if g.px > 0 else "touching"


def draw(frame, dets: list[Detection], hands: list[Hand] = (), cm_per_px=None,
         finger_cm: list[dict] = (), assess: Assessment | None = None):
    img = frame.copy()
    for h, fcm in zip(hands, finger_cm or [{}] * len(hands)):
        for a, b in SKELETON:
            cv2.line(img, tuple(map(int, h.pts[a])), tuple(map(int, h.pts[b])), HAND_COLOR, 2)
        for i, p in enumerate(h.pts):
            cv2.circle(img, tuple(map(int, p)), 4 if i in TIPS else 2, (255, 255, 255), -1)
        for name, tip in zip(FINGERS, TIPS):
            if name in fcm:
                x, y = map(int, h.pts[tip])
                _label(img, f"{fcm[name]:.1f}", x + 6, y - 6, HAND_COLOR, 0.45, 1)
    for d in dets:
        x1, y1, x2, y2 = d.box
        color = LEVEL_BGR[d.level] if d.level != SAFE else (0, 140, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2 + LEVEL_RANK[d.level])
        (ax, ay), (bx, by) = d.axis
        cv2.line(img, (ax, ay), (bx, by), (255, 255, 255), 4)
        cv2.line(img, (ax, ay), (bx, by), color, 2)
        for p in d.axis:
            cv2.circle(img, p, 5, color, -1)
        text = f"{d.label} {d.conf:.0%}  len {fmt_len(d.length_px, cm_per_px)}"
        _label(img, text + (f"  {d.level}" if d.level != SAFE else ""), x1, y1, color)

    # Gap rulers: hand ↔ item, for every item (closest one drawn thickest).
    if assess:
        for i, g in assess.per_item.items():
            draw_gap_ruler(img, g, LEVEL_BGR[dets[i].level], cm_per_px,
                           assess.ttc_s if assess.closest is g else None,
                           major=assess.closest is g)
    return img


def draw_gap_ruler(img, g, color, cm_per_px, ttc_s, major: bool):
    """A measuring line from the closest point on the hand to the closest point
    on the item: end caps, a tick every cm, and the gap (+ time to contact)."""
    (hx, hy), (ix, iy) = g.p_hand, g.p_item
    ph, pi = (int(hx), int(hy)), (int(ix), int(iy))
    if g.px < 1:  # touching: mark the contact point
        cv2.circle(img, ph, 14 if major else 9, color, 3)
        cv2.circle(img, ph, 4, color, -1)
        _label(img, "TOUCHING", ph[0] + 16, ph[1], color, 0.75 if major else 0.5, 2 if major else 1)
        return
    ux, uy = (ix - hx) / g.px, (iy - hy) / g.px   # unit vector hand → item
    nx, ny = -uy, ux                               # perpendicular, for ticks
    thick = 3 if major else 1
    cv2.line(img, ph, pi, (255, 255, 255), thick + 3)
    cv2.line(img, ph, pi, color, thick)
    cap = 12 if major else 7
    for x, y in (ph, pi):  # end caps
        cv2.line(img, (int(x - nx * cap), int(y - ny * cap)),
                 (int(x + nx * cap), int(y + ny * cap)), color, thick + 1)
    if cm_per_px and major:  # tick every cm (every 5 cm if that gets crowded)
        step_cm = 1 if g.px * cm_per_px <= 30 else 5
        step_px = step_cm / cm_per_px
        if step_px >= 6:
            k = 1
            while k * step_px < g.px - 2:
                t = k * step_px
                x, y = hx + ux * t, hy + uy * t
                size = 8 if (k * step_cm) % 5 == 0 else 5
                cv2.line(img, (int(x - nx * size), int(y - ny * size)),
                         (int(x + nx * size), int(y + ny * size)), color, 2)
                k += 1
    text = fmt_len(g.px, cm_per_px)
    if ttc_s is not None:
        text += f" | contact in {ttc_s:.1f}s"
    if ny < 0:  # put the label on the lower side of the line (item labels sit on top)
        nx, ny = -nx, -ny
    mx, my = int((hx + ix) / 2 + nx * 24), int((hy + iy) / 2 + ny * 24 + 20)
    _label(img, text, mx, my, color, 0.75 if major else 0.5, 2 if major else 1)


def play_alert(root: tk.Misc, urgent: bool = False):
    """Non-blocking warning sound (macOS system sound, else the Tk bell)."""
    if sys.platform == "darwin":
        sound = "/System/Library/Sounds/" + ("Sosumi.aiff" if urgent else "Funk.aiff")
        try:
            subprocess.Popen(["afplay", sound], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except OSError:
            pass
    root.bell()


class App:
    def __init__(self, root: tk.Tk, source=None):
        self.root = root
        self.file_source = source  # video path overrides the camera
        root.title("Sharp Object Identifier")
        root.geometry("1320x840")
        root.minsize(980, 640)

        self.frames: queue.Queue = queue.Queue(maxsize=1)
        self.worker: Worker | None = None
        self.last_frame = None
        self.photo = None
        self.status_q: queue.Queue = queue.Queue()
        self.view = (1.0, 0, 0)          # display scale, x offset, y offset
        self.cal_points: list[tuple[float, float]] | None = None
        self.cm_per_px: float | None = None

        self.mode = tk.StringVar(value="world-l")
        self.hands = tk.BooleanVar(value=True)
        self.conf = tk.DoubleVar(value=0.25)
        self.imgsz = tk.StringVar(value="640")
        self.camera = tk.IntVar(value=0)
        self.status = tk.StringVar(value="Idle — press Start")
        self.fps = tk.StringVar(value="")
        self.scale_txt = tk.StringVar(value="Scale: pixels (no hand / not calibrated)")
        self.thresholds = Thresholds()
        self.caution_cm = tk.DoubleVar(value=self.thresholds.caution_cm)
        self.danger_cm = tk.DoubleVar(value=self.thresholds.danger_cm)
        self.alerts_on = tk.BooleanVar(value=True)
        self.last_level = SAFE
        self.last_alert_t = 0.0

        self._build()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.bind("<space>", lambda e: None if e.widget is self.classes_txt else self.toggle())
        root.bind("<Key-s>", lambda e: None if e.widget is self.classes_txt else self.snapshot())
        root.bind("<Escape>", lambda _e: self._cancel_calibration())
        self.root.after(15, self.tick)

    # ---------- layout ----------
    def _build(self):
        main = ttk.Frame(self.root, padding=8)
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)

        top = tk.Frame(main, bg="#555555")
        top.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.alert = tk.Label(top, text="CAMERA OFF", font=("Helvetica", 20, "bold"),
                              bg="#555555", fg="white", pady=4)
        self.alert.pack(fill="x")
        self.gap_lbl = tk.Label(top, text="", font=("Helvetica", 14), bg="#555555", fg="white",
                                pady=2)
        self.gap_lbl.pack(fill="x")
        self.banner = top

        self.video = tk.Label(main, bg="black", cursor="crosshair")
        self.video.grid(row=1, column=0, sticky="nsew")
        self.video.bind("<Button-1>", self._video_click)

        bar = ttk.Frame(main)
        bar.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        ttk.Label(bar, textvariable=self.status).pack(side="left")
        ttk.Label(bar, textvariable=self.fps).pack(side="right")

        side = ttk.Frame(main, padding=(10, 0, 0, 0))
        side.grid(row=0, column=1, rowspan=3, sticky="ns")

        self.start_btn = ttk.Button(side, text="Start  (space)", command=self.toggle)
        self.start_btn.pack(fill="x")
        ttk.Button(side, text="Snapshot  (s)", command=self.snapshot).pack(fill="x", pady=(4, 8))

        mode_box = ttk.LabelFrame(side, text="Model", padding=6)
        mode_box.pack(fill="x")
        for value, text in (("world-l", "Open vocabulary — accurate (World-L)"),
                            ("world-s", "Open vocabulary — fast (World-S)"),
                            ("coco", "COCO fast (knife/scissors/fork)")):
            ttk.Radiobutton(mode_box, text=text, value=value, variable=self.mode,
                            command=self._model_changed).pack(anchor="w")
        ttk.Checkbutton(mode_box, text="Track hands & fingers",
                        variable=self.hands, command=self._hands_changed).pack(anchor="w", pady=(4, 0))

        row = ttk.Frame(side)
        row.pack(fill="x", pady=(8, 0))
        ttk.Label(row, text="Camera index").pack(side="left")
        ttk.Spinbox(row, from_=0, to=9, width=4, textvariable=self.camera,
                    command=self._camera_changed).pack(side="right")
        row = ttk.Frame(side)
        row.pack(fill="x", pady=(4, 0))
        ttk.Label(row, text="Input size (bigger = small tips)").pack(side="left")
        size = ttk.Combobox(row, width=5, state="readonly", textvariable=self.imgsz,
                            values=("640", "960", "1280"))
        size.pack(side="right")
        size.bind("<<ComboboxSelected>>", self._imgsz_changed)

        conf_box = ttk.LabelFrame(side, text="Confidence threshold", padding=6)
        conf_box.pack(fill="x", pady=(8, 0))
        self.conf_lbl = ttk.Label(conf_box, text="0.25")
        self.conf_lbl.pack(anchor="e")
        ttk.Scale(conf_box, from_=0.05, to=0.9, variable=self.conf,
                  command=self._conf_changed).pack(fill="x")

        zone = ttk.LabelFrame(side, text="Danger zone (hand ↔ sharp item)", padding=6)
        zone.pack(fill="x", pady=(8, 0))
        for text, var in (("Caution within (cm)", self.caution_cm),
                          ("Danger within (cm)", self.danger_cm)):
            row = ttk.Frame(zone)
            row.pack(fill="x")
            ttk.Label(row, text=text).pack(side="left")
            ttk.Spinbox(row, from_=0.5, to=50, increment=0.5, width=5, textvariable=var,
                        command=self._zone_changed).pack(side="right")
        ttk.Checkbutton(zone, text="Warning sound on danger",
                        variable=self.alerts_on).pack(anchor="w", pady=(4, 0))

        meas = ttk.LabelFrame(side, text="Measurement", padding=6)
        meas.pack(fill="x", pady=(8, 0))
        ttk.Label(meas, textvariable=self.scale_txt, wraplength=280).pack(anchor="w")
        btns = ttk.Frame(meas)
        btns.pack(fill="x", pady=(4, 0))
        self.cal_btn = ttk.Button(btns, text="Calibrate…", command=self._start_calibration)
        self.cal_btn.pack(side="left")
        ttk.Button(btns, text="Clear", command=self._clear_calibration).pack(side="left", padx=4)

        cls_box = ttk.LabelFrame(side, text="Sharp classes (one per line)", padding=6)
        cls_box.pack(fill="x", pady=(8, 0))
        txt_row = ttk.Frame(cls_box)
        txt_row.pack(fill="x")
        self.classes_txt = tk.Text(txt_row, height=6, width=30, undo=True)
        self.classes_txt.insert("1.0", "\n".join(DEFAULT_SHARP_CLASSES))
        sb = ttk.Scrollbar(txt_row, command=self.classes_txt.yview)
        self.classes_txt.configure(yscrollcommand=sb.set)
        self.classes_txt.pack(side="left", fill="x", expand=True)
        sb.pack(side="right", fill="y")
        btns = ttk.Frame(cls_box)
        btns.pack(fill="x", pady=(4, 0))
        self.apply_btn = ttk.Button(btns, text="Apply", command=self._model_changed)
        self.apply_btn.pack(side="left")
        ttk.Button(btns, text="Reset", command=self._reset_classes).pack(side="left", padx=4)

        out_box = ttk.LabelFrame(side, text="Sharp items & fingers", padding=6)
        out_box.pack(fill="both", expand=True, pady=(8, 0))
        self.det_list = tk.Listbox(out_box, height=8, width=32, font=("Menlo", 11))
        self.det_list.pack(fill="both", expand=True)

    # ---------- settings ----------
    def _spec(self) -> ModelSpec | None:
        classes = tuple(c.strip() for c in self.classes_txt.get("1.0", "end").splitlines()
                        if c.strip())
        if self.mode.get().startswith("world") and not classes:
            self.status.set("Add at least one class")
            return None
        return ModelSpec(self.mode.get(), classes)

    def _model_changed(self):
        state = "normal" if self.mode.get().startswith("world") else "disabled"
        self.classes_txt.configure(state=state)
        self.apply_btn.configure(state=state)
        if self.worker and (spec := self._spec()):
            self.worker.request_model(spec)  # hot swap — camera keeps running

    def _reset_classes(self):
        self.classes_txt.delete("1.0", "end")
        self.classes_txt.insert("1.0", "\n".join(DEFAULT_SHARP_CLASSES))
        self._model_changed()

    def _hands_changed(self):
        if self.worker:
            self.worker.track_hands = self.hands.get()

    def _zone_changed(self):
        try:
            caution, danger = float(self.caution_cm.get()), float(self.danger_cm.get())
        except (tk.TclError, ValueError):
            return
        self.thresholds.caution_cm = max(caution, danger)
        self.thresholds.danger_cm = min(caution, danger)
        self.thresholds.contact_cm = min(1.0, self.thresholds.danger_cm / 2)

    def _camera_changed(self):
        if self.worker and self.file_source is None:
            self.worker.set_source(int(self.camera.get()))

    def _imgsz_changed(self, _e=None):
        if self.worker:
            self.worker.imgsz = int(self.imgsz.get())

    def _conf_changed(self, _v):
        v = round(self.conf.get(), 2)
        self.conf_lbl.configure(text=f"{v:.2f}")
        if self.worker:
            self.worker.conf = v

    def set_status(self, msg: str):  # called from worker threads
        self.status_q.put(msg)

    # ---------- calibration ----------
    def _start_calibration(self):
        if self.last_frame is None:
            self.status.set("Start the camera first")
            return
        self.cal_points = []
        self.status.set("Calibrate: click both ends of something of known length "
                        "(e.g. a ruler) held where you'll measure. Esc to cancel.")

    def _cancel_calibration(self):
        if self.cal_points is not None:
            self.cal_points = None
            self.status.set("Calibration cancelled")

    def _clear_calibration(self):
        self.cm_per_px = None
        if self.worker:
            self.worker.cm_per_px = None
        self.status.set("Calibration cleared — using hand estimate when a hand is visible")

    def _video_click(self, e):
        if self.cal_points is None:
            return
        s, ox, oy = self.view
        self.cal_points.append(((e.x - ox) / s, (e.y - oy) / s))
        if len(self.cal_points) < 2:
            self.status.set("Calibrate: now click the other end")
            return
        px = math.dist(*self.cal_points)
        cm = simpledialog.askfloat("Calibrate", f"Real distance between the points ({px:.0f} px), in cm:",
                                   parent=self.root, minvalue=0.1, maxvalue=1000)
        self.cal_points = None
        if cm and px > 5:
            self.cm_per_px = cm / px
            if self.worker:
                self.worker.cm_per_px = self.cm_per_px
            self.status.set(f"Calibrated: {px:.0f} px = {cm:g} cm")
        else:
            self.status.set("Calibration cancelled")

    # ---------- start / stop ----------
    def toggle(self):
        self.stop() if self.worker else self.start()

    def start(self):
        spec = self._spec()
        if spec is None:
            return
        source = self.file_source if self.file_source is not None else int(self.camera.get())
        self.worker = Worker(source, spec, round(self.conf.get(), 2), int(self.imgsz.get()),
                             self.frames, self.set_status)
        self.worker.track_hands = self.hands.get()
        self.worker.cm_per_px = self.cm_per_px
        self._zone_changed()
        self.worker.thresholds = self.thresholds
        self.worker.monitor = ProximityMonitor(self.thresholds)
        self.worker.start()
        self.start_btn.configure(text="Stop  (space)")

    def stop(self):
        if self.worker:
            self.worker.stop()
            self.worker.join(timeout=3)
            self.worker = None
        self.start_btn.configure(text="Start  (space)")
        self.status.set("Stopped")
        self.fps.set("")
        self._set_banner("CAMERA OFF", "", "#555555")

    def snapshot(self):
        if self.last_frame is None:
            return
        SNAPSHOT_DIR.mkdir(exist_ok=True)
        path = SNAPSHOT_DIR / f"sharp_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
        cv2.imwrite(str(path), self.last_frame)
        self.status.set(f"Saved {path.name}")

    # ---------- render loop ----------
    def tick(self):
        while not self.status_q.empty():
            self.status.set(self.status_q.get_nowait())

        if self.worker and not self.worker.is_alive():
            # Should not happen (the worker retries forever) — restart it.
            self.status.set("Camera worker stopped unexpectedly — restarting")
            self.worker = None
            self.start()

        try:
            r: Result = self.frames.get_nowait()
        except queue.Empty:
            pass
        else:
            self.last_frame = r.frame
            self._show(r.frame)
            self._update_panel(r)
            self.fps.set(f"{r.fps:.1f} FPS")
        self.root.after(15, self.tick)

    def _show(self, frame):
        if self.cal_points:
            frame = frame.copy()
            for p in self.cal_points:
                cv2.circle(frame, tuple(map(int, p)), 7, CAL_COLOR, -1)
        w = max(self.video.winfo_width(), 1)
        h = max(self.video.winfo_height(), 1)
        fh, fw = frame.shape[:2]
        s = min(w / fw, h / fh)
        dw, dh = max(int(fw * s), 1), max(int(fh * s), 1)
        self.view = (s, (w - dw) / 2, (h - dh) / 2)  # Label centres the image
        frame = cv2.resize(frame, (dw, dh))
        self.photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        self.video.configure(image=self.photo)

    def _update_panel(self, r: Result):
        if r.scale_src == "calibrated":
            self.scale_txt.set(f"Scale: calibrated ({1 / r.cm_per_px:.1f} px/cm)")
        elif r.scale_src == "hand estimate":
            self.scale_txt.set("Scale: estimated from hand size (≈, calibrate for accuracy)")
        else:
            self.scale_txt.set("Scale: pixels (no hand / not calibrated)")

        self.det_list.delete(0, "end")
        if r.dets:
            self.det_list.insert("end", "  item             gap  length")
        a = r.assess
        gaps = a.per_item if a else {}
        order = sorted(range(len(r.dets)),
                       key=lambda i: (-LEVEL_RANK[r.dets[i].level], gaps[i].px if i in gaps else 1e9))
        for i in order:
            d = r.dets[i]
            gap = gap_text(gaps[i], r.cm_per_px) if i in gaps else "no hand"
            self.det_list.insert("end", f"  {d.label[:11]:<11} {gap:>8} "
                                        f"{fmt_len(d.length_px, r.cm_per_px):>7}")
            if d.level != SAFE:  # colour the row like the banner instead of an icon
                self.det_list.itemconfigure("end", foreground=LEVEL_UI[d.level])
        for h, fcm in zip(r.hands, r.finger_cm):
            self.det_list.insert("end", f"  {h.side} hand (cm{', est.' if not self.cm_per_px else ''})")
            for name, v in fcm.items():
                self.det_list.insert("end", f"     {name:<8} {v:5.1f} cm")

        if not r.model_ready:
            self._set_banner("CAMERA LIVE — LOADING MODEL…", "", "#555555")
            return
        if not r.dets:
            extra = f"{len(r.hands)} hand{'s' * (len(r.hands) != 1)} tracked" if r.hands else ""
            self._set_banner("SAFE — NO SHARP OBJECTS", extra, LEVEL_UI[SAFE])
            self.last_level = SAFE
            return
        names = ", ".join(sorted({d.label for d in r.dets})).upper()
        if not a or a.closest is None:
            self._set_banner(f"SHARP OBJECT IN VIEW: {names}",
                             "No hand detected — distance unknown", LEVEL_UI[CAUTION])
            self.last_level = SAFE
            return

        g = a.closest
        item = r.dets[g.item].label
        dist = gap_text(g, r.cm_per_px)
        motion = ""
        if a.speed_cm_s is not None and abs(a.speed_cm_s) >= 2:
            motion = (f" · approaching {a.speed_cm_s:.0f} cm/s" if a.speed_cm_s > 0
                      else f" · moving away {-a.speed_cm_s:.0f} cm/s")
        ttc = f" · contact in {a.ttc_s:.1f} s" if a.ttc_s is not None else ""
        detail = f"{g.part} ↔ {item}: {dist}{motion}{ttc}"
        title = {
            SAFE: f"SAFE — {item.upper()} {dist} AWAY",
            CAUTION: f"CAUTION — HAND {dist} FROM {item.upper()}",
            DANGER: f"DANGER — {g.part.upper()} {dist} FROM {item.upper()}",
            CONTACT: (f"CONTACT — {g.part.upper()} TOUCHING {item.upper()}" if g.px < 1 else
                      f"CONTACT — {g.part.upper()} {dist} FROM {item.upper()}"),
        }[a.level]
        self._set_banner(title, detail, LEVEL_UI[a.level])
        self._maybe_alert(a.level, approaching=a.speed_cm_s is None or a.speed_cm_s > -2)
        self.last_level = a.level

    def _set_banner(self, title: str, detail: str, bg: str):
        self.banner.configure(bg=bg)
        self.alert.configure(text=title, bg=bg)
        self.gap_lbl.configure(text=detail, bg=bg)

    def _maybe_alert(self, level: str, approaching: bool = True):
        """Warning sound when entering DANGER/CONTACT; repeats every 3 s while still closing in."""
        if not self.alerts_on.get() or LEVEL_RANK[level] < LEVEL_RANK[DANGER]:
            return
        now = time.monotonic()
        escalated = LEVEL_RANK[level] > LEVEL_RANK[self.last_level]
        still_in_danger = LEVEL_RANK[self.last_level] >= LEVEL_RANK[DANGER]
        if escalated:
            if now - self.last_alert_t < 0.5:
                return
        elif not (still_in_danger and approaching and now - self.last_alert_t >= 3.0):
            return  # no repeat alerts while backing away
        self.last_alert_t = now
        play_alert(self.root, urgent=level == CONTACT)

    def on_close(self):
        self.stop()
        self.root.destroy()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", help="video file or image to use instead of the camera (for testing)")
    p.add_argument("--autostart", action="store_true", help="start detection immediately")
    args = p.parse_args()

    root = tk.Tk()
    app = App(root, source=args.source)
    if args.autostart:
        root.after(200, app.start)
    root.mainloop()


if __name__ == "__main__":
    main()
