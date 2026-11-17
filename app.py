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
SHARP_COLOR = (0, 140, 255)   # BGR orange
DANGER_COLOR = (0, 0, 255)    # BGR red
HAND_COLOR = (255, 160, 0)    # BGR blue
CAL_COLOR = (0, 220, 0)       # BGR green
HAND_MARGIN = 0.25            # grow hand box by 25% when checking proximity


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
    near_hand: bool = False

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
                        dets = self._detect(model, wanted, ignore, frame, hands)
                    except Exception as e:  # noqa: BLE001 — never let a bad frame kill the feed
                        self.status_cb(f"Inference error (skipping frame): {e}")

                cm_per_px, src = self._scale(hands)
                finger_cm = [
                    {f: px * self.cm_per_px for f, px in h.finger_px.items()}
                    if self.cm_per_px else dict(h.finger_cm_est)
                    for h in hands
                ]
                annotated = draw(frame, dets, hands, cm_per_px, finger_cm)

                now = time.perf_counter()
                fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
                last = now

                # Keep only the newest frame so the UI never lags behind.
                try:
                    self.out.get_nowait()
                except queue.Empty:
                    pass
                self.out.put(Result(annotated, dets, hands, cm_per_px, src,
                                    model is not None, fps, finger_cm))
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

    def _detect(self, model, wanted, ignore, frame, hands: list[Hand]):
        res = model.predict(frame, conf=self.conf, imgsz=self.imgsz,
                            classes=wanted, device=self.device, verbose=False,
                            agnostic_nms=True)[0]  # one label per object, so decoys win
        dets = []
        for b in res.boxes:
            label = res.names[int(b.cls)]
            if label in ignore:
                continue
            box = tuple(map(int, b.xyxy[0].tolist()))
            dets.append(Detection(label, float(b.conf), box, item_axis(frame, box),
                                  any(near(box, h.box) for h in hands)))
        return dets

    def stop(self):
        self.stop_event.set()


def near(a, b) -> bool:
    """True if box a overlaps box b grown by HAND_MARGIN of its size."""
    bx1, by1, bx2, by2 = b
    mx, my = (bx2 - bx1) * HAND_MARGIN, (by2 - by1) * HAND_MARGIN
    bx1, by1, bx2, by2 = bx1 - mx, by1 - my, bx2 + mx, by2 + my
    ax1, ay1, ax2, ay2 = a
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1


def _label(img, text, x, y, color, scale=0.7, thick=2):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    y = max(y, th + 10)
    cv2.rectangle(img, (x, y - th - 10), (x + tw + 10, y), color, -1)
    cv2.putText(img, text, (x + 5, y - 6), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick)


def draw(frame, dets: list[Detection], hands: list[Hand] = (), cm_per_px=None,
         finger_cm: list[dict] = ()):
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
        color = DANGER_COLOR if d.near_hand else SHARP_COLOR
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        (ax, ay), (bx, by) = d.axis
        cv2.line(img, (ax, ay), (bx, by), (255, 255, 255), 4)
        cv2.line(img, (ax, ay), (bx, by), color, 2)
        for p in d.axis:
            cv2.circle(img, p, 5, color, -1)
        text = f"{d.label} {d.conf:.0%}  {fmt_len(d.length_px, cm_per_px)}"
        _label(img, text + ("  NEAR HAND" if d.near_hand else ""), x1, y1, color)
    return img


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

        self.alert = tk.Label(main, text="CAMERA OFF", font=("Helvetica", 18, "bold"),
                              bg="#555555", fg="white", pady=6)
        self.alert.grid(row=0, column=0, sticky="ew", pady=(0, 6))

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
        ttk.Checkbutton(mode_box, text="Track hands & fingers, warn when near",
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

        out_box = ttk.LabelFrame(side, text="Detections & lengths", padding=6)
        out_box.pack(fill="both", expand=True, pady=(8, 0))
        self.det_list = tk.Listbox(out_box, height=8, font=("Menlo", 11))
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
        self.alert.configure(text="CAMERA OFF", bg="#555555")

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
        for d in sorted(r.dets, key=lambda d: (not d.near_hand, -d.conf)):
            self.det_list.insert("end", f"{'⚠' if d.near_hand else ' '} {d.label[:16]:<16} "
                                        f"{fmt_len(d.length_px, r.cm_per_px):>8}")
        for h, fcm in zip(r.hands, r.finger_cm):
            self.det_list.insert("end", f"✋ {h.side} hand (cm{', est.' if not self.cm_per_px else ''})")
            for name, v in fcm.items():
                self.det_list.insert("end", f"     {name:<8} {v:5.1f} cm")

        danger = sorted({d.label for d in r.dets if d.near_hand})
        sharp = sorted({d.label for d in r.dets})
        if not r.model_ready:
            self.alert.configure(text="CAMERA LIVE — LOADING MODEL…", bg="#555555")
        elif danger:
            self.alert.configure(text=f"⚠ DANGER: {', '.join(danger).upper()} NEAR HAND",
                                 bg="#c62828")
        elif sharp:
            self.alert.configure(text=f"SHARP OBJECT: {', '.join(sharp).upper()}", bg="#e67e22")
        else:
            extra = f" ({len(r.hands)} hand{'s' * (len(r.hands) != 1)} tracked)" if r.hands else ""
            self.alert.configure(text=f"NO SHARP OBJECTS{extra}", bg="#1e7f3c")

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
