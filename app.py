"""Sharp object identifier — live YOLO detection from the desktop camera with a Tk UI.

Two detection modes:
  * Open vocabulary (YOLO-World): detects any sharp item you list by name,
    and can warn when a sharp item is close to a hand.
  * COCO (YOLO11): fast, but only knows "knife", "scissors" and "fork".

The camera stays live from Start until Stop: model reloads, class edits,
resolution changes and camera hiccups never close the feed.

Run:  uv run app.py              (desktop camera)
      uv run app.py --source clip.mp4   (video file, for testing)
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import ttk

import cv2
from PIL import Image, ImageTk

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
HAND_CLASS = "hand"

MODEL_WORLD = "yolov8s-worldv2.pt"
MODEL_COCO = "yolo11n.pt"

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"
SHARP_COLOR = (0, 140, 255)   # BGR orange
DANGER_COLOR = (0, 0, 255)    # BGR red
HAND_COLOR = (255, 160, 0)    # BGR blue
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
    near_hand: bool = False


@dataclass(frozen=True)
class ModelSpec:
    mode: str                 # "world" | "coco"
    classes: tuple[str, ...]  # sharp classes (world mode)
    hands: bool               # also detect hands (world mode)


class Camera:
    """Camera (or video file) that reconnects itself instead of giving up."""

    def __init__(self, source):
        self.source = source
        self.cap = None

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
        self.mirror = isinstance(source, int)
        self.device = pick_device()
        self.stop_event = threading.Event()

        self._model = None
        self._model_spec: ModelSpec | None = None
        self._wanted = None
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

        try:
            if spec.mode == "world":
                self.status_cb(f"Loading {MODEL_WORLD} — camera stays live…")
                model = YOLO(MODEL_WORLD)
                names = list(spec.classes) + ([HAND_CLASS] if spec.hands else [])
                model.set_classes(names)
                wanted = None
            else:
                self.status_cb(f"Loading {MODEL_COCO} — camera stays live…")
                model = YOLO(MODEL_COCO)
                wanted = [i for i, n in model.names.items() if n in COCO_SHARP_CLASSES]
        except Exception as e:  # noqa: BLE001
            self.status_cb(f"Model load failed (camera still live): {e}")
            return
        with self._lock:
            if gen != self._load_gen:  # a newer request superseded this one
                return
            self._model, self._model_spec, self._wanted = model, spec, wanted
        self.status_cb(f"Detecting on {self.device.upper()}")

    # ---- main loop ----
    def run(self):
        cam = Camera(self.source)
        current_source = self.source
        last = time.perf_counter()
        fps = 0.0
        cam_down_since = None
        try:
            while not self.stop_event.is_set():
                if self.source != current_source:
                    cam.release()
                    cam = Camera(self.source)
                    current_source = self.source

                frame = cam.read()
                if frame is None:
                    if cam_down_since is None:
                        cam_down_since = time.time()
                        self.status_cb(
                            f"Camera {current_source} unavailable — retrying… "
                            "(macOS: allow Camera access for your terminal)"
                        )
                    self.stop_event.wait(1.0)
                    continue
                if cam_down_since is not None:
                    cam_down_since = None
                    self.status_cb("Camera reconnected")
                if self.mirror:
                    frame = cv2.flip(frame, 1)

                with self._lock:
                    model, spec, wanted = self._model, self._model_spec, self._wanted

                dets: list[Detection] = []
                hands: list[Detection] = []
                if model is not None:
                    try:
                        dets, hands = self._detect(model, spec, wanted, frame)
                    except Exception as e:  # noqa: BLE001 — never let a bad frame kill the feed
                        self.status_cb(f"Inference error (skipping frame): {e}")

                annotated = draw(frame, dets, hands)
                now = time.perf_counter()
                fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
                last = now

                # Keep only the newest frame so the UI never lags behind.
                try:
                    self.out.get_nowait()
                except queue.Empty:
                    pass
                self.out.put((annotated, dets, bool(hands), model is not None, fps))
        finally:
            cam.release()

    def _detect(self, model, spec: ModelSpec, wanted, frame):
        res = model.predict(frame, conf=self.conf, imgsz=self.imgsz,
                            classes=wanted, device=self.device, verbose=False)[0]
        sharp, hands = [], []
        for b in res.boxes:
            label = res.names[int(b.cls)]
            box = tuple(map(int, b.xyxy[0].tolist()))
            d = Detection(label, float(b.conf), box)
            (hands if spec.hands and label == HAND_CLASS else sharp).append(d)
        for d in sharp:
            d.near_hand = any(near(d.box, h.box) for h in hands)
        return sharp, hands

    def stop(self):
        self.stop_event.set()


def near(a, b) -> bool:
    """True if box a overlaps box b grown by HAND_MARGIN of its size."""
    bx1, by1, bx2, by2 = b
    mx, my = (bx2 - bx1) * HAND_MARGIN, (by2 - by1) * HAND_MARGIN
    bx1, by1, bx2, by2 = bx1 - mx, by1 - my, bx2 + mx, by2 + my
    ax1, ay1, ax2, ay2 = a
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1


def _label(img, text, x1, y1, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    ty = max(y1, th + 10)
    cv2.rectangle(img, (x1, ty - th - 10), (x1 + tw + 10, ty), color, -1)
    cv2.putText(img, text, (x1 + 5, ty - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def draw(frame, dets: list[Detection], hands: list[Detection] = ()):
    img = frame.copy()
    for h in hands:
        x1, y1, x2, y2 = h.box
        cv2.rectangle(img, (x1, y1), (x2, y2), HAND_COLOR, 2)
        _label(img, f"hand {h.conf:.0%}", x1, y1, HAND_COLOR)
    for d in dets:
        x1, y1, x2, y2 = d.box
        color = DANGER_COLOR if d.near_hand else SHARP_COLOR
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        _label(img, f"{d.label} {d.conf:.0%}" + ("  NEAR HAND" if d.near_hand else ""), x1, y1, color)
    return img


class App:
    def __init__(self, root: tk.Tk, source=None):
        self.root = root
        self.file_source = source  # video/image path overrides the camera
        root.title("Sharp Object Identifier")
        root.geometry("1280x800")
        root.minsize(900, 600)

        self.frames: queue.Queue = queue.Queue(maxsize=1)
        self.worker: Worker | None = None
        self.last_frame = None
        self.photo = None
        self.status_q: queue.Queue = queue.Queue()

        self.mode = tk.StringVar(value="world")
        self.hands = tk.BooleanVar(value=True)
        self.conf = tk.DoubleVar(value=0.25)
        self.imgsz = tk.StringVar(value="640")
        self.camera = tk.IntVar(value=0)
        self.status = tk.StringVar(value="Idle — press Start")
        self.fps = tk.StringVar(value="")

        self._build()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.bind("<space>", lambda e: None if e.widget is self.classes_txt else self.toggle())
        root.bind("<Key-s>", lambda e: None if e.widget is self.classes_txt else self.snapshot())
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

        self.video = tk.Label(main, bg="black")
        self.video.grid(row=1, column=0, sticky="nsew")

        bar = ttk.Frame(main)
        bar.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        ttk.Label(bar, textvariable=self.status).pack(side="left")
        ttk.Label(bar, textvariable=self.fps).pack(side="right")

        side = ttk.Frame(main, padding=(10, 0, 0, 0))
        side.grid(row=0, column=1, rowspan=3, sticky="ns")

        self.start_btn = ttk.Button(side, text="Start  (space)", command=self.toggle)
        self.start_btn.pack(fill="x")
        ttk.Button(side, text="Snapshot  (s)", command=self.snapshot).pack(fill="x", pady=(4, 10))

        mode_box = ttk.LabelFrame(side, text="Model", padding=6)
        mode_box.pack(fill="x")
        ttk.Radiobutton(mode_box, text="Open vocabulary (YOLO-World)", value="world",
                        variable=self.mode, command=self._model_changed).pack(anchor="w")
        ttk.Radiobutton(mode_box, text="COCO fast (knife/scissors/fork)", value="coco",
                        variable=self.mode, command=self._model_changed).pack(anchor="w")
        self.hands_chk = ttk.Checkbutton(mode_box, text="Warn when sharp item is near a hand",
                                         variable=self.hands, command=self._model_changed)
        self.hands_chk.pack(anchor="w", pady=(4, 0))

        row = ttk.Frame(side)
        row.pack(fill="x", pady=(10, 0))
        ttk.Label(row, text="Camera index").pack(side="left")
        ttk.Spinbox(row, from_=0, to=9, width=4, textvariable=self.camera,
                    command=self._camera_changed).pack(side="right")

        row = ttk.Frame(side)
        row.pack(fill="x", pady=(6, 0))
        ttk.Label(row, text="Input size (bigger = small tips)").pack(side="left")
        size = ttk.Combobox(row, width=5, state="readonly", textvariable=self.imgsz,
                            values=("640", "960", "1280"))
        size.pack(side="right")
        size.bind("<<ComboboxSelected>>", self._imgsz_changed)

        conf_box = ttk.LabelFrame(side, text="Confidence threshold", padding=6)
        conf_box.pack(fill="x", pady=(10, 0))
        self.conf_lbl = ttk.Label(conf_box, text="0.25")
        self.conf_lbl.pack(anchor="e")
        ttk.Scale(conf_box, from_=0.05, to=0.9, variable=self.conf,
                  command=self._conf_changed).pack(fill="x")

        cls_box = ttk.LabelFrame(side, text="Sharp classes (one per line)", padding=6)
        cls_box.pack(fill="x", pady=(10, 0))
        txt_row = ttk.Frame(cls_box)
        txt_row.pack(fill="x")
        self.classes_txt = tk.Text(txt_row, height=9, width=30, undo=True)
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

        det_box = ttk.LabelFrame(side, text="Detections", padding=6)
        det_box.pack(fill="both", expand=True, pady=(10, 0))
        self.det_list = tk.Listbox(det_box, height=7)
        self.det_list.pack(fill="both", expand=True)

    # ---------- settings ----------
    def _spec(self) -> ModelSpec | None:
        classes = tuple(c.strip() for c in self.classes_txt.get("1.0", "end").splitlines()
                        if c.strip())
        world = self.mode.get() == "world"
        if world and not classes:
            self.status.set("Add at least one class")
            return None
        return ModelSpec(self.mode.get(), classes, world and self.hands.get())

    def _model_changed(self):
        world = self.mode.get() == "world"
        state = "normal" if world else "disabled"
        self.classes_txt.configure(state=state)
        self.apply_btn.configure(state=state)
        self.hands_chk.configure(state=state)
        if self.worker and (spec := self._spec()):
            self.worker.request_model(spec)  # hot swap — camera keeps running

    def _reset_classes(self):
        self.classes_txt.delete("1.0", "end")
        self.classes_txt.insert("1.0", "\n".join(DEFAULT_SHARP_CLASSES))
        self._model_changed()

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
            self.set_status("Camera worker stopped unexpectedly — restarting")
            self.worker = None
            self.start()

        try:
            frame, dets, hands_seen, model_ready, fps = self.frames.get_nowait()
        except queue.Empty:
            pass
        else:
            self.last_frame = frame
            self._show(frame)
            self._update_dets(dets, hands_seen, model_ready)
            self.fps.set(f"{fps:.1f} FPS")
        self.root.after(15, self.tick)

    def _show(self, frame):
        w = max(self.video.winfo_width(), 1)
        h = max(self.video.winfo_height(), 1)
        fh, fw = frame.shape[:2]
        scale = min(w / fw, h / fh)
        if scale > 0:
            frame = cv2.resize(frame, (max(int(fw * scale), 1), max(int(fh * scale), 1)))
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.photo = ImageTk.PhotoImage(img)
        self.video.configure(image=self.photo)

    def _update_dets(self, dets: list[Detection], hands_seen: bool, model_ready: bool):
        self.det_list.delete(0, "end")
        for d in sorted(dets, key=lambda d: (not d.near_hand, -d.conf)):
            self.det_list.insert("end", f"{'⚠ ' if d.near_hand else ''}{d.label}  {d.conf:.0%}")
        danger = sorted({d.label for d in dets if d.near_hand})
        sharp = sorted({d.label for d in dets})
        if not model_ready:
            self.alert.configure(text="CAMERA LIVE — LOADING MODEL…", bg="#555555")
        elif danger:
            self.alert.configure(text=f"⚠ DANGER: {', '.join(danger).upper()} NEAR HAND",
                                 bg="#c62828")
        elif sharp:
            self.alert.configure(text=f"SHARP OBJECT: {', '.join(sharp).upper()}", bg="#e67e22")
        else:
            extra = " (hand in view)" if hands_seen else ""
            self.alert.configure(text=f"NO SHARP OBJECTS{extra}", bg="#1e7f3c")

    def on_close(self):
        self.stop()
        self.root.destroy()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", help="video file to use instead of the camera (for testing)")
    p.add_argument("--autostart", action="store_true", help="start detection immediately")
    args = p.parse_args()

    root = tk.Tk()
    app = App(root, source=args.source)
    if args.autostart:
        root.after(200, app.start)
    root.mainloop()


if __name__ == "__main__":
    main()
