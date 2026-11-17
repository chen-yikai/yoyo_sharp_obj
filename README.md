# Sharp Object Identifier

Live YOLO detection of sharp items from your desktop camera, with a Tkinter UI.

![Sharp Object Identifier detecting cutlery](docs/screenshot.png)

*YOLO-World mode on a cutlery photo (confidence 0.40). The real fork and knife are caught, but the two spoons and the dinner knife are also tagged "fork". See [Limitations](#limitations).*

## Quick start

Requires [uv](https://docs.astral.sh/uv/) (installs Python 3.12 and all dependencies automatically).

```bash
cd sharp_obj
uv run app.py
```

Press **Start** (or `space`) to turn on the camera. Model weights download on the first start (~30 MB for YOLO-World, ~6 MB for YOLO11n).

On macOS, the first start asks for Camera permission for your terminal. If you denied it, enable it under **System Settings › Privacy & Security › Camera** and restart the terminal.

Without uv:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install ultralytics opencv-python pillow "clip @ git+https://github.com/ultralytics/CLIP.git"
python app.py
```

## Detection modes

| Mode | Model | Detects | Speed |
|---|---|---|---|
| **Open vocabulary** (default) | `yolov8s-worldv2.pt` | Anything you list in the *Sharp classes* box | Slower |
| **COCO fast** | `yolo11n.pt` | knife, scissors, fork only | Faster |

Default sharp classes: knife, kitchen knife, scissors, razor blade, box cutter, needle, syringe, screwdriver, fork, saw, axe, broken glass. Edit the list (one per line) and press Start to apply it.

## UI

- **Alert banner**: green when nothing sharp is in view, red with the item names when something is.
- **Video**: mirrored camera feed with a red box and confidence score on each detection.
- **Confidence threshold**: slider, applied live. Raise it to cut false positives.
- **Camera index**: pick another camera if you have more than one (applied on Start).
- **Detections**: current hits sorted by confidence.
- **Status bar**: device in use (MPS / CUDA / CPU) and FPS.

| Key | Action |
|---|---|
| `space` | Start / stop |
| `s` | Save the annotated frame to `snapshots/` |

## Limitations

- Out-of-the-box models are not trained specifically for sharp objects. Similar-looking items get confused, as with the spoons labelled "fork" in the screenshot. Raise the confidence threshold or remove classes you don't care about (e.g. `fork`).
- Small, thin, or edge-on blades (needles, razor blades) are hard to see at webcam resolution.
- For dependable results, fine-tune a YOLO model on your own labelled images of the objects you care about.
- This is a demo, not a safety system. Don't rely on it for security screening.

## Files

```
app.py            Tk app + detection thread
pyproject.toml    Dependencies (uv)
docs/screenshot.png
snapshots/        Created on first snapshot
```
