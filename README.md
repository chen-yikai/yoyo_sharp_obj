# Sharp Object Identifier

Live detection of sharp items from your desktop camera, with a Tkinter UI. It finds blades, points and needles with YOLO, tracks your hands and fingers with MediaPipe, warns when a sharp item gets close to a hand, and measures the length of every item and finger.

![Sharp Object Identifier warning that a sharp item is near a hand](docs/screenshot.png)

*A statue's hand holding a stylus (still-image source, YOLO-World L). The stylus is flagged and measured along its length, the hand's skeleton and finger lengths are shown, and the banner turns red because the item is touching the hand. Note the label says "knife": the model often picks the wrong name for an item even when it correctly flags it as sharp. See [Limitations](#limitations).*

## Quick start

Requires [uv](https://docs.astral.sh/uv/), which installs Python 3.12 and all dependencies automatically.

```bash
cd sharp_obj
uv run app.py
```

Press **Start** (or `space`) to turn on the camera. Model weights download on first use: about 90 MB for YOLO-World L, 8 MB for the hand model.

On macOS, the first start asks for Camera permission for your terminal. If you denied it, enable it under **System Settings › Privacy & Security › Camera** and restart the terminal.

To test without a camera, pass a video or an image:

```bash
uv run app.py --source clip.mp4 --autostart
uv run app.py --source photo.jpg --autostart
```

## What it does

**Detects sharp items.** By default it looks for 43 kinds of item that can cut or puncture a hand: knives, box cutters, scalpels, razor blades and scissors; pens, pen nibs, pencils, compasses, thumbtacks and push pins; needles, syringes, safety pins, nails, screwdrivers and skewers; broken glass, metal shards and can lids. You can edit the list in the app.

**Ignores harmless look-alikes.** Spoons, chopsticks, markers, phones and similar items are also given to the model as options, then dropped from the results. That gives the model a better name for a spoon than "knife".

**Tracks hands and fingers.** MediaPipe finds 21 points on each hand and draws a skeleton over it. The banner shows:

| Banner | Meaning |
|---|---|
| 🟩 NO SHARP OBJECTS | Nothing sharp in view |
| 🟧 SHARP OBJECT: … | Sharp item in view |
| 🟥 DANGER: … NEAR HAND | A sharp item overlaps or touches a hand |

**Measures lengths.**
- **Sharp items** are measured along their long axis (the line drawn on each item), not the box diagonal.
- **Fingers** are measured from the knuckle to the tip; the thumb is measured from its base joint. Each length is labelled at the fingertip and listed in the side panel.

**Keeps the camera running.** From Start to Stop the feed stays live. Model switches and class edits load in the background, and the camera reconnects on its own if it drops.

## Measuring in centimetres

A camera only sees pixels, so the app picks a scale in this order:

1. **Calibrated.** Click **Calibrate…**, click both ends of something of known length (a ruler or a credit card, which is 8.56 cm) held where you'll be measuring, then type the real length. This is the most accurate option.
2. **Hand estimate.** With no calibration, the app uses MediaPipe's estimate of the visible hand's real size. It's roughly right for an average adult hand, and is shown as "est." in the panel.
3. **Pixels.** With no calibration and no hand in view, lengths are shown in px.

Scales are only accurate for objects at the **same distance from the camera** as the reference. Fingers pointing toward the camera look shorter than they are.

## Models

| Option | Weights | Detects | Speed (M-series Mac) |
|---|---|---|---|
| **Open vocabulary — accurate** (default) | `yolov8l-worldv2.pt` | Anything in the class list | ~33 FPS model only |
| Open vocabulary — fast | `yolov8s-worldv2.pt` | Anything in the class list | ~80 FPS model only |
| COCO fast | `yolo11n.pt` | knife, scissors, fork only | fastest |

The whole app, including hand tracking and drawing, runs at about 10–25 FPS.

## Controls

| Control | Action |
|---|---|
| `space` | Start / stop |
| `s` | Save the annotated frame to `snapshots/` |
| Confidence slider | Applies live. Raise it to cut false alarms |
| Input size | 960 or 1280 helps with small tips (pen nibs, pins) but runs slower |
| Sharp classes → **Apply** | Use your edited list without stopping the camera |
| **Calibrate…** / `Esc` | Start / cancel calibration |

## Limitations

- **Labels are often wrong even when detection is right.** In testing, a fountain pen and a thumbtack were both flagged as sharp but labelled "scissors", and a stylus was labelled "knife". Treat the label as a hint and the warning as the signal.
- **There are false alarms.** A porcelain soup spoon was flagged as "scissors". Raise the confidence threshold, or remove classes you don't need.
- **Small and thin points** (pen nibs, needles, razor blades) are hard to see at webcam resolution. Bring them closer or raise the input size.
- **Hand tracking is built for real hands.** It handles live webcam hands well but can miss unusual ones; a marble statue's hand stopped being detected after video compression.
- **Lengths are 2D estimates.** Items or fingers tilted toward the camera measure short, and the "near hand" check is a 2D overlap test, not true depth.
- **This is a demo, not a safety system.** Don't rely on it to protect anyone. For dependable results, train a YOLO model on labelled photos of your own objects.

## Files

```
app.py            Tk UI, camera worker, YOLO detection, drawing
hands.py          MediaPipe hand + finger tracking and finger lengths
measure.py        Item length along its axis, px → cm formatting
pyproject.toml    Dependencies (uv)
docs/screenshot.png
models/           Hand model, downloaded on first run
snapshots/        Created on first snapshot
```
