# Finger Draw

A real-time air drawing app controlled entirely by hand gestures via webcam. Built as a practice project to learn computer vision and MediaPipe hand tracking — not perfect, but functional and a good foundation.

---

## What it does

Uses your webcam to detect hand landmarks in real time. Point your index finger to draw on screen, switch colors with devil horns, erase with an open palm, and more — no mouse or keyboard needed.

---

## Gestures

| Gesture | Action |
|---|---|
| ☝️ Index finger only | Draw |
| ✊ Fist | Lift pen / pause |
| 🖐️ Open palm | Erase |
| 🤘 Devil horns (hold) | Cycle color |
| ✌️ Peace sign (hold) | Clear canvas |
| 👍 Thumb up (hold) | Save as drawing.png |
| `S` key | Save (keyboard shortcut) |
| `C` key | Clear (keyboard shortcut) |
| `Q` key | Quit |

Color, clear, and save gestures require holding for ~0.35 seconds to avoid accidental triggers.

---

## Known flaws

- Gesture detection isn't perfectly reliable — lighting and hand angle affect accuracy
- Drawing can feel slightly laggy depending on your machine
- Color cycling can occasionally misfire if your hand is at an awkward angle
- Only one hand supported at a time
- No undo

This was built for learning purposes and intentionally kept simple. It works well enough to demonstrate the concept.

---

## Stack

- Python 3.11
- MediaPipe 0.10.x (Tasks API)
- OpenCV
- NumPy

---

## Setup

**Requires Python 3.11** — MediaPipe does not yet support Python 3.12+.

```bash
pip install opencv-python mediapipe numpy
```

On first run, the app will automatically download the MediaPipe hand landmark model (~25 MB) into your project folder.

```bash
python finger_draw.py
```

---

## Notes

- Works best in good lighting with your hand clearly visible
- Keep your hand roughly 40–60 cm from the webcam for best detection
- The model file `hand_landmarker.task` will appear in your folder after first run — do not delete it

---

*Practice project*
