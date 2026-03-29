import cv2
import mediapipe as mp
import numpy as np
import time
import urllib.request
import os

# ── Download model ───────────────────────────────────────────
MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmarker model (~25 MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.")

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

options = mp_vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=mp_vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.65,
    min_hand_presence_confidence=0.65,
    min_tracking_confidence=0.55,
)
detector = mp_vision.HandLandmarker.create_from_options(options)

# ── Window / canvas ──────────────────────────────────────────
CAM_W, CAM_H = 1280, 720
cv2.namedWindow("Finger Draw", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Finger Draw", CAM_W, CAM_H)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
canvas = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)

# ── Palette ──────────────────────────────────────────────────
COLOR_LIST  = [(0,0,255),(0,220,80),(255,120,0),(0,200,220),(180,0,220),(255,255,255)]
COLOR_NAMES = ["RED","GREEN","BLUE","YELLOW","PURPLE","WHITE"]
color_idx   = 0
cur_color   = COLOR_LIST[0]
BRUSH       = 7
ERASER_R    = 45

# ── State ────────────────────────────────────────────────────
prev_x, prev_y      = None, None
mode                = "draw"      # "draw" | "erase" | "idle"
gesture_label       = ""
label_timer         = 0
save_done           = False
save_timer          = 0
# Debounce: only trigger a one-shot gesture once per new raise
last_gesture        = "idle"
gesture_hold_start  = 0.0
HOLD_SECS           = 0.35        # must hold color/clear/save for 0.35s to fire

# ── Landmark ids ─────────────────────────────────────────────
WRIST=0; THUMB_TIP=4; THUMB_IP=3
INDEX_TIP=8;  INDEX_MCP=5
MIDDLE_TIP=12; MIDDLE_MCP=9
RING_TIP=16;  RING_MCP=13
PINKY_TIP=20; PINKY_MCP=17

def finger_up(lm, tip, mcp):
    return lm[tip].y < lm[mcp].y

def ldist(lm, a, b):
    return ((lm[a].x-lm[b].x)**2 + (lm[a].y-lm[b].y)**2) ** 0.5

# ── Gesture detection ────────────────────────────────────────
def detect_gesture(lm):
    idx  = finger_up(lm, INDEX_TIP,  INDEX_MCP)
    mid  = finger_up(lm, MIDDLE_TIP, MIDDLE_MCP)
    ring = finger_up(lm, RING_TIP,   RING_MCP)
    pink = finger_up(lm, PINKY_TIP,  PINKY_MCP)
    thumb_up   = lm[THUMB_TIP].y < lm[THUMB_IP].y - 0.04
    thumb_down = lm[THUMB_TIP].y > lm[WRIST].y + 0.02

    # ✊ Fist — all fingers curled (lift pen)
    if not idx and not mid and not ring and not pink:
        if not thumb_up:
            return "fist"

    # 🖐 Open palm — all 4 fingers up → erase
    if idx and mid and ring and pink:
        return "erase"

    # 🤘 Devil horns — index + pinky up, middle + ring down → color
    if idx and not mid and not ring and pink:
        return "color"

    # ✌️ Peace — index + middle up, ring + pinky down → clear
    if idx and mid and not ring and not pink:
        return "clear"

    # 👍 Thumb up — thumb raised, all fingers curled → save
    if thumb_up and not idx and not mid and not ring and not pink:
        return "save"

    # ☝️ Index only — draw
    if idx and not mid and not ring and not pink:
        return "draw"

    return "idle"

# ── Draw skeleton ─────────────────────────────────────────────
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17)
]
def draw_skeleton(frame, lm, w, h):
    pts = [(int(l.x*w), int(l.y*h)) for l in lm]
    for a,b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (200,200,200), 1, cv2.LINE_AA)
    for pt in pts:
        cv2.circle(frame, pt, 3, (0,255,136), -1)

# ── Palette bar ───────────────────────────────────────────────
def draw_palette(frame):
    for i, col in enumerate(COLOR_LIST):
        x1 = 10 + i*46
        cv2.rectangle(frame, (x1, 10), (x1+38, 52), col, -1)
        cv2.rectangle(frame, (x1, 10), (x1+38, 52), (60,60,60), 1)
        if i == color_idx:
            cv2.rectangle(frame, (x1-3,7), (x1+41,55), (255,255,255), 2)
            cv2.putText(frame, COLOR_NAMES[i], (x1, 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,200), 1)

# ── HUD ──────────────────────────────────────────────────────
GESTURE_GUIDE = [
    ("DRAW",  "index finger only"),
    ("PAUSE", "fist"),
    ("ERASE", "open palm"),
    ("COLOR", "devil horns"),
    ("CLEAR", "peace sign"),
    ("SAVE",  "thumbs up"),
]
def draw_hud(frame):
    fh, fw = frame.shape[:2]

    # Bottom bar
    ov = frame.copy()
    cv2.rectangle(ov, (0, fh-48), (fw, fh), (15,15,15), -1)
    cv2.addWeighted(ov, 0.7, frame, 0.3, 0, frame)
    mtext = f"MODE: {mode.upper()}  |  COLOR: {COLOR_NAMES[color_idx]}  |  Q=quit  S=save  C=clear"
    cv2.putText(frame, mtext, (12, fh-16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (190,190,190), 1, cv2.LINE_AA)

    # Right-side guide panel
    px, py = fw-190, 80
    cv2.rectangle(frame, (px-8, py-20), (fw-8, py + len(GESTURE_GUIDE)*22 + 8),
                  (20,20,20), -1)
    cv2.rectangle(frame, (px-8, py-20), (fw-8, py + len(GESTURE_GUIDE)*22 + 8),
                  (60,60,60), 1)
    cv2.putText(frame, "GESTURES", (px, py-4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140,140,140), 1)
    for i,(action, desc) in enumerate(GESTURE_GUIDE):
        col = (0,255,136) if (action.lower()==mode or
               (action=="DRAW" and mode=="draw")) else (140,140,140)
        cv2.putText(frame, f"{action}: {desc}", (px, py+18+i*22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)

    # Brush size dot
    cv2.circle(frame, (fw-30, fh-70),
               ERASER_R//3 if mode=="erase" else BRUSH+2,
               (80,80,80) if mode=="erase" else cur_color, -1)

    # Flash label (big center text)
    if gesture_label and time.time()-label_timer < 1.4:
        alpha = max(0, 1.4-(time.time()-label_timer)) / 1.4
        txt_col = tuple(int(c*alpha) for c in (255,255,255))
        cv2.putText(frame, gesture_label, (fw//2-160, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, txt_col, 2, cv2.LINE_AA)

    # Save confirmation
    if save_done and time.time()-save_timer < 2.5:
        cv2.putText(frame, "SAVED: drawing.png", (fw//2-160, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,136), 2, cv2.LINE_AA)

# ── Hold-gesture debounce ─────────────────────────────────────
# Returns True the FIRST time gesture has been held >= HOLD_SECS
hold_state   = {"gesture": "", "start": 0.0, "fired": False}

def held_long_enough(g):
    """Returns True exactly once per new hold of gesture g."""
    if hold_state["gesture"] != g:
        hold_state["gesture"] = g
        hold_state["start"]   = time.time()
        hold_state["fired"]   = False
        return False
    if not hold_state["fired"] and time.time()-hold_state["start"] >= HOLD_SECS:
        hold_state["fired"] = True
        return True
    return False

# ── Main loop ─────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame  = cv2.flip(frame, 1)
    fh, fw = frame.shape[:2]
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_img)

    gesture = "idle"

    if result.hand_landmarks:
        lm = result.hand_landmarks[0]
        draw_skeleton(frame, lm, fw, fh)
        gesture = detect_gesture(lm)

        ix = int(lm[INDEX_TIP].x * fw)
        iy = int(lm[INDEX_TIP].y * fh)

        # ── DRAW ─────────────────────────────────────────────
        if gesture == "draw":
            mode = "draw"
            if prev_x is not None:
                cv2.line(canvas, (prev_x,prev_y), (ix,iy), cur_color, BRUSH)
            prev_x, prev_y = ix, iy
            cv2.circle(frame, (ix,iy), BRUSH+2, cur_color, -1)

        # ── FIST = lift pen ───────────────────────────────────
        elif gesture == "fist":
            mode = "idle"
            prev_x, prev_y = None, None

        # ── ERASE ─────────────────────────────────────────────
        elif gesture == "erase":
            mode = "erase"
            cv2.circle(canvas, (ix,iy), ERASER_R, (0,0,0), -1)
            cv2.circle(frame,  (ix,iy), ERASER_R, (70,70,70), 2)
            prev_x, prev_y = None, None

        # ── COLOR (hold 0.35s to cycle) ───────────────────────
        elif gesture == "color":
            prev_x, prev_y = None, None
            if held_long_enough("color"):
                color_idx     = (color_idx+1) % len(COLOR_LIST)
                cur_color     = COLOR_LIST[color_idx]
                gesture_label = f"COLOR: {COLOR_NAMES[color_idx]}"
                label_timer   = time.time()
                # reset so next hold cycles again
                hold_state["fired"] = False
                hold_state["start"] = time.time() + 0.5   # short cooldown

        # ── CLEAR (hold 0.35s) ────────────────────────────────
        elif gesture == "clear":
            prev_x, prev_y = None, None
            if held_long_enough("clear"):
                canvas[:]     = 0
                gesture_label = "CANVAS CLEARED"
                label_timer   = time.time()
                save_done     = False

        # ── SAVE (hold 0.35s) ─────────────────────────────────
        elif gesture == "save":
            prev_x, prev_y = None, None
            if held_long_enough("save") and not save_done:
                out = frame.copy()
                mask = canvas.astype(bool).any(axis=2)
                out[mask] = cv2.addWeighted(frame, 0.15, canvas, 0.85, 0)[mask]
                cv2.imwrite("drawing.png", out)
                save_done     = True
                save_timer    = time.time()
                gesture_label = "SAVED!"
                label_timer   = time.time()

        else:
            prev_x, prev_y = None, None

        # Reset hold state when gesture changes
        if gesture not in ("color","clear","save"):
            if hold_state["gesture"] in ("color","clear","save"):
                hold_state["gesture"] = ""

    else:
        prev_x, prev_y = None, None
        hold_state["gesture"] = ""

    # Composite canvas
    mask = canvas.astype(bool).any(axis=2)
    frame[mask] = cv2.addWeighted(frame, 0.15, canvas, 0.85, 0)[mask]

    draw_palette(frame)
    draw_hud(frame)
    cv2.imshow("Finger Draw", frame)

    key = cv2.waitKey(1) & 0xFF
    if   key == ord('q'): break
    elif key == ord('s'):
        cv2.imwrite("drawing.png", frame)
        print("Saved drawing.png")
    elif key == ord('c'):
        canvas[:] = 0
        print("Canvas cleared")

cap.release()
cv2.destroyAllWindows()