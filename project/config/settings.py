from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------
# VIDEO / QUEUE
# --------------------------------------------------------------------------
MAX_VIDEOS = 4
BATCH_SIZE = 4
FRAME_SKIP = 1

# Single source of truth for queue size (previously duplicated with
# different values in tkinter_app.py -> caused config drift).
QUEUE_SIZE = 6

DISPLAY_SIZE = (640, 360)

# --------------------------------------------------------------------------
# MODEL / INFERENCE
# --------------------------------------------------------------------------
YOLO_CONF = 0.55
YOLO_IOU = 0.45

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_HALF_PRECISION = DEVICE == "cuda"  # FP16 inference on GPU for speed

MODEL_YOLO = BASE_DIR / "models" / "yolo11n-pose.pt"
MODEL_XGB = BASE_DIR / "models" / "trained_model4.json"

# --------------------------------------------------------------------------
# EVENT / ACCURACY TUNING
# --------------------------------------------------------------------------
EVENT_THRESHOLD = 0.55

# Rolling average window (frames) applied to classifier scores before
# thresholding. Smooths out single-frame classifier noise, reducing
# false-positive "cheating" flags caused by one bad pose read.
SCORE_SMOOTHING_WINDOW = 15

# When a student is NOT currently flagged as cheating, their accumulated
# cheating_time decays at this rate (units per second). Without decay the
# original engine kept students permanently flagged after a single brief
# incident; this makes "attempts" reflect sustained behaviour instead.
CHEATING_TIME_DECAY_PER_SEC = 0.5

# IoU threshold below which a track<->detection match is rejected.
TRACK_IOU_THRESHOLD = 0.45

# Frames a track can go unmatched before being dropped.
TRACK_MAX_MISSING = 5

# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
TABLE_REFRESH_MS = 200
RESULT_PUMP_MS = 30
