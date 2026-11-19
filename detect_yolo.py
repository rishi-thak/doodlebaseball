"""YOLO ball detector. Weights: $BALL_WEIGHTS or weights/ball.pt. Device: $BALL_DEVICE (default mps)."""
import os

from ultralytics import YOLO


CX0, CY0, C = 320, 140, 320  # crop: pitch corridor, ball path stays inside it
_model = YOLO(os.environ.get("BALL_WEIGHTS", "weights/ball.pt"))
_device = os.environ.get("BALL_DEVICE", "mps")


def detect(frame):
    """Return [(x, y, r, conf)] in full-frame pixels, best first."""
    res = _model.predict(frame[CY0:CY0 + C, CX0:CX0 + C], imgsz=C, conf=0.25,
                         device=_device, verbose=False)[0]
    out = [((x1 + x2) / 2 + CX0, (y1 + y2) / 2 + CY0, (x2 - x1 + y2 - y1) / 4, conf)
           for (x1, y1, x2, y2), conf in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist())]
    return sorted(out, key=lambda d: -d[3])
