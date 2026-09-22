"""
Part 2 – Fast Safety-Critical Loop
YOLO detection → pinhole distance → TTC → hazard decision
Runs on every fresh frame. Never does heavy work (face, OCR, etc.).
"""

from __future__ import annotations
import time
from typing import List, Optional, Tuple
import numpy as np
from ultralytics import YOLO

from utils.geometry import SimpleTracker, TrackState


# Classes we care about for obstacle / navigation alerts
RELEVANT_CLASSES = {
    "person", "chair", "couch", "bed", "dining table", "toilet",
    "tv", "laptop", "backpack", "handbag", "suitcase",
    "car", "motorcycle", "bus", "truck", "bicycle",
    "dog", "cat", "bottle",
}


class SafetyLoop:
    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        conf_thresh: float = 0.35,
        focal_length_px: float = 500.0,
        device: str = "",  # "" = auto (CUDA if available)
    ):
        print(f"[SafetyLoop] loading {model_name} ...")
        self.model = YOLO(model_name)
        # Force a warm-up inference so first real frame is fast
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        self.model.predict(dummy, verbose=False, conf=0.5)
        print("[SafetyLoop] model ready")

        self.conf_thresh = conf_thresh
        self.tracker = SimpleTracker(focal_length_px=focal_length_px)
        self.device = device
        self.last_infer_ms = 0.0

    def process(self, frame: np.ndarray, timestamp: float) -> Tuple[List[TrackState], List[dict]]:
        """
        Returns:
            tracks : list of TrackState (with distance & TTC filled)
            hazards: list of hazard dicts ready for the audio arbiter
        """
        t0 = time.perf_counter()
        results = self.model.predict(
            frame,
            conf=self.conf_thresh,
            verbose=False,
            device=self.device or None,
        )
        self.last_infer_ms = (time.perf_counter() - t0) * 1000.0

        detections = []
        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = self.model.names[cls_id]
                    if name not in RELEVANT_CLASSES:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append((name, conf, (x1, y1, x2, y2)))

        tracks = self.tracker.update(detections, timestamp)
        hazards = self._evaluate_hazards(tracks)
        return tracks, hazards

    def _evaluate_hazards(self, tracks: List[TrackState]) -> List[dict]:
        """
        Priority rules (from architecture review + slides):
          Ch1 immediate : TTC < 1.5 s
          Urgent        : TTC < 2.0 s AND distance < 2 m
          Nav cue       : 1.5 m < distance < 4 m
        """
        hazards = []
        for t in tracks:
            if t.distance_m is None:
                continue

            priority = None
            msg = None

            if t.ttc_s is not None and t.ttc_s < 1.5:
                priority = 1  # Channel 1 – collision
                msg = f"Collision {t.class_name}"
            elif t.ttc_s is not None and t.ttc_s < 2.0 and t.distance_m < 2.0:
                priority = 1
                msg = f"Urgent {t.class_name} {t.distance_m:.1f}m"
            elif 1.5 < t.distance_m < 4.0:
                priority = 2  # Channel 2 – navigation
                msg = f"{t.class_name} ahead, {t.distance_m:.1f} meters"

            if priority is not None:
                # Simple left/right pan from bbox centre
                cx = (t.bbox[0] + t.bbox[2]) / 2.0
                # Assume frame width ~640 for pan; real width is handled in arbiter
                pan = (cx / 640.0) * 2.0 - 1.0  # -1 left … +1 right
                hazards.append({
                    "priority": priority,
                    "track_id": t.track_id,
                    "class_name": t.class_name,
                    "distance_m": t.distance_m,
                    "ttc_s": t.ttc_s,
                    "message": msg,
                    "pan": max(-1.0, min(1.0, pan)),
                    "bbox": t.bbox,
                })
        return hazards
