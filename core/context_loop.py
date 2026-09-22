"""
Part 3 – Throttled Context Loop (Face / Scene)

Runs on a separate schedule so it never blocks the safety-critical path.
Triggered only every N frames OR when a stable “person” track exists.
"""

from __future__ import annotations
import time
import threading
from typing import Optional, List, Dict, Any
import cv2
import numpy as np

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False
    print("[ContextLoop] mediapipe not installed – face detection disabled")


class ContextLoop:
    def __init__(
        self,
        every_n_frames: int = 8,
        min_person_stable_frames: int = 3,
    ):
        self.every_n_frames = every_n_frames
        self.min_person_stable_frames = min_person_stable_frames
        self._frame_counter = 0
        self._lock = threading.Lock()
        self._latest_result: Optional[Dict[str, Any]] = None
        self._person_stable_count = 0

        if MP_AVAILABLE:
            self.mp_face = mp.solutions.face_detection
            self.face_detector = self.mp_face.FaceDetection(
                model_selection=0,  # short-range
                min_detection_confidence=0.55,
            )
            print("[ContextLoop] MediaPipe Face Detection ready")
        else:
            self.face_detector = None

        # Placeholder local “contacts” – replace with real embeddings later
        self.known_contacts = {
            # "embedding_hash": "Name"
        }

    def should_run(self, has_stable_person: bool) -> bool:
        self._frame_counter += 1
        if has_stable_person:
            self._person_stable_count += 1
        else:
            self._person_stable_count = 0

        if self._frame_counter % self.every_n_frames == 0:
            return True
        if self._person_stable_count >= self.min_person_stable_frames:
            return True
        return False

    def process_async(self, frame: np.ndarray):
        """Fire-and-forget; result is stored for later polling."""
        if self.face_detector is None:
            return
        t = threading.Thread(target=self._run, args=(frame.copy(),), daemon=True)
        t.start()

    def _run(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(rgb)
        faces = []
        if results.detections:
            h, w = frame.shape[:2]
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = int((bbox.xmin + bbox.width) * w)
                y2 = int((bbox.ymin + bbox.height) * h)
                conf = det.score[0] if det.score else 0.0
                # Placeholder: real system would crop, embed, match SQLite
                name = "Unknown person"
                faces.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": conf,
                    "name": name,
                })

        with self._lock:
            self._latest_result = {
                "timestamp": time.perf_counter(),
                "faces": faces,
                "message": faces[0]["name"] if faces else None,
            }

    def get_latest(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._latest_result
