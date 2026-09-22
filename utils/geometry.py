"""
Monocular pinhole geometry + Time-to-Collision helpers.
Matches the formulas on slides 6–7 of the Assistive AI Smart Glasses deck.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np


# Approximate real-world heights (metres) for common COCO / custom classes.
# Extend this table as you add new classes.
CLASS_HEIGHTS_M: Dict[str, float] = {
    "person": 1.70,
    "chair": 0.85,
    "couch": 0.80,
    "bed": 0.60,
    "dining table": 0.75,
    "toilet": 0.70,
    "tv": 0.55,
    "laptop": 0.25,
    "backpack": 0.45,
    "handbag": 0.30,
    "suitcase": 0.55,
    "bottle": 0.25,
    "cup": 0.12,
    "car": 1.50,
    "motorcycle": 1.20,
    "bus": 3.00,
    "truck": 3.20,
    "bicycle": 1.10,
    "dog": 0.50,
    "cat": 0.30,
}


@dataclass
class TrackState:
    """Simple per-object state for smoothing and TTC."""
    track_id: int
    class_name: str
    bbox: Tuple[float, float, float, float]  # x1,y1,x2,y2
    area: float = 0.0
    smoothed_area: float = 0.0
    prev_smoothed_area: float = 0.0
    distance_m: Optional[float] = None
    ttc_s: Optional[float] = None
    last_seen: float = 0.0
    frames_since_update: int = 0


def pinhole_distance(
    bbox_h_px: float,
    class_name: str,
    focal_length_px: float = 500.0,
) -> Optional[float]:
    """
    Distance = (f * H_real) / h_pixel

    focal_length_px : calibrated focal length of the camera in pixels.
                      For phone cameras at QVGA/VGA a value around 400–700 is typical.
                      Calibrate once with a known object at a known distance.
    """
    h_real = CLASS_HEIGHTS_M.get(class_name.lower())
    if h_real is None or bbox_h_px < 2.0:
        return None
    return (focal_length_px * h_real) / bbox_h_px


def bbox_area(bbox: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, (x2 - x1) * (y2 - y1))


def estimate_ttc(
    current_area: float,
    prev_area: float,
    dt: float,
    min_area: float = 100.0,
) -> Optional[float]:
    """
    TTC ≈ A(t) / (dA/dt)

    Positive TTC means the object is approaching.
    Returns None if the object is receding or data is unreliable.
    """
    if dt <= 1e-3 or current_area < min_area or prev_area < min_area:
        return None
    dA = current_area - prev_area
    if dA <= 0:  # not growing → not approaching
        return None
    ttc = current_area / (dA / dt)
    # Clamp to a sensible range for navigation
    if ttc < 0.1 or ttc > 30.0:
        return None
    return ttc


def iou(box_a: Tuple[float, float, float, float],
        box_b: Tuple[float, float, float, float]) -> float:
    """Intersection-over-Union for simple greedy tracking."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class SimpleTracker:
    """
    Lightweight IOU tracker + exponential smoothing of area for stable TTC.
    Keeps the safety loop under a few milliseconds.
    """

    def __init__(
        self,
        iou_thresh: float = 0.3,
        max_age: int = 15,
        smooth_alpha: float = 0.4,
        focal_length_px: float = 500.0,
    ):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.smooth_alpha = smooth_alpha
        self.focal_length_px = focal_length_px
        self.tracks: Dict[int, TrackState] = {}
        self._next_id = 1

    def update(
        self,
        detections: list,  # list of (class_name, conf, bbox)
        timestamp: float,
    ) -> list[TrackState]:
        """
        detections: [(class_name, confidence, (x1,y1,x2,y2)), ...]
        Returns list of active TrackState objects.
        """
        # Age existing tracks
        for t in self.tracks.values():
            t.frames_since_update += 1

        unmatched_dets = list(range(len(detections)))
        matched_track_ids = set()

        # Greedy IOU matching
        for tid, track in list(self.tracks.items()):
            best_iou = 0.0
            best_idx = -1
            for i in unmatched_dets:
                cls, conf, bbox = detections[i]
                if cls != track.class_name:
                    continue
                score = iou(track.bbox, bbox)
                if score > best_iou:
                    best_iou = score
                    best_idx = i
            if best_iou >= self.iou_thresh and best_idx >= 0:
                cls, conf, bbox = detections[best_idx]
                area = bbox_area(bbox)
                prev_smooth = track.smoothed_area
                smooth = (
                    self.smooth_alpha * area
                    + (1.0 - self.smooth_alpha) * track.smoothed_area
                    if track.smoothed_area > 0
                    else area
                )
                h_px = max(1.0, bbox[3] - bbox[1])
                dist = pinhole_distance(h_px, cls, self.focal_length_px)
                dt = max(1e-3, timestamp - track.last_seen)
                ttc = estimate_ttc(smooth, prev_smooth, dt)

                track.bbox = bbox
                track.area = area
                track.prev_smoothed_area = prev_smooth
                track.smoothed_area = smooth
                track.distance_m = dist
                track.ttc_s = ttc
                track.last_seen = timestamp
                track.frames_since_update = 0
                matched_track_ids.add(tid)
                unmatched_dets.remove(best_idx)

        # Create new tracks for unmatched detections
        for i in unmatched_dets:
            cls, conf, bbox = detections[i]
            area = bbox_area(bbox)
            h_px = max(1.0, bbox[3] - bbox[1])
            dist = pinhole_distance(h_px, cls, self.focal_length_px)
            tid = self._next_id
            self._next_id += 1
            self.tracks[tid] = TrackState(
                track_id=tid,
                class_name=cls,
                bbox=bbox,
                area=area,
                smoothed_area=area,
                prev_smoothed_area=area,
                distance_m=dist,
                ttc_s=None,
                last_seen=timestamp,
                frames_since_update=0,
            )
            matched_track_ids.add(tid)

        # Drop stale tracks
        to_delete = [
            tid for tid, t in self.tracks.items()
            if t.frames_since_update > self.max_age
        ]
        for tid in to_delete:
            del self.tracks[tid]

        return list(self.tracks.values())
