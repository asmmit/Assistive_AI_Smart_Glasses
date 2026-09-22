"""
Part 1 – Video Ingestion & Frame-Freshness Buffer (Drop-Oldest Pipeline)

- Background capture thread continuously grabs the newest frame
- Single-slot queue (maxsize=1) → drop-oldest semantics
- Lightweight Variance-of-Laplacian blur / motion gate
- Works with laptop webcam OR phone camera streamed via IP Webcam / DroidCam
"""

from __future__ import annotations
import threading
import time
import queue
from typing import Optional, Tuple
import cv2
import numpy as np


def variance_of_laplacian(gray: np.ndarray) -> float:
    """Classic focus measure. Low value ≈ blurry or static."""
    return cv2.Laplacian(gray, cv2.CV_64F).var()


class DropOldestFrameBuffer:
    """
    Thread-safe drop-oldest frame buffer.

    Capture thread → put_nowait (overwrites if full)
    Consumer thread → get (always receives the freshest frame)
    """

    def __init__(
        self,
        source: str | int = 0,
        max_width: int = 640,
        blur_threshold: float = 40.0,
        target_fps: float = 20.0,
    ):
        """
        source:
            0          → default laptop webcam
            "http://..." → phone IP Webcam / DroidCam MJPEG or HTTP stream
            path/to/video.mp4 for offline testing
        """
        self.source = source
        self.max_width = max_width
        self.blur_threshold = blur_threshold
        self.target_dt = 1.0 / max(1.0, target_fps)

        self._q: queue.Queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None

        # Stats
        self.frames_captured = 0
        self.frames_dropped = 0
        self.frames_blur_skipped = 0
        self.last_fps = 0.0

    def start(self) -> bool:
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            print(f"[FrameBuffer] ERROR: cannot open source {self.source}")
            return False

        # Prefer lower resolution for lower latency (phone cameras are high-res)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.max_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.max_width * 0.75))
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # ask backend for minimal buffering

        self._stop.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print(f"[FrameBuffer] started – source={self.source}")
        return True

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
        print("[FrameBuffer] stopped")

    def _capture_loop(self):
        last_t = time.perf_counter()
        frame_count = 0
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # Downscale if needed
            h, w = frame.shape[:2]
            if w > self.max_width:
                scale = self.max_width / w
                frame = cv2.resize(frame, (self.max_width, int(h * scale)))

            timestamp = time.perf_counter()
            packet = (frame, timestamp)

            # Drop-oldest: if queue is full, discard the old frame and put the new one
            try:
                self._q.put_nowait(packet)
            except queue.Full:
                try:
                    self._q.get_nowait()
                    self.frames_dropped += 1
                except queue.Empty:
                    pass
                try:
                    self._q.put_nowait(packet)
                except queue.Full:
                    pass

            self.frames_captured += 1
            frame_count += 1
            now = time.perf_counter()
            if now - last_t >= 1.0:
                self.last_fps = frame_count / (now - last_t)
                frame_count = 0
                last_t = now

            # Soft rate limit so we don't hammer a phone stream
            elapsed = time.perf_counter() - timestamp
            sleep = self.target_dt - elapsed
            if sleep > 0:
                time.sleep(sleep)

    def get_fresh_frame(
        self,
        timeout: float = 0.5,
        require_sharp: bool = True,
    ) -> Optional[Tuple[np.ndarray, float]]:
        """
        Returns (frame_bgr, timestamp) or None.
        If require_sharp=True, skips heavily blurred frames (Variance of Laplacian).
        """
        try:
            frame, ts = self._q.get(timeout=timeout)
        except queue.Empty:
            return None

        if require_sharp:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Downsample for speed
            small = cv2.resize(gray, (0, 0), fx=0.25, fy=0.25)
            vol = variance_of_laplacian(small)
            if vol < self.blur_threshold:
                self.frames_blur_skipped += 1
                return None  # skip this frame, wait for next

        return frame, ts

    def stats(self) -> str:
        return (
            f"cap={self.frames_captured} drop={self.frames_dropped} "
            f"blur_skip={self.frames_blur_skipped} fps={self.last_fps:.1f}"
        )
