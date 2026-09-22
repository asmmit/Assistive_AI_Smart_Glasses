"""
Part 4 – Audio Arbitration & Priority Output Engine

Three channels:
  Ch1 – Immediate collision earcons (highest priority, can preempt everything)
  Ch2 – Short navigation voice tokens
  Ch3 – On-demand face / scene info

Debounce per track_id so the same object does not spam.

Audio backends (tried in order):
  1. pygame          (if installed)
  2. winsound        (Windows built-in – works on Python 3.14)
  3. console print   (always works)
"""

from __future__ import annotations
import time
import threading
import queue
import sys
from typing import Optional, Dict, Any

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------
PYGAME_OK = False
WINSOUND_OK = False

try:
    import pygame
    pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
    PYGAME_OK = True
except Exception:
    pass

if not PYGAME_OK and sys.platform == "win32":
    try:
        import winsound
        WINSOUND_OK = True
    except Exception:
        pass


def _make_beep_array(freq: float = 880.0, duration: float = 0.12, volume: float = 0.6, pan: float = 0.0):
    """Only used when pygame is available."""
    import numpy as np
    sample_rate = 22050
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = np.sin(freq * 2 * np.pi * t) * volume
    envelope = np.ones_like(wave)
    fade = max(1, int(0.01 * sample_rate))
    envelope[:fade] = np.linspace(0, 1, fade)
    envelope[-fade:] = np.linspace(1, 0, fade)
    wave *= envelope
    left = wave * (1.0 - max(0.0, pan))
    right = wave * (1.0 + min(0.0, pan))
    stereo = np.column_stack((left, right))
    return (stereo * 32767).astype(np.int16)


class AudioArbiter:
    def __init__(self, debounce_s: float = 1.6):
        self.debounce_s = debounce_s
        self._last_spoken: Dict[int, float] = {}
        self._lock = threading.Lock()
        self._current_priority = 99
        self._stop = threading.Event()
        self._q: queue.PriorityQueue = queue.PriorityQueue()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

        self.earcons = {}
        if PYGAME_OK:
            try:
                self.earcons = {
                    "collision": _make_beep_array(1200, 0.15, 0.8),
                    "urgent": _make_beep_array(900, 0.12, 0.7),
                    "nav": _make_beep_array(600, 0.08, 0.45),
                }
            except Exception:
                pass

        backend = "pygame" if PYGAME_OK else ("winsound" if WINSOUND_OK else "console")
        print(f"[AudioArbiter] ready (backend={backend})")

    def submit(self, hazard: dict):
        priority = hazard.get("priority", 3)
        track_id = hazard.get("track_id", -1)
        now = time.perf_counter()

        with self._lock:
            last = self._last_spoken.get(track_id, 0.0)
            if now - last < self.debounce_s and priority > 1:
                return
            if priority == 1 or now - last >= self.debounce_s:
                self._last_spoken[track_id] = now

        self._q.put((priority, now, hazard))

    def submit_context(self, message: str):
        self._q.put((3, time.perf_counter(), {
            "priority": 3,
            "message": message,
            "pan": 0.0,
            "track_id": -999,
        }))

    def _play(self, priority: int, pan: float = 0.0):
        if PYGAME_OK and self.earcons:
            try:
                import numpy as np
                key = "collision" if priority == 1 else "nav"
                sound = self.earcons.get(key, self.earcons.get("nav"))
                if sound is None:
                    return
                left = sound[:, 0] * (1.0 - max(0.0, pan))
                right = sound[:, 1] * (1.0 + min(0.0, pan))
                stereo = np.column_stack((left, right)).astype(np.int16)
                snd = pygame.sndarray.make_sound(stereo)
                snd.play()
                time.sleep(0.12 if priority == 1 else 0.08)
                return
            except Exception as e:
                print(f"[AudioArbiter] pygame play error: {e}")

        if WINSOUND_OK:
            try:
                # Frequency mapping: higher pitch = more urgent
                freq = 1200 if priority == 1 else (900 if priority == 2 else 600)
                dur = 150 if priority == 1 else 100
                winsound.Beep(freq, dur)
                return
            except Exception as e:
                print(f"[AudioArbiter] winsound error: {e}")

        # Fallback: just the console message (already printed)

    def _worker_loop(self):
        while not self._stop.is_set():
            try:
                priority, ts, hazard = self._q.get(timeout=0.2)
            except queue.Empty:
                continue

            with self._lock:
                if priority > self._current_priority:
                    continue
                self._current_priority = priority

            msg = hazard.get("message", "")
            pan = float(hazard.get("pan", 0.0))
            print(f"[Audio Ch{priority}] {msg}  (pan={pan:+.2f})")

            self._play(priority, pan)

            with self._lock:
                self._current_priority = 99

    def stop(self):
        self._stop.set()
        if self._worker.is_alive():
            self._worker.join(timeout=1.0)
