#!/usr/bin/env python3
"""
Assistive AI Smart Glasses – Laptop Prototype
=============================================
Phone camera (IP Webcam / DroidCam / similar)  →  Laptop runs the full pipeline.

Parts implemented:
  1. Drop-oldest frame buffer + blur gate
  2. Safety-critical loop (YOLO + pinhole distance + TTC)
  3. Throttled context loop (face detection)
  4. Audio arbitration (Ch1 preemption)

Usage examples:
  python main.py                          # laptop webcam
  python main.py --source 0
  python main.py --source http://192.168.1.5:8080/video
  python main.py --source http://192.168.1.5:4747/video   # DroidCam
"""

from __future__ import annotations
import argparse
import time
import sys
import cv2
import numpy as np

from core.frame_buffer import DropOldestFrameBuffer
from core.safety_loop import SafetyLoop
from core.context_loop import ContextLoop
from core.audio_arbiter import AudioArbiter


def draw_overlay(frame, tracks, hazards, fps, infer_ms, buffer_stats, context_msg):
    """Draw bounding boxes, distance, TTC and status text."""
    for t in tracks:
        x1, y1, x2, y2 = map(int, t.bbox)
        color = (0, 255, 0)
        if t.ttc_s is not None and t.ttc_s < 1.5:
            color = (0, 0, 255)          # red – collision
        elif t.ttc_s is not None and t.ttc_s < 2.0:
            color = (0, 165, 255)        # orange – urgent
        elif t.distance_m is not None and t.distance_m < 4.0:
            color = (0, 255, 255)        # yellow – nav

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = t.class_name
        if t.distance_m is not None:
            label += f" {t.distance_m:.1f}m"
        if t.ttc_s is not None:
            label += f" TTC={t.ttc_s:.1f}s"
        cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # Status bar
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 78), (0, 0, 0), -1)
    cv2.putText(frame, f"FPS {fps:.1f}  |  Infer {infer_ms:.0f}ms  |  {buffer_stats}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    cv2.putText(frame, "Ch1=Collision (red)  Ch2=Nav (yellow)  q=quit",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    if context_msg:
        cv2.putText(frame, f"Context: {context_msg}",
                    (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 255), 1)
    return frame


def parse_args():
    p = argparse.ArgumentParser(description="Assistive AI Smart Glasses – laptop prototype")
    p.add_argument(
        "--source",
        default="0",
        help="Camera source: 0 (laptop webcam) or http://IP:PORT/video (phone IP Webcam / DroidCam)",
    )
    p.add_argument("--width", type=int, default=640, help="Max processing width")
    p.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold")
    p.add_argument("--focal", type=float, default=500.0,
                   help="Focal length in pixels (calibrate for your phone)")
    p.add_argument("--model", default="yolov8n.pt", help="Ultralytics model name or path")
    p.add_argument("--no-blur-gate", action="store_true", help="Disable blur skipping")
    return p.parse_args()


def main():
    args = parse_args()

    # Convert numeric string to int for OpenCV
    source = int(args.source) if args.source.isdigit() else args.source

    print("=" * 60)
    print("  Assistive AI Smart Glasses – Laptop Prototype")
    print("  Phone camera → Laptop (drop-oldest + dual-loop + TTC)")
    print("=" * 60)
    print(f"Source     : {source}")
    print(f"Model      : {args.model}")
    print(f"Focal (px) : {args.focal}")
    print()

    # ── Part 1: Frame buffer ──────────────────────────────────────────────
    buffer = DropOldestFrameBuffer(
        source=source,
        max_width=args.width,
        blur_threshold=35.0,
        target_fps=18.0,
    )
    if not buffer.start():
        print("Failed to open camera source. Check the URL / index.")
        print("\nPhone camera tips:")
        print("  1. Install 'IP Webcam' (Android) or 'DroidCam'")
        print("  2. Start the server and note the http://IP:PORT address")
        print("  3. Run:  python main.py --source http://YOUR_IP:8080/video")
        sys.exit(1)

    # ── Part 2: Safety loop ───────────────────────────────────────────────
    safety = SafetyLoop(
        model_name=args.model,
        conf_thresh=args.conf,
        focal_length_px=args.focal,
    )

    # ── Part 3: Context loop ──────────────────────────────────────────────
    context = ContextLoop(every_n_frames=10)

    # ── Part 4: Audio arbiter ─────────────────────────────────────────────
    audio = AudioArbiter(debounce_s=1.5)

    print("\nRunning… press 'q' in the video window to quit.\n")

    fps_smooth = 0.0
    last_t = time.perf_counter()
    frame_idx = 0

    try:
        while True:
            packet = buffer.get_fresh_frame(
                timeout=0.8,
                require_sharp=not args.no_blur_gate,
            )
            if packet is None:
                # No sharp frame available right now – keep UI alive
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                continue

            frame, ts = packet
            frame_idx += 1

            # Safety-critical path (every frame)
            tracks, hazards = safety.process(frame, ts)

            # Submit hazards to audio
            for h in hazards:
                audio.submit(h)

            # Context loop (throttled)
            has_person = any(t.class_name == "person" for t in tracks)
            if context.should_run(has_person):
                context.process_async(frame)

            ctx = context.get_latest()
            ctx_msg = None
            if ctx and ctx.get("message"):
                ctx_msg = ctx["message"]
                # Optional: also announce via Ch3 (commented to avoid chatter)
                # audio.submit_context(ctx_msg)

            # FPS
            now = time.perf_counter()
            dt = now - last_t
            last_t = now
            inst_fps = 1.0 / dt if dt > 0 else 0.0
            fps_smooth = 0.85 * fps_smooth + 0.15 * inst_fps

            # Overlay + display
            vis = draw_overlay(
                frame.copy(),
                tracks,
                hazards,
                fps_smooth,
                safety.last_infer_ms,
                buffer.stats(),
                ctx_msg,
            )
            cv2.imshow("Assistive AI Smart Glasses – Laptop Prototype", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                # Quick screenshot
                name = f"capture_{int(time.time())}.jpg"
                cv2.imwrite(name, vis)
                print(f"Saved {name}")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        buffer.stop()
        audio.stop()
        cv2.destroyAllWindows()
        print("Clean exit.")


if __name__ == "__main__":
    main()
