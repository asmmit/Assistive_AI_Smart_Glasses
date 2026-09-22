
# Assistive AI Smart Glasses – Laptop Prototype

**Phone camera → Laptop runs the full AI pipeline**

Works on **Windows / macOS / Linux**. Audio uses `winsound` on Windows when pygame is not available (important for Python 3.14).



## 1. What you need

| Item | Purpose |
|------|---------|
| Laptop | Runs all AI + audio |
| Phone (Android/iOS) | Camera only |
| Same Wi-Fi for phone & laptop | Low-latency stream |

**Phone apps**
- Android: IP Webcam or DroidCam
- iOS: EpocCam or any HTTP/MJPEG camera app

---

## 2. Install (Windows PowerShell)

Open PowerShell in the project folder and run:

```powershell
# Create virtual environment
python -m venv venv

# Activate it (WINDOWS syntax – do NOT use "source")
.\venv\Scripts\Activate.ps1

# If you get an execution-policy error, run this once:
# Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# Install packages (pygame is optional and skipped on Python 3.14)
pip install -r requirements.txt
```

**macOS / Linux** activation is different:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

---

## 3. Start the phone camera

### IP Webcam (Android – recommended)
1. Open **IP Webcam** → scroll down → **Start server**
2. Note the address, e.g. `http://192.168.1.42:8080`
3. Stream URL = `http://192.168.1.42:8080/video`

### DroidCam
- Stream URL = `http://IP:4747/video`

Test the URL in a browser first – you should see the live camera.

---

## 4. Run

```powershell
# Make sure the venv is activated (you should see (venv) in the prompt)

# Phone camera – replace with YOUR phone IP
python main.py --source http://192.168.1.42:8080/video

# Laptop webcam (for testing without phone)
python main.py --source 0
```

Useful options:
```powershell
python main.py --source http://192.168.1.42:8080/video --width 480 --conf 0.4 --focal 520
```

| Flag | Meaning | Default |
|------|---------|---------|
| `--source` | `0` or `http://IP:PORT/video` | `0` |
| `--width` | Max width (lower = faster) | 640 |
| `--conf` | YOLO confidence | 0.35 |
| `--focal` | Focal length in pixels (calibrate distance) | 500 |
| `--no-blur-gate` | Disable blur skipping | off |

- Press **q** in the video window to quit  
- Press **s** to save a screenshot  

---

## 5. What you should see / hear

- Coloured boxes  
  - **Red** = collision (TTC < 1.5 s) → high beep (Ch1)  
  - **Orange** = urgent  
  - **Yellow** = navigation cue  
- Top status bar: FPS, inference time, buffer stats  
- Console lines like:  
  `[Audio Ch1] Collision person  (pan=-0.32)`  
- On Windows you will hear system beeps via `winsound` (no pygame needed)

---

## 6. Calibrating distance

```
Distance = (f × H_real) / h_pixel
```

1. Stand a person (~1.70 m) exactly 2 metres from the phone  
2. Run the program and look at the displayed distance  
3. Change `--focal` until the number is ≈ 2.0 m  
   - Distance too high → increase `--focal`  
   - Distance too low → decrease `--focal`  

Typical phone values: 400–700.

---

## 7. Troubleshooting

| Problem | Fix |
|---------|-----|
| `source` is not recognized | You are on Windows. Use `.\venv\Scripts\Activate.ps1` |
| Execution policy error | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| pygame build error | Ignore it. The code uses Windows `winsound` automatically. |
| cannot open source | Phone & laptop on same Wi-Fi? Open the URL in a browser first. |
| Very laggy | Use `--width 480` or `320`. Close other apps on the phone. |
| No sound | Check console for `[Audio Ch…]` lines. On Windows beeps should play. |
| YOLO download slow | First run downloads `yolov8n.pt` (~6 MB). Wait. |

---

## 8. Project structure

```
assistive_ai_smart_glasses/
├── main.py
├── requirements.txt
├── README.md
├── core/
│   ├── frame_buffer.py     # Part 1 – drop-oldest + blur gate
│   ├── safety_loop.py      # Part 2 – YOLO + distance + TTC
│   ├── context_loop.py     # Part 3 – throttled face detection
│   └── audio_arbiter.py    # Part 4 – priority audio (winsound/pygame)
└── utils/
    └── geometry.py         # Pinhole formula + tracker
```

---

**Team:** Asmit · Aryan · Soyam  
Prototype ready for field testing and later port to the Android + ESP32 system.
=======
# Assistive_AI_Smart_Glasses
Real-time assistive vision for smart glasses: drop-oldest camera buffer, YOLOv8 obstacle detection + TTC alerts, throttled face recognition, and priority audio arbitration.
>>>>>>> 91103c153a592083a86fb90acba9c1876c7bc1f7
