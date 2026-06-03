# Smart Door Access System

A production-ready smart door lock combining **local facial recognition** with **remote mobile control via Blynk IoT** — built in Python for Raspberry Pi (and testable on Fedora/any Linux).

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        face_unlock_gui.py                           │
│  (Tkinter main thread — orchestrates all subsystems)                │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │  FaceDatabase │  │ FaceRecognizer│  │       BlynkBridge        │  │
│  │ database.py  │  │ recognizer.py │  │    blynk_bridge.py       │  │
│  │              │  │               │  │  (daemon MQTT thread)    │  │
│  │ Pickle store │  │ HOG/CNN model │  │  paho-mqtt → blynk.cloud │  │
│  └──────┬───────┘  └──────┬────────┘  └───────────┬──────────────┘  │
│         │                 │                        │                  │
│  ┌──────▼───────┐  ┌──────▼────────┐  ┌───────────▼──────────────┐  │
│  │  EnrollmentS │  │   DoorLock    │  │      AccessLogger        │  │
│  │  enroller.py │  │  hardware.py  │  │     access_log.py        │  │
│  │ Multi-sample │  │  GPIO / stub  │  │    Rotating JSONL        │  │
│  └──────────────┘  └───────────────┘  └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                               │
                        USB Camera (OpenCV)
                               │
                    ┌──────────▼──────────┐
                    │  Solenoid Door Lock  │
                    │  via GPIO relay      │
                    └─────────────────────┘
```

---

## Project File Map

| File | Purpose |
|------|---------|
| `face_unlock_gui.py` | Main application — Tkinter GUI + system orchestration |
| `config.py` | All tuneable parameters (GPIO pins, tolerances, Blynk config) |
| `database.py` | Thread-safe, file-backed face encoding store |
| `recognizer.py` | Face recognition engine with confirm-frames + cooldown |
| `hardware.py` | GPIO door lock controller with Pi/stub auto-detection |
| `blynk_bridge.py` | Blynk IoT integration via paho-mqtt (TLS) |
| `access_log.py` | Rotating JSONL structured audit logger |
| `enroller.py` | Multi-sample guided face enrollment workflow |
| `enroll_cli.py` | Headless CLI enrollment tool (SSH/no-display Pi) |
| `requirements.txt` | Python dependencies |
| `tests/` | Unit tests (no camera / hardware required) |

---

## Hardware Bill of Materials

| Component | Recommended Model | Notes |
|-----------|------------------|-------|
| Single-board computer | Raspberry Pi 4 / Pi 5 (2 GB+ RAM) | Pi 3 works but is slow for `dlib` |
| USB Webcam | Logitech C270 / C920 | Any OpenCV-compatible USB cam |
| Power supply | Official Pi PSU (5V 3A) | Undersized PSU causes lockups |
| Solenoid door lock | 12V/24V electromagnetic or solenoid bolt | Choose lock voltage to match relay |
| Relay module | 5V single-channel relay (e.g. SunFounder) | Opto-isolated preferred |
| Power for relay | 12V or 24V DC wall adapter | Sized for the lock's current draw |
| MicroSD card | 32 GB+ Class 10 A2 | Faster card = smoother installs |

### GPIO Wiring Diagram

```
Raspberry Pi GPIO Header
  Pin 12 (GPIO 18) ──► Relay IN
  Pin 2  (5V)      ──► Relay VCC
  Pin 6  (GND)     ──► Relay GND

Relay
  COM ──► + terminal of door lock power supply
  NO  ──► + terminal of solenoid lock
  (Lock GND) ──► − terminal of power supply
```

> **Safety**: Always use an opto-isolated relay. Never connect mains voltage directly to the Pi.

---

## Software Installation

### Prerequisites (all platforms)

```bash
# Fedora / RHEL
sudo dnf install cmake gcc gcc-c++ python3-devel python3-pip \
                 boost-devel libX11-devel libXext-devel

# Debian / Raspberry Pi OS
sudo apt update
sudo apt install cmake build-essential python3-dev python3-pip \
                 libboost-all-dev libx11-dev libopenblas-dev
```

### Python environment

```bash
cd /path/to/rasp-face
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> **Raspberry Pi note — dlib compilation**: dlib must be compiled from source.
> Before running `pip install -r requirements.txt` on a Pi, increase swap:
> ```bash
> sudo dphys-swapfile swapoff
> sudo sed -i 's/CONF_SWAPSIZE=.*/CONF_SWAPSIZE=1024/' /etc/dphys-swapfile
> sudo dphys-swapfile setup && sudo dphys-swapfile swapon
> ```
> After install, revert: `sudo sed -i 's/CONF_SWAPSIZE=.*/CONF_SWAPSIZE=100/' /etc/dphys-swapfile`

### Blynk Setup

1. Create a free account at [blynk.cloud](https://blynk.cloud)
2. Create a **New Template** → choose any name, platform = `Raspberry Pi`, connection = `WiFi`
3. Add **Datastreams** (Virtual Pins):

| Virtual Pin | Name | Data Type | Widget |
|------------|------|-----------|--------|
| V1 | Unlock Button | Integer 0–1 | Button (Switch mode) |
| V2 | Lock Status | Integer 0–1 | LED |
| V3 | Access Log | String | Terminal |
| V4 | Last User | String | Label |

4. Create a **Device** from the template → copy the **Auth Token**
5. Set the token in your environment:
   ```bash
   export BLYNK_AUTH_TOKEN="your_token_here"
   # Or add to /etc/environment for permanent system-wide access
   ```

---

## Running the Application

### GUI mode (with display)

```bash
source .venv/bin/activate
export BLYNK_AUTH_TOKEN="your_token"
export ADMIN_PIN="your_secure_pin"    # default is 1234 — CHANGE THIS
python face_unlock_gui.py
```

### Headless CLI enrollment (SSH / no display)

```bash
# List enrolled people
python enroll_cli.py --list

# Enroll a new person (captures 3 frames by default)
python enroll_cli.py --name "Alice Smith"

# Enroll with more samples for better accuracy
python enroll_cli.py --name "Bob Jones" --samples 8

# Remove a person
python enroll_cli.py --remove "Alice Smith"
```

### Running as a systemd service (Raspberry Pi auto-start)

Create `/etc/systemd/system/smart-door.service`:

```ini
[Unit]
Description=Smart Door Access System
After=network.target

[Service]
User=pi
WorkingDirectory=/home/pi/rasp-face
Environment=BLYNK_AUTH_TOKEN=your_token_here
Environment=ADMIN_PIN=your_secure_pin
Environment=DISPLAY=:0
ExecStart=/home/pi/rasp-face/.venv/bin/python face_unlock_gui.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable smart-door
sudo systemctl start smart-door
sudo systemctl status smart-door
```

---

## Face Enrollment Procedure

### GUI Enrollment (recommended)

1. Launch the application
2. Stand 30–60 cm from the camera, face well lit, no backlight
3. Click **"👤 Enroll New Face"**
4. Enter the admin PIN
5. Type the person's name and click OK
6. **Hold still** while the progress bar fills (3 samples averaged)
7. A confirmation dialog appears on success

### CLI Enrollment (headless Pi)

```bash
python enroll_cli.py --name "Alice Smith" --samples 5
```

The script provides real-time feedback and exits 0 on success.

### Best Practices for Enrollment

- Enroll in typical lighting conditions (don't enroll in bright sun if the lock is indoors)
- Enroll with and without glasses if the person wears them occasionally
- Multiple enrollments of the same name add additional encodings (up to 10 per person)
- For security-critical deployments use `--samples 8` or higher

---

## Configuration Reference

Edit `config.py` or set environment variables:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FACE_TOLERANCE` | `0.55` | Match strictness — lower = stricter (0.0–1.0) |
| `FACE_MODEL` | `"hog"` | `"hog"` (fast) or `"cnn"` (accurate, needs GPU) |
| `UNLOCK_DURATION_SEC` | `3` | Seconds before auto-relock |
| `GPIO_LOCK_PIN` | `18` | BCM GPIO pin for relay |
| `LOCK_ACTIVE_HIGH` | `True` | `True` = HIGH unlocks; `False` = LOW unlocks |
| `ADMIN_PIN` | `"1234"` | Admin PIN — **change via `ADMIN_PIN` env var** |
| `BLYNK_AUTH_TOKEN` | — | Required for cloud control |
| `MIN_ENCODINGS_PER_PERSON` | `3` | Samples per enrollment session |

---

## Running Tests

```bash
source .venv/bin/activate
python -m pytest tests/ -v
```

Tests do **not** require a camera, GPIO hardware, or Blynk connectivity — they use mocks and stubs throughout.

Expected output:
```
tests/test_database.py ............ PASSED
tests/test_recognizer.py ..... PASSED
tests/test_hardware.py ....... PASSED
tests/test_access_log.py ...... PASSED
```

---

## Troubleshooting

### Camera not detected
```bash
# Check available cameras
ls /dev/video*
# Test with OpenCV
python -c "import cv2; c=cv2.VideoCapture(0); print(c.isOpened())"
```
Change `CAMERA_INDEX` in `config.py` if the default (0) is wrong.

### `ModuleNotFoundError: No module named 'face_recognition'`
- Ensure your virtual environment is activated: `source .venv/bin/activate`
- Rebuild dlib after increasing swap (see Installation section)

### `dlib compilation hangs / Pi freezes`
1. Increase swap to 1024 MB (see Installation)
2. Close all GUI apps (`sudo systemctl stop lightdm`)
3. Compile with fewer threads: `pip install dlib --global-option="--no" --global-option="avx"`

### Door doesn't unlock (GPIO)
- Confirm `GPIO_LOCK_PIN` matches your physical wiring
- Check relay LED lights up when triggering a manual unlock
- Test GPIO directly: `python -c "import RPi.GPIO as G; G.setmode(G.BCM); G.setup(18,G.OUT); G.output(18,True)"`

### Blynk shows "Offline"
- Verify `BLYNK_AUTH_TOKEN` is set correctly
- Check internet connectivity: `ping blynk.cloud`
- Inspect logs: `journalctl -u smart-door -f`

### False acceptances (wrong person unlocks door)
- Lower `FACE_TOLERANCE` to `0.45` or `0.40` in `config.py`
- Increase `confirm_frames` in `face_unlock_gui.py` (default 3)
- Re-enroll with more samples in consistent lighting

### False rejections (authorized person can't unlock)
- Raise `FACE_TOLERANCE` to `0.60`
- Enroll additional encodings in different lighting
- Ensure camera exposure is adequate (not backlit)

---

## Access Log

Events are written to `access_log.jsonl` as newline-delimited JSON:

```jsonl
{"timestamp":"2026-06-04T09:12:00","event":"STARTUP","source":"system"}
{"timestamp":"2026-06-04T09:13:45","event":"ACCESS_GRANTED","source":"face","name":"Alice","confidence":0.8821}
{"timestamp":"2026-06-04T09:15:00","event":"REMOTE_UNLOCK","source":"blynk"}
{"timestamp":"2026-06-04T09:16:10","event":"ENROLLMENT","source":"admin","name":"Bob"}
```

Parse recent events:
```bash
tail -n 50 access_log.jsonl | python -m json.tool
```

---

## Security Notes

1. **Change the default admin PIN** — set `ADMIN_PIN` env var before first use
2. **Protect the auth token** — never commit it to version control; use env vars or a secrets manager
3. **Physical security** — the relay/solenoid is a single point of failure; consider a backup mechanical lock
4. **Network security** — Blynk TLS encrypts all traffic; ensure your Pi's WiFi password is strong
5. **Database backup** — back up `live_database.pickle` regularly; losing it means re-enrolling everyone
6. **Tolerance tuning** — a tolerance of 0.55 is balanced; security-critical applications should use 0.45 or lower
