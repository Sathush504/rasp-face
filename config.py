"""
config.py — Central configuration for the Smart Door Access System.

All tuneable parameters live here so operators can adjust them
without touching business logic.
"""

import os

# Load .env file if present (e.g. BLYNK_AUTH_TOKEN, ADMIN_PIN)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass   # python-dotenv not installed — fall back to shell env vars only

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_FILE = os.path.join(BASE_DIR, "live_database.pickle")
LOG_FILE = os.path.join(BASE_DIR, "access_log.jsonl")     # newline-delimited JSON

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0          # 0 = first USB/built-in camera
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
PROCESS_SCALE = 0.25      # downscale factor before recognition (speed vs accuracy)
VIDEO_LOOP_MS = 15        # GUI refresh interval in milliseconds

# ---------------------------------------------------------------------------
# Face Recognition
# ---------------------------------------------------------------------------
FACE_TOLERANCE = 0.55     # lower = stricter match (0.0–1.0)
FACE_MODEL = "hog"        # "hog" (fast, CPU) | "cnn" (accurate, GPU/slow on Pi)
MIN_ENCODINGS_PER_PERSON = 3   # samples captured per enrollment session

# ---------------------------------------------------------------------------
# Door Lock Hardware (Raspberry Pi GPIO)
# ---------------------------------------------------------------------------
GPIO_LOCK_PIN = 18         # BCM pin connected to relay/solenoid gate
LOCK_ACTIVE_HIGH = True    # True if HIGH = unlocked; False if LOW = unlocked
UNLOCK_DURATION_SEC = 3    # seconds the lock stays open after auth success

# ---------------------------------------------------------------------------
# Blynk IoT
# ---------------------------------------------------------------------------
# Populate these via environment variables or edit directly.
# Never commit real tokens to version control.
BLYNK_AUTH_TOKEN = os.getenv("BLYNK_AUTH_TOKEN", "YOUR_BLYNK_AUTH_TOKEN_HERE")
BLYNK_SERVER = "blynk.cloud"
BLYNK_PORT = 443           # TLS

# Virtual Pin mapping (configure matching Datastreams in Blynk Console)
VPIN_UNLOCK_BUTTON = 0     # V0 — Button widget: sends 1 to unlock
VPIN_STATUS_LED = 1        # V1 — LED widget: shows lock state (1=unlocked)
VPIN_ACCESS_LOG = 2        # V2 — Terminal/Text widget: last access event
VPIN_LAST_USER = 3         # V3 — Label widget: name of last authenticated user

# ---------------------------------------------------------------------------
# Logging / Audit
# ---------------------------------------------------------------------------
LOG_MAX_BYTES = 5 * 1024 * 1024   # 5 MB before rotation
LOG_BACKUP_COUNT = 3

# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------
MAX_ENROLLMENT_SAMPLES = 10   # hard cap on encodings stored per person
ADMIN_PIN = os.getenv("ADMIN_PIN", "1234")   # PIN to protect admin actions in GUI
