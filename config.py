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
LOCK_ACTIVE_HIGH = False    # True if HIGH = unlocked; False if LOW = unlocked
UNLOCK_DURATION_SEC = 3    # seconds the lock stays open after auth success
REMOTE_GPIO_IP = os.getenv("REMOTE_GPIO_IP", None)  # Set to Raspberry Pi IP to control GPIO over WiFi

# ---------------------------------------------------------------------------
# Blynk IoT
# ---------------------------------------------------------------------------
# Populate these via environment variables or edit directly.
# Never commit real tokens to version control.
BLYNK_AUTH_TOKEN = os.getenv("BLYNK_AUTH_TOKEN", "YOUR_BLYNK_AUTH_TOKEN_HERE")
BLYNK_SERVER = "blynk.cloud"
BLYNK_PORT = 443           # TLS

# Virtual Pin mapping (configure matching Datastreams in Blynk Console)
VPIN_UNLOCK_BUTTON = "Unlock Button"  # V0 — Button widget
VPIN_STATUS_LED = "Lock Status"        # V1 — LED widget
VPIN_ACCESS_LOG = "Access Log"        # V2 — Terminal/Text widget
VPIN_LAST_USER = "Last User"          # V3 — Label widget

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
FALLBACK_PIN = os.getenv("FALLBACK_PIN", "9999")  # Fallback PIN to unlock via Blynk terminal
# Access schedules format: {"Name": ("Start_HH:MM", "End_HH:MM")}
# If a person is not in this dictionary, they have 24/7 access.
USER_SCHEDULES = {
    "Guest": ("09:00", "17:00")
}
