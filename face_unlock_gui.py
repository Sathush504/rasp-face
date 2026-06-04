"""
face_unlock_gui.py — Main application: Smart Door Access System GUI.

Architecture
------------
  FaceUnlockApp (Tk root)
    ├── FaceDatabase        — persistent face-encoding store
    ├── FaceRecognizer      — processes camera frames, fires AuthResult events
    ├── DoorLock            — GPIO / stub solenoid controller
    ├── BlynkBridge         — MQTT cloud integration (daemon thread)
    ├── AccessLogger        — rotating JSONL audit log
    └── EnrollmentSession   — guided multi-sample face capture workflow

The GUI runs entirely in the Tkinter main thread.  Camera frames are
scheduled via window.after() so the event loop stays responsive.
All cross-thread calls (Blynk → GUI) go through Tk's after_idle().
"""

import datetime
import logging
import os
import sys
import threading
import tkinter as tk
import http.server
import socketserver
import time
from tkinter import messagebox, simpledialog
from typing import Optional

import cv2
import customtkinter as ctk
from PIL import Image, ImageTk

# ---------------------------------------------------------------------------
# Local modules
# ---------------------------------------------------------------------------
from access_log import AccessLogger
from blynk_bridge import BlynkBridge
from config import (
    ADMIN_PIN,
    BLYNK_AUTH_TOKEN,
    CAMERA_INDEX,
    DATABASE_FILE,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    GPIO_LOCK_PIN,
    LOCK_ACTIVE_HIGH,
    PROCESS_SCALE,
    UNLOCK_DURATION_SEC,
    VIDEO_LOOP_MS,
    VPIN_ACCESS_LOG,
    VPIN_LAST_USER,
    VPIN_STATUS_LED,
    VPIN_UNLOCK_BUTTON,
    USER_SCHEDULES,
    FALLBACK_PIN,
    LIVENESS_ENABLED,
)
from database import FaceDatabase
from enroller import EnrollmentSession
from hardware import DoorLock
from recognizer import AuthResult, FaceRecognizer, RecognitionEvent

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MJPEG Streaming Server for Blynk App / External Viewers
# ---------------------------------------------------------------------------
STREAM_PORT = 8090
latest_frame_jpeg = None
latest_frame_lock = threading.Lock()

def set_latest_frame(frame):
    global latest_frame_jpeg
    if frame is None:
        return
    ret, jpeg = cv2.imencode(".jpg", frame)
    if ret:
        with latest_frame_lock:
            latest_frame_jpeg = jpeg.tobytes()

class MJPEGStreamingHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Silence HTTP logs to prevent console spam

    def do_GET(self):
        if self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    with latest_frame_lock:
                        frame = latest_frame_jpeg
                    if frame is None:
                        time.sleep(0.03)
                        continue
                    # Direct binary write with leading CRLF for Webkit/Safari/iOS compatibility
                    self.wfile.write(
                        b"\r\n--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(frame)).encode("utf-8") + b"\r\n\r\n"
                    )
                    self.wfile.write(frame)
                    time.sleep(0.07)  # limit stream to ~15 FPS
            except Exception:
                pass
        elif self.path.startswith("/frame.jpg"):
            with latest_frame_lock:
                frame = latest_frame_jpeg
            if frame is not None:
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(frame)))
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                self.end_headers()
                self.wfile.write(frame)
            else:
                self.send_response(404)
                self.end_headers()
        elif self.path in ("/", "/index.html"):
            # Serve responsive, double-buffered HTML viewer to bypass iOS Safari MJPEG buffering bug
            html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Door Camera Feed</title>
    <style>
        html, body { margin: 0; padding: 0; width: 100%; height: 100%; background-color: #000; display: flex; align-items: center; justify-content: center; overflow: hidden; }
        img { width: 100%; height: 100%; max-width: 100%; max-height: 100%; object-fit: contain; }
    </style>
</head>
<body>
    <img id="feed" src="/frame.jpg" alt="Live Camera Feed">
    <script>
        const img = document.getElementById('feed');
        function loadNext() {
            const nextImg = new Image();
            nextImg.onload = () => {
                img.src = nextImg.src;
                setTimeout(loadNext, 60); // limit to ~15 FPS
            };
            nextImg.onerror = () => {
                setTimeout(loadNext, 500);
            };
            nextImg.src = '/frame.jpg?t=' + Date.now();
        }
        loadNext();
    </script>
</body>
</html>
"""
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

def start_streaming_server():
    socketserver.TCPServer.allow_reuse_address = True
    server = socketserver.TCPServer(("", STREAM_PORT), MJPEGStreamingHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    logger.info("MJPEG Cam Stream Server started on port %d", STREAM_PORT)


# ---------------------------------------------------------------------------
# Colour / style constants
# ---------------------------------------------------------------------------
CLR_BG = "#0f1117"
CLR_PANEL = "#1a1d27"
CLR_ACCENT = "#00e5ff"
CLR_GREEN = "#00c853"
CLR_RED = "#ff1744"
CLR_AMBER = "#ffc400"
CLR_TEXT = "#e8eaf6"
CLR_MUTED = "#546e7a"
CLR_BORDER = "#263238"


# ===========================================================================
# Main application window
# ===========================================================================
class FaceUnlockApp:
    """Top-level GUI orchestrating all subsystems."""

    def __init__(self, window: ctk.CTk, title: str = "Smart Door Access System"):
        self.window = window
        self.window.title(title)
        self.window.configure(fg_color=CLR_BG)
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)

        # ------------------------------------------------------------------
        # Subsystem initialisation
        # ------------------------------------------------------------------
        self._startup_time = time.time()
        self._db = FaceDatabase(DATABASE_FILE)
        self._door = DoorLock(
            pin=GPIO_LOCK_PIN,
            active_high=LOCK_ACTIVE_HIGH,
            unlock_duration=UNLOCK_DURATION_SEC,
        )
        self._door.set_state_change_callback(self._on_lock_state_change)
        self._recognizer = FaceRecognizer(self._db, confirm_frames=3, cooldown_sec=5)
        self._recognizer.add_event_callback(self._on_recognition_event)
        self._access_log = AccessLogger()
        self._blynk = BlynkBridge(
            auth_token=BLYNK_AUTH_TOKEN,
            on_remote_unlock=self._remote_unlock_requested,
            on_command=self._on_blynk_command_received,
            vpin_unlock=VPIN_UNLOCK_BUTTON,
            vpin_status=VPIN_STATUS_LED,
            vpin_log=VPIN_ACCESS_LOG,
            vpin_last_user=VPIN_LAST_USER,
        )

        # ------------------------------------------------------------------
        # Camera
        # ------------------------------------------------------------------
        self._cap = cv2.VideoCapture(CAMERA_INDEX)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self._camera_ok = self._cap.isOpened()
        if not self._camera_ok:
            logger.warning("Camera index %d not available.", CAMERA_INDEX)
        else:
            start_streaming_server()

        # ------------------------------------------------------------------
        # Enrollment session state
        # ------------------------------------------------------------------
        self._enroll_session: Optional[EnrollmentSession] = None
        self._enroll_overlay_active = False
        self._enroll_is_remote = False
        self._last_bgr = None

        # ------------------------------------------------------------------
        # Background Face Processing Thread
        # ------------------------------------------------------------------
        self._processing_lock = threading.Lock()
        self._processing_thread_active = True
        self._latest_boxes = []
        self._latest_labels = []
        self._pending_frame = None
        self._processing_thread = threading.Thread(target=self._background_processing_loop, daemon=True)
        self._processing_thread.start()

        # ------------------------------------------------------------------
        # Build UI
        # ------------------------------------------------------------------
        self._build_ui()
        self._access_log.log_event("STARTUP", source="system")
        self._update_status_bar(f"System ready  |  {self._db.person_count()} person(s) enrolled")
        self._poll_camera()

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------
    def _build_ui(self) -> None:
        """Assemble the complete UI layout."""
        # Fonts
        self._font_title = ("Inter", 13, "bold")
        self._font_label = ("Inter", 10)
        self._font_mono = ("Courier", 9)
        self._font_status_big = ("Inter", 16, "bold")

        # ── Top bar ───────────────────────────────────────────────────────
        top_bar = ctk.CTkFrame(self.window, fg_color=CLR_PANEL, corner_radius=0)
        top_bar.pack(fill=tk.X)

        ctk.CTkLabel(
            top_bar,
            text="🔐  Smart Door Access System",
            text_color=CLR_ACCENT,
            font=self._font_title,
        ).pack(side=tk.LEFT, padx=14, pady=8)

        self._lbl_blynk_status = ctk.CTkLabel(
            top_bar, text="● Blynk: Connecting…",
            text_color=CLR_AMBER,
            font=self._font_label,
        )
        self._lbl_blynk_status.pack(side=tk.RIGHT, padx=14, pady=8)

        # ── Main content area ─────────────────────────────────────────────
        content = ctk.CTkFrame(self.window, fg_color=CLR_BG, corner_radius=0)
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        # Left: video + lock badge
        left_col = ctk.CTkFrame(content, fg_color="transparent")
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._video_canvas = tk.Canvas(
            left_col, bg="black",
            width=FRAME_WIDTH, height=FRAME_HEIGHT,
            highlightthickness=2, highlightbackground=CLR_BORDER,
            bd=0
        )
        self._video_canvas.pack()

        # Lock status badge
        self._lock_badge = ctk.CTkLabel(
            left_col,
            text="🔒  LOCKED",
            fg_color=CLR_RED, text_color="white",
            font=self._font_status_big,
            corner_radius=8,
            height=36
        )
        self._lock_badge.pack(fill=tk.X, pady=(6, 0))

        # Enrollment progress bar (hidden by default)
        self._enroll_frame = ctk.CTkFrame(left_col, fg_color="transparent")
        self._enroll_label = ctk.CTkLabel(
            self._enroll_frame, text="", text_color=CLR_ACCENT,
            font=self._font_label,
        )
        self._enroll_label.pack()
        self._enroll_bar = ctk.CTkProgressBar(
            self._enroll_frame, width=FRAME_WIDTH, height=10,
            corner_radius=5, progress_color=CLR_ACCENT
        )
        self._enroll_bar.set(0.0)
        self._enroll_bar.pack(fill=tk.X, pady=4)

        # ── Status bar (created BEFORE control panel so _update_status_bar
        # is safe to call during _refresh_people_list inside _build_control_panel)
        self._status_var = tk.StringVar(value="Initialising…")
        self._status_label = ctk.CTkLabel(
            self.window,
            textvariable=self._status_var,
            fg_color=CLR_BORDER, text_color=CLR_MUTED,
            font=self._font_mono,
            corner_radius=0,
            height=20,
            anchor=tk.W
        )
        self._status_label.pack(fill=tk.X, side=tk.BOTTOM)

        # Right panel: controls + log
        right_col = ctk.CTkFrame(content, fg_color=CLR_PANEL, width=260, corner_radius=8)
        right_col.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))
        right_col.pack_propagate(False)

        self._build_control_panel(right_col)
        self._build_log_panel(right_col)

        # Periodically refresh Blynk connection indicator
        self._poll_blynk_status()

    def _build_control_panel(self, parent: ctk.CTkFrame) -> None:
        ctk.CTkLabel(
            parent, text="CONTROLS",
            text_color=CLR_MUTED, font=self._font_mono
        ).pack(anchor=tk.W, padx=10, pady=(10, 2))

        self._btn_enroll = ctk.CTkButton(
            parent, text="👤  Enroll New Face",
            command=self._start_enrollment,
            fg_color="#2563eb", hover_color="#1d4ed8", text_color="white",
            font=self._font_label, height=32, corner_radius=6
        )
        self._btn_enroll.pack(padx=10, pady=4, fill=tk.X)

        self._btn_remove = ctk.CTkButton(
            parent, text="🗑️  Remove Person",
            command=self._remove_person,
            fg_color="#4b5563", hover_color="#374151", text_color="white",
            font=self._font_label, height=32, corner_radius=6
        )
        self._btn_remove.pack(padx=10, pady=4, fill=tk.X)

        self._btn_remote_unlock = ctk.CTkButton(
            parent, text="🔓  Manual Unlock (Admin)",
            command=self._admin_unlock,
            fg_color="#0d9488", hover_color="#0f766e", text_color="white",
            font=self._font_label, height=32, corner_radius=6
        )
        self._btn_remote_unlock.pack(padx=10, pady=4, fill=tk.X)

        self._btn_clear = ctk.CTkButton(
            parent, text="⚠️  Clear All Profiles",
            command=self._clear_database,
            fg_color="#7f1d1d", hover_color="#b91c1c", text_color="white",
            font=self._font_label, height=32, corner_radius=6
        )
        self._btn_clear.pack(padx=10, pady=(14, 4), fill=tk.X)

        # Enrolled people list
        ctk.CTkLabel(
            parent, text="ENROLLED",
            text_color=CLR_MUTED, font=self._font_mono
        ).pack(anchor=tk.W, padx=10, pady=(14, 2))

        list_frame = ctk.CTkFrame(parent, fg_color="transparent")
        list_frame.pack(padx=10, fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(list_frame, bg=CLR_BORDER, bd=0)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._people_list = tk.Listbox(
            list_frame,
            yscrollcommand=scrollbar.set,
            bg=CLR_BG, fg=CLR_TEXT,
            selectbackground=CLR_ACCENT, selectforeground=CLR_BG,
            font=self._font_mono, relief=tk.FLAT, bd=0,
            highlightthickness=0
        )
        self._people_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self._people_list.yview)

        self._refresh_people_list()

    def _build_log_panel(self, parent: ctk.CTkFrame) -> None:
        ctk.CTkLabel(
            parent, text="ACCESS LOG",
            text_color=CLR_MUTED, font=self._font_mono
        ).pack(anchor=tk.W, padx=10, pady=(10, 2))

        log_frame = ctk.CTkFrame(parent, fg_color="transparent")
        log_frame.pack(padx=10, fill=tk.BOTH, expand=True, pady=(0, 10))

        log_scroll = tk.Scrollbar(log_frame, bg=CLR_BORDER, bd=0)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self._log_text = tk.Text(
            log_frame,
            yscrollcommand=log_scroll.set,
            bg=CLR_BG, fg=CLR_GREEN,
            font=self._font_mono,
            relief=tk.FLAT, bd=0,
            state=tk.DISABLED, wrap=tk.WORD,
            height=10,
            highlightthickness=0
        )
        self._log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scroll.config(command=self._log_text.yview)

        # Colour tags
        self._log_text.tag_config("grant", foreground=CLR_GREEN)
        self._log_text.tag_config("deny", foreground=CLR_RED)
        self._log_text.tag_config("info", foreground=CLR_AMBER)
        self._log_text.tag_config("blynk", foreground=CLR_ACCENT)

    # -----------------------------------------------------------------------
    # Camera polling loop
    # -----------------------------------------------------------------------
    def _poll_camera(self) -> None:
        """Scheduled every VIDEO_LOOP_MS ms — reads a frame and processes it."""
        if not self._camera_ok:
            self._show_no_camera()
            self.window.after(1000, self._poll_camera)
            return

        ret, bgr = self._cap.read()
        if not ret:
            self._show_no_camera()
            self.window.after(VIDEO_LOOP_MS, self._poll_camera)
            return

        bgr = cv2.flip(bgr, 1)      # mirror effect
        self._last_bgr = bgr.copy()  # Save for snapshot purposes

        # Enrollment mode intercept
        if self._enroll_session is not None:
            self._handle_enrollment_frame(bgr)
        else:
            # Send frame to background thread for processing (non-blocking!)
            with self._processing_lock:
                self._pending_frame = bgr.copy()
            
            # Draw the last known face boxes and labels (very fast!)
            with self._processing_lock:
                boxes = self._latest_boxes
                labels = self._latest_labels
            self._draw_overlays(bgr, boxes, labels)

        self._render_frame(bgr)
        set_latest_frame(bgr)
        self.window.after(VIDEO_LOOP_MS, self._poll_camera)

    def _background_processing_loop(self) -> None:
        while self._processing_thread_active:
            frame_to_process = None
            with self._processing_lock:
                if self._pending_frame is not None:
                    frame_to_process = self._pending_frame
                    self._pending_frame = None
            
            if frame_to_process is not None:
                try:
                    boxes, labels, event = self._recognizer.process_frame(frame_to_process)
                    with self._processing_lock:
                        self._latest_boxes = boxes
                        self._latest_labels = labels
                    if event:
                        self._on_recognition_event(event)
                except Exception as e:
                    logger.exception("Error in background face processing: %s", e)
            
            # Sleep slightly to prevent CPU pinning
            time.sleep(0.015)

    def _draw_overlays(self, frame, boxes, labels) -> None:
        import numpy as np  # inline to keep top-level lean
        for (top, right, bottom, left), name in zip(boxes, labels):
            colour = (0, 200, 80) if name != "Unknown" else (255, 60, 60)
            cv2.rectangle(frame, (left, top), (right, bottom), colour, 2)
            cv2.rectangle(frame, (left, bottom - 28), (right, bottom), colour, cv2.FILLED)
            cv2.putText(
                frame, name,
                (left + 6, bottom - 6),
                cv2.FONT_HERSHEY_DUPLEX, 0.55,
                (255, 255, 255), 1
            )

    def _render_frame(self, bgr) -> None:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(rgb))
        self._video_canvas.create_image(0, 0, anchor=tk.NW, image=img)
        self._video_canvas._img_ref = img   # prevent GC

    def _show_no_camera(self) -> None:
        self._video_canvas.create_rectangle(
            0, 0, FRAME_WIDTH, FRAME_HEIGHT, fill="black"
        )
        self._video_canvas.create_text(
            FRAME_WIDTH // 2, FRAME_HEIGHT // 2,
            text="⚠ Camera unavailable",
            fill=CLR_AMBER, font=self._font_title,
        )

    # -----------------------------------------------------------------------
    # Enrollment workflow
    # -----------------------------------------------------------------------
    def _start_enrollment(self) -> None:
        if not self._verify_admin("enroll a new face"):
            return
        name = simpledialog.askstring(
            "New Enrollment",
            "Enter the person's full name:",
            parent=self.window,
        )
        if not name or not name.strip():
            return
        self._enroll_is_remote = False
        self._enroll_session = EnrollmentSession(name.strip(), self._db)
        self._enroll_frame.pack(fill=tk.X, pady=4)
        self._enroll_label.configure(text=f"Enrolling: {name.strip()}")
        self._update_status_bar(f"Enrollment started for '{name.strip()}' — look at the camera")
        logger.info("Enrollment session started for '%s'.", name.strip())

    def _start_remote_enrollment(self, name: str) -> None:
        """Start enrollment workflow requested via Blynk terminal."""
        if self._enroll_session is not None:
            logger.warning("Enrollment session already active. Ignoring remote request.")
            return
        self._enroll_is_remote = True
        self._enroll_session = EnrollmentSession(name, self._db)
        self._enroll_frame.pack(fill=tk.X, pady=4)
        self._enroll_label.configure(text=f"Enrolling: {name}")
        self._update_status_bar(f"Remote enrollment started for '{name}' — look at the camera")
        logger.info("Remote enrollment session started for '%s'.", name)
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self._blynk.send_access_event(f"[{ts}] ⏳ Remote enrollment started for '{name}' - look at the camera!")

    def _handle_enrollment_frame(self, bgr) -> None:
        result = self._enroll_session.feed_frame(bgr)
        self._enroll_label.configure(text=result.message)
        self._enroll_bar.set(result.progress / 100.0)

        # Draw a cyan border during enrollment
        h, w = bgr.shape[:2]
        cv2.rectangle(bgr, (0, 0), (w - 1, h - 1), (0, 229, 255), 4)
        pct_text = f"Capturing: {result.progress}%"
        cv2.putText(bgr, pct_text, (10, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 229, 255), 2)

        if result.done:
            self._enroll_session = None
            self._enroll_frame.pack_forget()
            self._refresh_people_list()
            self._update_status_bar(result.message)
            ts = datetime.datetime.now().strftime("%H:%M:%S")

            if result.success:
                name = result.message.split("'")[1] if "'" in result.message else "?"
                self._access_log.log_event(
                    "ENROLLMENT", name=name, source="blynk" if self._enroll_is_remote else "admin"
                )
                self._blynk.send_access_event(f"[{ts}] ✓ ENROLL COMPLETE: {name}")
                if not self._enroll_is_remote:
                    messagebox.showinfo("Enrollment Complete", result.message, parent=self.window)
            else:
                self._blynk.send_access_event(f"[{ts}] ✗ ENROLL FAILED: {result.message}")
                if not self._enroll_is_remote:
                    messagebox.showwarning("Enrollment Failed", result.message, parent=self.window)

    # -----------------------------------------------------------------------
    # Remove person
    # -----------------------------------------------------------------------
    def _remove_person(self) -> None:
        if not self._verify_admin("remove a person"):
            return
        people = self._db.list_people()
        if not people:
            messagebox.showinfo("No Profiles", "No enrolled profiles found.", parent=self.window)
            return

        # Simple selection dialog
        dialog = _SelectPersonDialog(self.window, people, title="Remove Person")
        name = dialog.result
        if not name:
            return

        confirm = messagebox.askyesno(
            "Confirm Removal",
            f"Delete ALL encodings for '{name}'?\nThis cannot be undone.",
            parent=self.window,
        )
        if confirm:
            success = self._db.remove_person(name)
            if success:
                self._access_log.log_event("REMOVAL", name=name, source="admin")
                self._blynk.send_access_event(f"[REMOVE] {name}")
                self._refresh_people_list()
                self._update_status_bar(f"Profile '{name}' removed.")
            else:
                messagebox.showerror("Error", f"Could not remove '{name}'.", parent=self.window)

    # -----------------------------------------------------------------------
    # Admin unlock
    # -----------------------------------------------------------------------
    def _admin_unlock(self) -> None:
        if not self._verify_admin("manually unlock the door"):
            return
        self._door.unlock(triggered_by="admin")
        self._access_log.log_event("REMOTE_UNLOCK", source="admin")
        self._append_log("Manual unlock by admin", tag="info")
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self._blynk.send_access_event(f"[{ts}] ✓ UNLOCKED: Admin GUI")

    # -----------------------------------------------------------------------
    # Remote unlock (from Blynk — runs in Blynk's MQTT thread)
    # -----------------------------------------------------------------------
    def _remote_unlock_requested(self) -> None:
        """Called from the Blynk MQTT thread — must be marshalled to Tk thread."""
        self.window.after_idle(self._do_remote_unlock)

    def _do_remote_unlock(self) -> None:
        self._door.unlock(triggered_by="blynk")
        self._access_log.log_event("REMOTE_UNLOCK", source="blynk")
        self._append_log("Remote unlock via Blynk", tag="blynk")
        self._update_status_bar("Door unlocked via Blynk app")
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self._blynk.send_access_event(f"[{ts}] ✓ UNLOCKED: Blynk Switch")

    def _on_blynk_command_received(self, cmd_text: str) -> None:
        """Called from the Blynk MQTT thread — must be marshalled to Tk thread."""
        self.window.after_idle(self._process_blynk_command, cmd_text)

    def _process_blynk_command(self, cmd_text: str) -> None:
        cmd = cmd_text.lower().strip()
        ts = datetime.datetime.now().strftime("%H:%M:%S")

        if cmd == "help":
            help_msg = (
                f"[{ts}] --- COMMANDS ---\n"
                "  help        - Show commands\n"
                "  status      - System state\n"
                "  log         - Last 5 events\n"
                "  unlock      - Open door\n"
                "  lock        - Secure door\n"
                "  enroll:Name - Enroll face\n"
                "  remove:Name - Delete user\n"
                "  pin:XXXX    - Fallback PIN unlock"
            )
            self._blynk.send_access_event(help_msg)

        elif cmd == "status":
            state = "UNLOCKED" if self._door.is_unlocked else "LOCKED"
            people = self._db.list_people()
            uptime_sec = int(time.time() - self._startup_time)
            uptime_str = f"{uptime_sec // 60}m {uptime_sec % 60}s"

            # Get dynamic thresholds from recognizer
            user_details = []
            with self._recognizer._lock:
                for name in people:
                    sched = USER_SCHEDULES.get(name, "Always")
                    if isinstance(sched, tuple) or isinstance(sched, list):
                        sched_str = f"{sched[0]}-{sched[1]}"
                    else:
                        sched_str = str(sched)
                    
                    thresh = self._recognizer._dynamic_threshold.get(name)
                    if thresh is not None:
                        user_details.append(f"  - {name} [{sched_str}] (Calibrated: {thresh:.3f})")
                    else:
                        user_details.append(f"  - {name} [{sched_str}] (Config: default)")

            users_str = "\n".join(user_details) if user_details else "  No users registered"

            status_msg = (
                f"[{ts}] --- STATUS ---\n"
                f"  Lock: {state}\n"
                f"  Liveness: {'ON' if LIVENESS_ENABLED else 'OFF'}\n"
                f"  Uptime: {uptime_str}\n"
                f"  Users:\n{users_str}"
            )
            self._blynk.send_access_event(status_msg)

        elif cmd == "log":
            recent = self._access_log.read_recent(n=5)
            log_lines = []
            for event in reversed(recent):
                log_ts = event.get("timestamp", "").split("T")[-1][:8]
                evt_type = event.get("event", "")
                if evt_type == "ACCESS_GRANTED":
                    name = event.get("name", "Unknown")
                    log_lines.append(f"  {log_ts} ✓ Grant: {name}")
                elif evt_type == "UNAUTHORIZED_ACCESS":
                    log_lines.append(f"  {log_ts} ✗ Alert: Unknown")
                else:
                    log_lines.append(f"  {log_ts} • {evt_type}")

            log_msg = f"[{ts}] --- RECENT LOGS ---\n" + "\n".join(log_lines)
            self._blynk.send_access_event(log_msg)

        elif cmd == "unlock":
            self._door.unlock(triggered_by="blynk_cmd")
            self._access_log.log_event("REMOTE_UNLOCK", source="blynk_cmd")
            self._append_log("Remote unlock via Blynk command", tag="blynk")
            self._update_status_bar("Door unlocked via Blynk terminal")
            self._blynk.send_access_event(f"[{ts}] ✓ UNLOCKED: Blynk Terminal")

        elif cmd == "lock":
            self._door.lock()
            self._access_log.log_event("REMOTE_LOCK", source="blynk_cmd")
            self._append_log("Remote lock via Blynk command", tag="blynk")
            self._update_status_bar("Door locked via Blynk terminal")
            self._blynk.send_access_event(f"[{ts}] 🔒 LOCKED: Blynk Terminal")

        elif cmd.startswith("enroll:"):
            enroll_name = cmd_text[7:].strip()
            if not enroll_name:
                self._blynk.send_access_event(f"[{ts}] ✗ Error: Name cannot be empty. Usage: enroll:Name")
            elif self._enroll_session is not None:
                self._blynk.send_access_event(f"[{ts}] ✗ Error: Enrollment session already active.")
            else:
                self._start_remote_enrollment(enroll_name)

        elif cmd.startswith("remove:"):
            remove_name = cmd_text[7:].strip()
            if not remove_name:
                self._blynk.send_access_event(f"[{ts}] ✗ Error: Name cannot be empty. Usage: remove:Name")
            else:
                people = self._db.list_people()
                match = None
                for person in people:
                    if person.lower() == remove_name.lower():
                        match = person
                        break

                if match:
                    success = self._db.remove_person(match)
                    if success:
                        self._access_log.log_event("REMOVAL", name=match, source="blynk")
                        self._blynk.send_access_event(f"[{ts}] ✓ REMOVED: {match}")
                        self._refresh_people_list()
                        self._update_status_bar(f"Profile '{match}' removed via Blynk.")
                    else:
                        self._blynk.send_access_event(f"[{ts}] ✗ Error: Failed to remove {match}.")
                else:
                    self._blynk.send_access_event(f"[{ts}] ✗ Error: User '{remove_name}' not found.")

        elif cmd.startswith("pin:"):
            pin_code = cmd_text[4:].strip()
            if pin_code == FALLBACK_PIN:
                self._door.unlock(triggered_by="blynk_pin_fallback")
                self._access_log.log_event("PIN_UNLOCK", source="blynk")
                self._append_log("Door unlocked via fallback PIN", tag="blynk")
                self._update_status_bar("Door unlocked via fallback PIN")
                self._blynk.send_access_event(f"[{ts}] ✓ UNLOCKED: Fallback PIN")
            else:
                self._blynk.send_access_event(f"[{ts}] ✗ Error: Invalid PIN.")

        else:
            self._blynk.send_access_event(f"[{ts}] Unknown command. Type 'help' for commands.")

    # -----------------------------------------------------------------------
    # Clear database
    # -----------------------------------------------------------------------
    def _clear_database(self) -> None:
        if not self._verify_admin("clear the entire database"):
            return
        confirm = messagebox.askyesno(
            "⚠ Dangerous Action",
            "Delete ALL enrolled face profiles permanently?",
            parent=self.window,
        )
        if confirm:
            self._db.clear_all()
            self._refresh_people_list()
            self._access_log.log_event("CLEAR_DATABASE", source="admin")
            self._update_status_bar("All profiles cleared.")
            messagebox.showinfo("Done", "All profiles have been removed.", parent=self.window)

    # -----------------------------------------------------------------------
    # Recognition event handler (from Recognizer — may be any thread)
    # -----------------------------------------------------------------------
    def _on_recognition_event(self, event: RecognitionEvent) -> None:
        self.window.after_idle(self._process_recognition_event, event)

    def _process_recognition_event(self, event: RecognitionEvent) -> None:
        ts = event.timestamp.strftime("%H:%M:%S")
        if event.result == AuthResult.AUTHORIZED:
            # Check user schedule
            if event.name in USER_SCHEDULES:
                allowed_start_str, allowed_end_str = USER_SCHEDULES[event.name]
                now_time = datetime.datetime.now().time()
                try:
                    start_time = datetime.datetime.strptime(allowed_start_str, "%H:%M").time()
                    end_time = datetime.datetime.strptime(allowed_end_str, "%H:%M").time()
                    
                    if not (start_time <= now_time <= end_time):
                        msg = f"[{ts}] ✗ DENIED: {event.name} (Outside allowed schedule {allowed_start_str}-{allowed_end_str})"
                        self._append_log(msg, tag="alert")
                        self._update_status_bar(f"Access denied: {event.name} (restricted hours)")
                        self._blynk.send_access_event(msg)
                        self._access_log.log_event(
                            "ACCESS_DENIED", name=event.name,
                            confidence=event.confidence, source="face",
                            extra={"reason": "outside_schedule"}
                        )
                        return
                except Exception as exc:
                    logger.error("Error parsing user schedule for %s: %s", event.name, exc)

            msg = f"[{ts}] ✓ ACCESS: {event.name} ({event.confidence:.0%})"
            self._append_log(msg, tag="grant")
            self._update_status_bar(f"Access granted → {event.name}")
            self._blynk.send_access_event(msg)
            self._blynk.update_last_user(event.name or "")
            self._access_log.log_event(
                "ACCESS_GRANTED", name=event.name,
                confidence=event.confidence, source="face"
            )
            # Unlock physical lock
            self._door.unlock(triggered_by=event.name or "face")

        elif event.result == AuthResult.UNKNOWN:
            msg = f"[{ts}] ✗ ALERT: Unknown person detected!"
            self._append_log(msg, tag="alert")
            self._update_status_bar("Unknown face detected!")
            self._blynk.send_access_event(msg)
            
            # Save snapshot of the intruder
            intruders_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "intruders")
            os.makedirs(intruders_dir, exist_ok=True)
            timestamp_str = event.timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"intruder_{timestamp_str}.jpg"
            filepath = os.path.join(intruders_dir, filename)
            if hasattr(self, "_last_bgr") and self._last_bgr is not None:
                cv2.imwrite(filepath, self._last_bgr)
                logger.info("Saved intruder snapshot to %s", filepath)
            
            self._access_log.log_event(
                "UNAUTHORIZED_ACCESS", source="face",
                extra={"snapshot": filename}
            )
            self._blynk.trigger_event("unauthorized_access")

    # -----------------------------------------------------------------------
    # Lock state callback (from DoorLock — may be any thread)
    # -----------------------------------------------------------------------
    def _on_lock_state_change(self, is_unlocked: bool) -> None:
        self.window.after_idle(self._update_lock_ui, is_unlocked)

    def _update_lock_ui(self, is_unlocked: bool) -> None:
        if is_unlocked:
            self._lock_badge.configure(text="🔓  UNLOCKED", fg_color=CLR_GREEN)
        else:
            self._lock_badge.configure(text="🔒  LOCKED", fg_color=CLR_RED)
        self._blynk.update_lock_status(is_unlocked)

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------
    def _verify_admin(self, action: str) -> bool:
        """Prompt for the admin PIN before sensitive actions."""
        if ADMIN_PIN == "1234":
            # Warn if default PIN is still set
            logger.warning("Default admin PIN in use — set ADMIN_PIN env var.")
        pin = simpledialog.askstring(
            "Admin Authentication",
            f"Enter admin PIN to {action}:",
            show="*",
            parent=self.window,
        )
        if pin != ADMIN_PIN:
            messagebox.showerror("Denied", "Incorrect admin PIN.", parent=self.window)
            return False
        return True

    def _refresh_people_list(self) -> None:
        self._people_list.delete(0, tk.END)
        for name in self._db.list_people():
            self._people_list.insert(tk.END, f"  {name}")
        count = self._db.person_count()
        enc = self._db.encoding_count()
        self._update_status_bar(
            f"{count} person(s) enrolled  |  {enc} total encodings"
        )

    def _append_log(self, text: str, tag: str = "info") -> None:
        self._log_text.config(state=tk.NORMAL)
        self._log_text.insert(tk.END, text + "\n", tag)
        self._log_text.see(tk.END)
        self._log_text.config(state=tk.DISABLED)

    def _update_status_bar(self, text: str) -> None:
        self._status_var.set(
            f"  {datetime.datetime.now().strftime('%H:%M:%S')}  |  {text}"
        )

    def _poll_blynk_status(self) -> None:
        """Refresh the Blynk connection indicator every 5 s."""
        if self._blynk.is_connected():
            self._lbl_blynk_status.configure(
                text="● Blynk: Online", text_color=CLR_GREEN
            )
        else:
            self._lbl_blynk_status.configure(
                text="● Blynk: Offline", text_color=CLR_AMBER
            )
        self.window.after(5000, self._poll_blynk_status)

    def _on_close(self) -> None:
        logger.info("Shutting down…")
        self._processing_thread_active = False
        self._access_log.log_event("SHUTDOWN", source="system")
        self._access_log.close()
        self._blynk.stop()
        self._door.cleanup()
        if self._cap.isOpened():
            self._cap.release()
        self.window.destroy()


# ===========================================================================
# Helper dialog — person selection from a list
# ===========================================================================
class _SelectPersonDialog(ctk.CTkToplevel):
    """Modal dialog with a listbox for selecting a person name."""

    def __init__(self, parent, people: list, title="Select Person"):
        super().__init__(parent)
        self.title(title)
        self.resizable(False, False)
        self.configure(fg_color=CLR_BG)
        self.result = None
        self.grab_set()

        ctk.CTkLabel(self, text="Select a person:", text_color=CLR_TEXT,
                     font=("Inter", 11, "bold")).pack(padx=16, pady=(12, 4))

        list_frame = ctk.CTkFrame(self, fg_color=CLR_PANEL, corner_radius=8)
        list_frame.pack(padx=16, pady=4)

        self._lb = tk.Listbox(
            list_frame, bg=CLR_PANEL, fg=CLR_TEXT,
            selectbackground=CLR_ACCENT, selectforeground=CLR_BG,
            font=("Courier", 10), relief=tk.FLAT, bd=0,
            width=30, height=min(len(people), 10),
            highlightthickness=0
        )
        for p in people:
            self._lb.insert(tk.END, p)
        self._lb.pack(padx=8, pady=8)
        self._lb.bind("<Double-Button-1>", self._ok)

        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(pady=10)

        ctk.CTkButton(
            btn_frame, text="Select", command=self._ok,
            fg_color=CLR_ACCENT, text_color=CLR_BG, font=("Inter", 11, "bold"),
            hover_color="#00b8d4", width=80, height=28, corner_radius=4
        ).pack(side=tk.LEFT, padx=6)

        ctk.CTkButton(
            btn_frame, text="Cancel", command=self.destroy,
            fg_color=CLR_BORDER, text_color=CLR_TEXT, font=("Inter", 11),
            hover_color="#374151", width=80, height=28, corner_radius=4
        ).pack(side=tk.LEFT, padx=6)

        self.wait_window()

    def _ok(self, _event=None) -> None:
        sel = self._lb.curselection()
        if sel:
            self.result = self._lb.get(sel[0])
        self.destroy()


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    root = ctk.CTk()
    root.resizable(False, False)
    app = FaceUnlockApp(root, "🔐 Smart Door Access System")
    root.mainloop()