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
from tkinter import font as tkfont
from tkinter import messagebox, simpledialog, ttk
from typing import Optional

import cv2
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
                    self.wfile.write(b"--frame\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(frame)))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                    time.sleep(0.07)  # limit stream to ~15 FPS
            except Exception:
                pass
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

    def __init__(self, window: tk.Tk, title: str = "Smart Door Access System"):
        self.window = window
        self.window.title(title)
        self.window.configure(bg=CLR_BG)
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)

        # ------------------------------------------------------------------
        # Subsystem initialisation
        # ------------------------------------------------------------------
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
        self._font_title = tkfont.Font(family="Inter", size=13, weight="bold")
        self._font_label = tkfont.Font(family="Inter", size=10)
        self._font_mono = tkfont.Font(family="Courier", size=9)
        self._font_status_big = tkfont.Font(family="Inter", size=16, weight="bold")

        # ── Top bar ───────────────────────────────────────────────────────
        top_bar = tk.Frame(self.window, bg=CLR_PANEL, pady=6)
        top_bar.pack(fill=tk.X)

        tk.Label(
            top_bar,
            text="🔐  Smart Door Access System",
            bg=CLR_PANEL,
            fg=CLR_ACCENT,
            font=self._font_title,
        ).pack(side=tk.LEFT, padx=14)

        self._lbl_blynk_status = tk.Label(
            top_bar, text="● Blynk: Connecting…",
            bg=CLR_PANEL, fg=CLR_AMBER,
            font=self._font_label,
        )
        self._lbl_blynk_status.pack(side=tk.RIGHT, padx=14)

        # ── Main content area ─────────────────────────────────────────────
        content = tk.Frame(self.window, bg=CLR_BG)
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        # Left: video + lock badge
        left_col = tk.Frame(content, bg=CLR_BG)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._video_canvas = tk.Canvas(
            left_col, bg="black",
            width=FRAME_WIDTH, height=FRAME_HEIGHT,
            highlightthickness=2, highlightbackground=CLR_BORDER,
        )
        self._video_canvas.pack()

        # Lock status badge
        self._lock_badge = tk.Label(
            left_col,
            text="🔒  LOCKED",
            bg=CLR_RED, fg="white",
            font=self._font_status_big,
            padx=20, pady=8,
        )
        self._lock_badge.pack(fill=tk.X, pady=(6, 0))

        # Enrollment progress bar (hidden by default)
        self._enroll_frame = tk.Frame(left_col, bg=CLR_BG)
        self._enroll_label = tk.Label(
            self._enroll_frame, text="", bg=CLR_BG, fg=CLR_ACCENT,
            font=self._font_label,
        )
        self._enroll_label.pack()
        self._enroll_bar = ttk.Progressbar(
            self._enroll_frame, length=FRAME_WIDTH, maximum=100
        )
        self._enroll_bar.pack(fill=tk.X)

        # ── Status bar (created BEFORE control panel so _update_status_bar
        # is safe to call during _refresh_people_list inside _build_control_panel)
        self._status_var = tk.StringVar(value="Initialising…")
        tk.Label(
            self.window,
            textvariable=self._status_var,
            bg=CLR_BORDER, fg=CLR_MUTED,
            font=self._font_mono,
            anchor=tk.W, padx=8, pady=3,
        ).pack(fill=tk.X, side=tk.BOTTOM)

        # Right panel: controls + log
        right_col = tk.Frame(content, bg=CLR_PANEL, width=260)
        right_col.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))
        right_col.pack_propagate(False)

        self._build_control_panel(right_col)
        self._build_log_panel(right_col)

        # Periodically refresh Blynk connection indicator
        self._poll_blynk_status()

    def _build_control_panel(self, parent: tk.Frame) -> None:
        tk.Label(
            parent, text="CONTROLS",
            bg=CLR_PANEL, fg=CLR_MUTED, font=self._font_mono
        ).pack(anchor=tk.W, padx=10, pady=(10, 2))

        btn_cfg = dict(
            bg=CLR_BORDER, fg=CLR_TEXT,
            activebackground=CLR_ACCENT, activeforeground=CLR_BG,
            font=self._font_label,
            relief=tk.FLAT, cursor="hand2",
            width=24, pady=6,
        )

        self._btn_enroll = tk.Button(
            parent, text="👤  Enroll New Face",
            command=self._start_enrollment, **btn_cfg
        )
        self._btn_enroll.pack(padx=10, pady=4, fill=tk.X)

        self._btn_remove = tk.Button(
            parent, text="🗑️  Remove Person",
            command=self._remove_person, **btn_cfg
        )
        self._btn_remove.pack(padx=10, pady=4, fill=tk.X)

        self._btn_remote_unlock = tk.Button(
            parent, text="🔓  Manual Unlock (Admin)",
            command=self._admin_unlock, **btn_cfg
        )
        self._btn_remote_unlock.pack(padx=10, pady=4, fill=tk.X)

        self._btn_clear = tk.Button(
            parent, text="⚠️  Clear All Profiles",
            command=self._clear_database,
            bg="#37000a", fg=CLR_RED,
            activebackground=CLR_RED, activeforeground="white",
            font=self._font_label, relief=tk.FLAT, cursor="hand2",
            width=24, pady=6,
        )
        self._btn_clear.pack(padx=10, pady=(14, 4), fill=tk.X)

        # Enrolled people list
        tk.Label(
            parent, text="ENROLLED", bg=CLR_PANEL,
            fg=CLR_MUTED, font=self._font_mono
        ).pack(anchor=tk.W, padx=10, pady=(14, 2))

        list_frame = tk.Frame(parent, bg=CLR_PANEL)
        list_frame.pack(padx=10, fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(list_frame, bg=CLR_BORDER)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._people_list = tk.Listbox(
            list_frame,
            yscrollcommand=scrollbar.set,
            bg=CLR_BG, fg=CLR_TEXT,
            selectbackground=CLR_ACCENT, selectforeground=CLR_BG,
            font=self._font_mono, relief=tk.FLAT, bd=0,
        )
        self._people_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self._people_list.yview)

        self._refresh_people_list()

    def _build_log_panel(self, parent: tk.Frame) -> None:
        tk.Label(
            parent, text="ACCESS LOG",
            bg=CLR_PANEL, fg=CLR_MUTED, font=self._font_mono
        ).pack(anchor=tk.W, padx=10, pady=(10, 2))

        log_frame = tk.Frame(parent, bg=CLR_PANEL)
        log_frame.pack(padx=10, fill=tk.BOTH, expand=True, pady=(0, 10))

        log_scroll = tk.Scrollbar(log_frame, bg=CLR_BORDER)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self._log_text = tk.Text(
            log_frame,
            yscrollcommand=log_scroll.set,
            bg=CLR_BG, fg=CLR_GREEN,
            font=self._font_mono,
            relief=tk.FLAT, bd=0,
            state=tk.DISABLED, wrap=tk.WORD,
            height=10,
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

        # Enrollment mode intercept
        if self._enroll_session is not None:
            self._handle_enrollment_frame(bgr)
        else:
            # Recognition mode
            boxes, labels, event = self._recognizer.process_frame(bgr)
            self._draw_overlays(bgr, boxes, labels)
            if event and event.result == AuthResult.AUTHORIZED:
                self._door.unlock(triggered_by=event.name or "face")

        self._render_frame(bgr)
        set_latest_frame(bgr)
        self.window.after(VIDEO_LOOP_MS, self._poll_camera)

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
        self._enroll_session = EnrollmentSession(name.strip(), self._db)
        self._enroll_frame.pack(fill=tk.X, pady=4)
        self._enroll_label.config(text=f"Enrolling: {name.strip()}")
        self._update_status_bar(f"Enrollment started for '{name.strip()}' — look at the camera")
        logger.info("Enrollment session started for '%s'.", name.strip())

    def _handle_enrollment_frame(self, bgr) -> None:
        result = self._enroll_session.feed_frame(bgr)
        self._enroll_label.config(text=result.message)
        self._enroll_bar["value"] = result.progress

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

            if result.success:
                name = result.message.split("'")[1] if "'" in result.message else "?"
                self._access_log.log_event(
                    "ENROLLMENT", name=name, source="admin"
                )
                self._blynk.send_access_event(f"[ENROLL] {name}")
                messagebox.showinfo("Enrollment Complete", result.message, parent=self.window)
            else:
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
            msg = f"[{ts}] ✓ ACCESS: {event.name} ({event.confidence:.0%})"
            self._append_log(msg, tag="grant")
            self._update_status_bar(f"Access granted → {event.name}")
            self._blynk.send_access_event(msg)
            self._blynk.update_last_user(event.name or "")
            self._access_log.log_event(
                "ACCESS_GRANTED", name=event.name,
                confidence=event.confidence, source="face"
            )
        elif event.result == AuthResult.UNKNOWN:
            msg = f"[{ts}] ✗ ALERT: Unknown person detected!"
            self._append_log(msg, tag="alert")
            self._update_status_bar("Unknown face detected!")
            self._blynk.send_access_event(msg)
            self._access_log.log_event(
                "UNAUTHORIZED_ACCESS", source="face"
            )

    # -----------------------------------------------------------------------
    # Lock state callback (from DoorLock — may be any thread)
    # -----------------------------------------------------------------------
    def _on_lock_state_change(self, is_unlocked: bool) -> None:
        self.window.after_idle(self._update_lock_ui, is_unlocked)

    def _update_lock_ui(self, is_unlocked: bool) -> None:
        if is_unlocked:
            self._lock_badge.config(text="🔓  UNLOCKED", bg=CLR_GREEN)
        else:
            self._lock_badge.config(text="🔒  LOCKED", bg=CLR_RED)
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
            self._lbl_blynk_status.config(
                text="● Blynk: Online", fg=CLR_GREEN
            )
        else:
            self._lbl_blynk_status.config(
                text="● Blynk: Offline", fg=CLR_AMBER
            )
        self.window.after(5000, self._poll_blynk_status)

    def _on_close(self) -> None:
        logger.info("Shutting down…")
        self._access_log.log_event("SHUTDOWN", source="system")
        self._blynk.stop()
        self._door.cleanup()
        if self._cap.isOpened():
            self._cap.release()
        self.window.destroy()


# ===========================================================================
# Helper dialog — person selection from a list
# ===========================================================================
class _SelectPersonDialog(tk.Toplevel):
    """Modal dialog with a listbox for selecting a person name."""

    def __init__(self, parent, people: list, title="Select Person"):
        super().__init__(parent)
        self.title(title)
        self.resizable(False, False)
        self.configure(bg=CLR_BG)
        self.result = None
        self.grab_set()

        tk.Label(self, text="Select a person:", bg=CLR_BG, fg=CLR_TEXT,
                 font=("Inter", 10)).pack(padx=16, pady=(12, 4))

        self._lb = tk.Listbox(
            self, bg=CLR_PANEL, fg=CLR_TEXT,
            selectbackground=CLR_ACCENT, selectforeground=CLR_BG,
            font=("Courier", 10), relief=tk.FLAT, bd=0,
            width=30, height=min(len(people), 10),
        )
        for p in people:
            self._lb.insert(tk.END, p)
        self._lb.pack(padx=16, pady=4)
        self._lb.bind("<Double-Button-1>", self._ok)

        btn_frame = tk.Frame(self, bg=CLR_BG)
        btn_frame.pack(pady=10)
        tk.Button(
            btn_frame, text="Select", command=self._ok,
            bg=CLR_ACCENT, fg=CLR_BG, font=("Inter", 10, "bold"),
            relief=tk.FLAT, padx=16, pady=4,
        ).pack(side=tk.LEFT, padx=6)
        tk.Button(
            btn_frame, text="Cancel", command=self.destroy,
            bg=CLR_BORDER, fg=CLR_TEXT, font=("Inter", 10),
            relief=tk.FLAT, padx=16, pady=4,
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
    root = tk.Tk()
    root.resizable(False, False)
    app = FaceUnlockApp(root, "🔐 Smart Door Access System")
    root.mainloop()