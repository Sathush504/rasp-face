"""
recognizer.py — Face recognition engine.

Wraps the ``face_recognition`` library with:
- Configurable tolerance and HOG/CNN model selection
- Consecutive-frame confirmation (anti-spoofing basic guard)
- Access-log event emission
"""

import datetime
import logging
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import face_recognition
import numpy as np

from config import FACE_MODEL, FACE_TOLERANCE, PROCESS_SCALE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
class AuthResult(Enum):
    AUTHORIZED = auto()     # known face matched
    UNKNOWN = auto()        # face detected but not in DB
    NO_FACE = auto()        # no face found in frame
    ERROR = auto()          # processing error


@dataclass
class RecognitionEvent:
    timestamp: datetime.datetime
    result: AuthResult
    name: Optional[str] = None                   # populated on AUTHORIZED
    confidence: Optional[float] = None           # 1 - face_distance (0-1)
    face_count: int = 0


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class FaceRecognizer:
    """
    Processes video frames and emits RecognitionEvent objects.

    Parameters
    ----------
    database : FaceDatabase
        Provides get_all_encodings_and_names().
    confirm_frames : int
        How many consecutive frames must match before AUTHORIZED is fired.
        Set to 1 to disable confirmation (faster, less safe).
    cooldown_sec : float
        Minimum seconds between consecutive unlock events for the same person.
    """

    def __init__(self, database, confirm_frames: int = 3,
                 cooldown_sec: float = 5.0):
        self._db = database
        self._confirm_frames = confirm_frames
        self._cooldown_sec = cooldown_sec
        self._lock = threading.Lock()

        # Consecutive-match tracking
        self._match_counter: Dict[str, int] = {}
        self._last_unlock_time: Dict[str, datetime.datetime] = {}

        # Registered callbacks
        self._event_callbacks: List[Callable[[RecognitionEvent], None]] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def add_event_callback(self, cb: Callable[[RecognitionEvent], None]) -> None:
        """Register a function to be called on every RecognitionEvent."""
        self._event_callbacks.append(cb)

    # ------------------------------------------------------------------
    # Main processing entry point
    # ------------------------------------------------------------------
    def process_frame(self, bgr_frame: np.ndarray) -> Tuple[
            List[Tuple[int, int, int, int]],
            List[str],
            Optional[RecognitionEvent]
    ]:
        """
        Analyse one BGR camera frame.

        Returns
        -------
        face_boxes : list of (top, right, bottom, left) in *original* resolution
        face_labels : list of names (or "Unknown")
        event : RecognitionEvent | None  — only set when a state change occurs
        """
        try:
            # --- Downscale for speed ---
            small = cv2.resize(bgr_frame, (0, 0),
                               fx=PROCESS_SCALE, fy=PROCESS_SCALE)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            locations = face_recognition.face_locations(rgb_small, model=FACE_MODEL)
            encodings = face_recognition.face_encodings(rgb_small, locations)

            if not locations:
                self._decay_counters()
                return [], [], None

            db_encodings, db_names = self._db.get_all_encodings_and_names()
            scale = int(1 / PROCESS_SCALE)

            face_boxes: List[Tuple[int, int, int, int]] = []
            face_labels: List[str] = []
            auth_event: Optional[RecognitionEvent] = None

            for loc, enc in zip(locations, encodings):
                top, right, bottom, left = (v * scale for v in loc)
                face_boxes.append((top, right, bottom, left))

                name = "Unknown"
                confidence: Optional[float] = None

                if db_encodings:
                    matches = face_recognition.compare_faces(
                        db_encodings, enc, tolerance=FACE_TOLERANCE
                    )
                    distances = face_recognition.face_distance(db_encodings, enc)
                    if True in matches:
                        best = int(np.argmin(distances))
                        if matches[best]:
                            name = db_names[best]
                            confidence = float(1.0 - distances[best])

                face_labels.append(name)

                # Confirmation logic
                if name != "Unknown":
                    event = self._handle_known_face(name, confidence)
                    if event:
                        auth_event = event
                else:
                    event = self._handle_unknown_face()
                    if event:
                        auth_event = event

            return face_boxes, face_labels, auth_event

        except Exception as exc:
            logger.exception("Frame processing error: %s", exc)
            event = RecognitionEvent(
                timestamp=datetime.datetime.now(),
                result=AuthResult.ERROR
            )
            self._dispatch(event)
            return [], [], event

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _handle_known_face(self, name: str,
                           confidence: float) -> Optional[RecognitionEvent]:
        with self._lock:
            self._match_counter[name] = self._match_counter.get(name, 0) + 1
            count = self._match_counter[name]

        if count < self._confirm_frames:
            logger.debug("Confirming match for '%s' (%d/%d).",
                         name, count, self._confirm_frames)
            return None

        # Reset counter so it doesn't fire every frame
        with self._lock:
            self._match_counter[name] = 0

        # Cooldown check
        now = datetime.datetime.now()
        with self._lock:
            last = self._last_unlock_time.get(name)
            if last and (now - last).total_seconds() < self._cooldown_sec:
                logger.debug("Cooldown active for '%s'.", name)
                return None
            self._last_unlock_time[name] = now

        event = RecognitionEvent(
            timestamp=now,
            result=AuthResult.AUTHORIZED,
            name=name,
            confidence=confidence,
            face_count=1
        )
        self._dispatch(event)
        return event

    def _handle_unknown_face(self) -> Optional[RecognitionEvent]:
        with self._lock:
            self._match_counter["Unknown"] = self._match_counter.get("Unknown", 0) + 1
            count = self._match_counter["Unknown"]

        if count < self._confirm_frames:
            return None

        # Reset counter
        with self._lock:
            self._match_counter["Unknown"] = 0

        # Cooldown check for unknown event (15 seconds)
        now = datetime.datetime.now()
        with self._lock:
            last = self._last_unlock_time.get("Unknown")
            if last and (now - last).total_seconds() < 15.0:
                return None
            self._last_unlock_time["Unknown"] = now

        event = RecognitionEvent(
            timestamp=now,
            result=AuthResult.UNKNOWN,
            face_count=1
        )
        self._dispatch(event)
        return event

    def _decay_counters(self) -> None:
        """Gradually reduce match counters when no face is visible."""
        with self._lock:
            to_delete = []
            for name in self._match_counter:
                self._match_counter[name] = max(
                    0, self._match_counter[name] - 1
                )
                if self._match_counter[name] == 0:
                    to_delete.append(name)
            for name in to_delete:
                del self._match_counter[name]

    def _dispatch(self, event: RecognitionEvent) -> None:
        for cb in self._event_callbacks:
            try:
                cb(event)
            except Exception:
                logger.exception("Error in recognition event callback.")
