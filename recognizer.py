"""
recognizer.py — Face recognition engine.

Wraps the ``face_recognition`` library with:
- Configurable tolerance and HOG/CNN model selection
- Consecutive-frame confirmation (anti-spoofing basic guard)
- Access-log event emission
"""

import datetime
import logging
import math
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import face_recognition
import numpy as np

from config import FACE_MODEL, FACE_TOLERANCE, PROCESS_SCALE, LIVENESS_ENABLED, EYE_AR_THRESH

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
class AuthResult(Enum):
    AUTHORIZED = auto()     # known face matched
    UNKNOWN = auto()        # face detected but not in DB
    NO_FACE = auto()        # no face found in frame
    ERROR = auto()          # processing error


@dataclass(slots=True)
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
        self._blink_state: Dict[str, str] = {}
        self._blink_counted: Dict[str, bool] = {}
        self._open_ear_history: Dict[str, List[float]] = {}
        self._dynamic_threshold: Dict[str, float] = {}

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

            scale = int(1 / PROCESS_SCALE)
            db_encodings, db_names = self._db.get_all_encodings_and_names()

            face_boxes: List[Tuple[int, int, int, int]] = []
            face_labels: List[str] = []
            auth_event: Optional[RecognitionEvent] = None

            for idx, (loc, enc) in enumerate(zip(locations, encodings)):
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

                # Blink detection logic for known faces (only run if liveness check is required)
                if LIVENESS_ENABLED and name != "Unknown" and not self._blink_counted.get(name, False):
                    rgb_full = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                    rgb_full = self._apply_clahe(rgb_full)  # Task 2: Apply adaptive lighting (CLAHE)
                    
                    full_loc = [(int(loc[0] * scale), int(loc[1] * scale), int(loc[2] * scale), int(loc[3] * scale))]
                    landmarks_list = face_recognition.face_landmarks(rgb_full, full_loc)
                    
                    if landmarks_list:
                        landmarks = landmarks_list[0]
                        ear = self._calculate_ear(landmarks)
                        
                        # Task 1: Dynamic EAR Calibration
                        history = self._open_ear_history.setdefault(name, [])
                        if len(history) < 5:
                            if ear > 0.12:
                                history.append(ear)
                            if len(history) == 5:
                                baseline = sum(history) / 5.0
                                # Set threshold to 72% of baseline, capped within reasonable bounds
                                self._dynamic_threshold[name] = float(max(0.15, min(0.24, baseline * 0.72)))
                                logger.info("✓ Dynamic EAR Calibration complete for '%s': Baseline=%.3f, Threshold=%.3f",
                                            name, baseline, self._dynamic_threshold[name])
                        else:
                            thresh = self._dynamic_threshold.get(name, EYE_AR_THRESH)
                            logger.info("Liveness [%s] -> EAR: %.3f (Threshold: %.2f)", name, ear, thresh)
                            
                            # Track state machine
                            current_state = self._blink_state.get(name, "OPEN")
                            if ear < thresh:  # Closed threshold
                                self._blink_state[name] = "CLOSED"
                            elif ear >= thresh and current_state == "CLOSED":
                                self._blink_state[name] = "OPEN"
                                self._blink_counted[name] = True
                                logger.info("✓ Liveness confirmed: Blink detected for '%s'!", name)

                # Set label text
                label_name = name
                if name != "Unknown" and LIVENESS_ENABLED:
                    if not self._blink_counted.get(name, False):
                        if len(self._open_ear_history.get(name, [])) < 5:
                            label_name = f"{name} (Calibrating...)"
                        else:
                            label_name = f"{name} (Blink to Unlock)"
                face_labels.append(label_name)

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

        # Check if they have blinked (liveness check)
        if LIVENESS_ENABLED and not self._blink_counted.get(name, False):
            # Keep match counter at confirmation threshold so we don't drop out of matching
            with self._lock:
                self._match_counter[name] = self._confirm_frames
            return None

        # Reset counter and blink state so it doesn't fire every frame
        with self._lock:
            self._match_counter[name] = 0
            self._blink_counted[name] = False

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
                if name in self._blink_state:
                    del self._blink_state[name]
                if name in self._blink_counted:
                    del self._blink_counted[name]

    def _calculate_ear(self, landmarks: dict) -> float:
        left_eye = landmarks.get("left_eye")
        right_eye = landmarks.get("right_eye")
        if not left_eye or not right_eye or len(left_eye) < 6 or len(right_eye) < 6:
            return 0.0

        # Optimize by using C-level math.hypot to calculate distances without function/numpy overhead
        l1_5 = math.hypot(left_eye[1][0] - left_eye[5][0], left_eye[1][1] - left_eye[5][1])
        l2_4 = math.hypot(left_eye[2][0] - left_eye[4][0], left_eye[2][1] - left_eye[4][1])
        l0_3 = math.hypot(left_eye[0][0] - left_eye[3][0], left_eye[0][1] - left_eye[3][1])

        r1_5 = math.hypot(right_eye[1][0] - right_eye[5][0], right_eye[1][1] - right_eye[5][1])
        r2_4 = math.hypot(right_eye[2][0] - right_eye[4][0], right_eye[2][1] - right_eye[4][1])
        r0_3 = math.hypot(right_eye[0][0] - right_eye[3][0], right_eye[0][1] - right_eye[3][1])

        if l0_3 == 0.0 or r0_3 == 0.0:
            return 0.0

        ear_left = (l1_5 + l2_4) / (2.0 * l0_3)
        ear_right = (r1_5 + r2_4) / (2.0 * r0_3)

        return (ear_left + ear_right) / 2.0

    def _apply_clahe(self, rgb_img: np.ndarray) -> np.ndarray:
        try:
            lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        except Exception:
            return rgb_img

    def _dispatch(self, event: RecognitionEvent) -> None:
        for cb in self._event_callbacks:
            try:
                cb(event)
            except Exception:
                logger.exception("Error in recognition event callback.")
