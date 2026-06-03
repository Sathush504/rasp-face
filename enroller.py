"""
enroller.py — Multi-sample face enrollment workflow.

Captures MIN_ENCODINGS_PER_PERSON frames of a face and stores the average
(or all individual) encodings for more robust later matching.
"""

import logging
import time
from typing import Optional

import cv2
import face_recognition
import numpy as np

from config import FACE_MODEL, MIN_ENCODINGS_PER_PERSON, PROCESS_SCALE
from database import FaceDatabase

logger = logging.getLogger(__name__)


class EnrollmentSession:
    """
    State-machine for guiding an operator through enrolling a new face.

    Usage (called from the GUI):
        session = EnrollmentSession(name, db)
        for each camera frame:
            result = session.feed_frame(bgr_frame)
            if result.done:
                break
    """

    class Result:
        __slots__ = ("done", "progress", "message", "success")

        def __init__(self, done=False, progress=0, message="", success=False):
            self.done = done
            self.progress = progress
            self.message = message
            self.success = success

    def __init__(self, name: str, db: FaceDatabase,
                 samples_required: int = MIN_ENCODINGS_PER_PERSON):
        self.name = name.strip()
        self._db = db
        self._samples_required = samples_required
        self._collected: list[np.ndarray] = []
        self._last_sample_time = 0.0

    @property
    def progress(self) -> int:
        return len(self._collected)

    def feed_frame(self, bgr_frame: np.ndarray) -> "EnrollmentSession.Result":
        """
        Process one camera frame. Returns a Result indicating progress.

        A minimum of 0.5s is enforced between samples so consecutive
        near-identical frames don't all count as one sample.
        """
        now = time.monotonic()
        if now - self._last_sample_time < 0.5:
            pct = int(100 * len(self._collected) / self._samples_required)
            return self.Result(
                progress=pct,
                message=f"Hold still… ({len(self._collected)}/{self._samples_required})"
            )

        small = cv2.resize(bgr_frame, (0, 0),
                           fx=PROCESS_SCALE, fy=PROCESS_SCALE)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        locations = face_recognition.face_locations(rgb, model=FACE_MODEL)

        if len(locations) == 0:
            return self.Result(message="No face detected — look at the camera.")

        if len(locations) > 1:
            return self.Result(message="Multiple faces — only one person please.")

        encodings = face_recognition.face_encodings(rgb, locations)
        if not encodings:
            return self.Result(message="Could not encode face — try again.")

        self._collected.append(encodings[0])
        self._last_sample_time = now

        if len(self._collected) < self._samples_required:
            pct = int(100 * len(self._collected) / self._samples_required)
            return self.Result(
                progress=pct,
                message=f"Capturing… ({len(self._collected)}/{self._samples_required})"
            )

        # All samples captured — compute mean encoding and store
        return self._commit()

    def _commit(self) -> "EnrollmentSession.Result":
        """Average collected encodings and save to database."""
        try:
            mean_enc = np.mean(self._collected, axis=0)
            success = self._db.enroll(self.name, mean_enc)
            if success:
                logger.info(
                    "Enrollment complete for '%s' (%d samples averaged).",
                    self.name, len(self._collected)
                )
                return self.Result(
                    done=True, progress=100, success=True,
                    message=f"✓ '{self.name}' enrolled successfully!"
                )
            else:
                return self.Result(
                    done=True, success=False,
                    message=f"'{self.name}' already has the maximum number of encodings."
                )
        except Exception as exc:
            logger.exception("Enrollment commit failed: %s", exc)
            return self.Result(
                done=True, success=False,
                message=f"Enrollment failed: {exc}"
            )
