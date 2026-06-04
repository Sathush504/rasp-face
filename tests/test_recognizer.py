"""
tests/test_recognizer.py — Unit tests for FaceRecognizer.

These tests use synthetic numpy encodings (no real camera or images needed)
so they run on any CI/dev machine.
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Skip entire module if core vision libraries aren't installed
try:
    import cv2  # noqa: F401
    import face_recognition  # noqa: F401
except ImportError:
    import unittest
    raise unittest.SkipTest(
        "cv2 / face_recognition not installed — skipping recognizer tests."
    )

from database import FaceDatabase
from recognizer import AuthResult, FaceRecognizer, RecognitionEvent


def _enc(value: float) -> np.ndarray:
    """Create a 128-D unit vector populated with a single repeating value."""
    v = np.full(128, value, dtype=np.float64)
    return v / np.linalg.norm(v)


ALICE_ENC = _enc(0.1)
BOB_ENC = _enc(0.9)
UNKNOWN_ENC = _enc(0.5)


class TestFaceRecognizer(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".pickle", delete=False)
        self._tmp.close()
        os.unlink(self._tmp.name)
        self.db = FaceDatabase(self._tmp.name)
        self.db.enroll("Alice", ALICE_ENC)
        self.db.enroll("Bob", BOB_ENC)
        self._patcher = patch("recognizer.LIVENESS_ENABLED", False)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        if os.path.exists(self._tmp.name):
            os.unlink(self._tmp.name)

    @patch("recognizer.face_recognition")
    def test_authorized_after_confirm_frames(self, mock_fr):
        """AUTHORIZED event fires only after confirm_frames consecutive matches."""
        mock_fr.face_locations.return_value = [(0, 100, 100, 0)]
        mock_fr.face_encodings.return_value = [ALICE_ENC]
        mock_fr.compare_faces.return_value = [True, False]
        mock_fr.face_distance.return_value = np.array([0.3, 0.8])

        events = []
        recognizer = FaceRecognizer(self.db, confirm_frames=3, cooldown_sec=0)
        recognizer.add_event_callback(events.append)

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Frames 1 & 2 — no event yet
        for _ in range(2):
            recognizer.process_frame(dummy_frame)
        self.assertEqual(len([e for e in events if e.result == AuthResult.AUTHORIZED]), 0)

        # Frame 3 — AUTHORIZED should fire
        recognizer.process_frame(dummy_frame)
        authorized = [e for e in events if e.result == AuthResult.AUTHORIZED]
        self.assertEqual(len(authorized), 1)
        self.assertEqual(authorized[0].name, "Alice")

    @patch("recognizer.face_recognition")
    def test_cooldown_prevents_repeated_unlock(self, mock_fr):
        """Second match within cooldown window should NOT fire AUTHORIZED again."""
        mock_fr.face_locations.return_value = [(0, 100, 100, 0)]
        mock_fr.face_encodings.return_value = [ALICE_ENC]
        mock_fr.compare_faces.return_value = [True, False]
        mock_fr.face_distance.return_value = np.array([0.3, 0.8])

        events = []
        recognizer = FaceRecognizer(self.db, confirm_frames=1, cooldown_sec=999)
        recognizer.add_event_callback(events.append)

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        recognizer.process_frame(dummy_frame)   # fires
        recognizer.process_frame(dummy_frame)   # should be blocked by cooldown
        recognizer.process_frame(dummy_frame)

        authorized = [e for e in events if e.result == AuthResult.AUTHORIZED]
        self.assertEqual(len(authorized), 1)

    @patch("recognizer.face_recognition")
    def test_no_face_returns_empty(self, mock_fr):
        mock_fr.face_locations.return_value = []
        mock_fr.face_encodings.return_value = []

        recognizer = FaceRecognizer(self.db, confirm_frames=1, cooldown_sec=0)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        boxes, labels, event = recognizer.process_frame(dummy_frame)

        self.assertEqual(boxes, [])
        self.assertEqual(labels, [])
        self.assertIsNone(event)

    @patch("recognizer.face_recognition")
    def test_unknown_face_emits_unknown_event(self, mock_fr):
        mock_fr.face_locations.return_value = [(0, 100, 100, 0)]
        mock_fr.face_encodings.return_value = [UNKNOWN_ENC]
        mock_fr.compare_faces.return_value = [False, False]
        mock_fr.face_distance.return_value = np.array([0.7, 0.7])

        events = []
        recognizer = FaceRecognizer(self.db, confirm_frames=1, cooldown_sec=0)
        recognizer.add_event_callback(events.append)

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        recognizer.process_frame(dummy_frame)

        unknown_events = [e for e in events if e.result == AuthResult.UNKNOWN]
        self.assertGreater(len(unknown_events), 0)
        self.assertIsNone(unknown_events[0].name)

    @patch("recognizer.face_recognition")
    def test_multiple_callbacks_all_called(self, mock_fr):
        mock_fr.face_locations.return_value = [(0, 100, 100, 0)]
        mock_fr.face_encodings.return_value = [ALICE_ENC]
        mock_fr.compare_faces.return_value = [True, False]
        mock_fr.face_distance.return_value = np.array([0.2, 0.8])

        cb1, cb2 = MagicMock(), MagicMock()
        recognizer = FaceRecognizer(self.db, confirm_frames=1, cooldown_sec=0)
        recognizer.add_event_callback(cb1)
        recognizer.add_event_callback(cb2)

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        recognizer.process_frame(dummy_frame)

        cb1.assert_called()
        cb2.assert_called()


if __name__ == "__main__":
    unittest.main()
