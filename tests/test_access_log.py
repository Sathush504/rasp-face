"""
tests/test_access_log.py — Unit tests for AccessLogger.
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Patch config path before importing AccessLogger
import config
_tmp_log = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
_tmp_log.close()
config.LOG_FILE = _tmp_log.name

from access_log import AccessLogger


class TestAccessLogger(unittest.TestCase):

    def setUp(self):
        self._log_path = tempfile.NamedTemporaryFile(
            suffix=".jsonl", delete=False
        ).name
        self.logger = AccessLogger(log_path=self._log_path)

    def tearDown(self):
        if os.path.exists(self._log_path):
            os.unlink(self._log_path)

    def test_log_event_creates_file(self):
        self.logger.log_event("STARTUP", source="system")
        self.assertTrue(os.path.exists(self._log_path))

    def test_log_event_valid_json(self):
        self.logger.log_event("ACCESS_GRANTED", name="Alice", confidence=0.95, source="face")
        records = self.logger.read_recent(5)
        self.assertGreater(len(records), 0)
        r = records[0]
        self.assertEqual(r["event"], "ACCESS_GRANTED")
        self.assertEqual(r["name"], "Alice")
        self.assertAlmostEqual(r["confidence"], 0.95, places=3)

    def test_read_recent_order(self):
        """Most recent event should be first in read_recent()."""
        self.logger.log_event("STARTUP", source="system")
        self.logger.log_event("ACCESS_GRANTED", name="Alice", source="face")
        records = self.logger.read_recent(10)
        # read_recent returns newest first
        self.assertEqual(records[0]["event"], "ACCESS_GRANTED")

    def test_extra_fields_included(self):
        self.logger.log_event("ERROR", source="system", extra={"detail": "camera_fail"})
        records = self.logger.read_recent(1)
        self.assertIn("detail", records[0])
        self.assertEqual(records[0]["detail"], "camera_fail")

    def test_read_recent_empty_file(self):
        records = self.logger.read_recent(10)
        self.assertEqual(records, [])

    def test_multiple_events_all_logged(self):
        for i in range(5):
            self.logger.log_event("PING", extra={"seq": i}, source="test")
        records = self.logger.read_recent(10)
        self.assertEqual(len(records), 5)


if __name__ == "__main__":
    unittest.main()
