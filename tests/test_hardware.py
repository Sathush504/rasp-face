"""
tests/test_hardware.py — Unit tests for DoorLock (uses stub GPIO on non-Pi).
"""

import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hardware import DoorLock


class TestDoorLock(unittest.TestCase):

    def _make_lock(self, duration=0.2):
        """Create a DoorLock on a fake pin with short unlock duration."""
        return DoorLock(pin=18, active_high=True, unlock_duration=duration, remote_ip="")

    def test_initial_state_is_locked(self):
        lock = self._make_lock()
        self.assertFalse(lock.is_unlocked)
        lock.cleanup()

    def test_unlock_changes_state(self):
        lock = self._make_lock()
        lock.unlock(triggered_by="test")
        self.assertTrue(lock.is_unlocked)
        lock.cleanup()

    def test_auto_relock_after_duration(self):
        lock = self._make_lock(duration=0.15)
        lock.unlock(triggered_by="test")
        self.assertTrue(lock.is_unlocked)
        time.sleep(0.3)
        self.assertFalse(lock.is_unlocked)
        lock.cleanup()

    def test_manual_lock(self):
        lock = self._make_lock()
        lock.unlock(triggered_by="test")
        self.assertTrue(lock.is_unlocked)
        lock.lock()
        self.assertFalse(lock.is_unlocked)
        lock.cleanup()

    def test_state_change_callback_on_unlock(self):
        states = []
        lock = self._make_lock(duration=0.15)
        lock.set_state_change_callback(states.append)
        lock.unlock(triggered_by="test")
        time.sleep(0.3)   # wait for auto-relock
        lock.cleanup()
        # Should have fired at least unlock (True) and relock (False)
        self.assertIn(True, states)
        self.assertIn(False, states)

    def test_multiple_unlock_calls_reset_timer(self):
        """Second unlock before auto-relock should reset the timer."""
        lock = self._make_lock(duration=0.25)
        lock.unlock(triggered_by="test1")
        time.sleep(0.15)
        lock.unlock(triggered_by="test2")   # reset timer
        time.sleep(0.15)
        # Still unlocked because timer was reset
        self.assertTrue(lock.is_unlocked)
        time.sleep(0.15)
        # Now should have relocked
        self.assertFalse(lock.is_unlocked)
        lock.cleanup()

    def test_cleanup_does_not_raise(self):
        lock = self._make_lock()
        try:
            lock.cleanup()
        except Exception as e:
            self.fail(f"cleanup() raised an exception: {e}")


if __name__ == "__main__":
    unittest.main()
