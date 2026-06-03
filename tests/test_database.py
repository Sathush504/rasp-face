"""
tests/test_database.py — Unit tests for FaceDatabase.
"""

import os
import tempfile
import unittest

import numpy as np

# Add parent directory to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import FaceDatabase, FaceRecord


def _random_enc():
    """Return a normalised random 128-D face encoding."""
    v = np.random.rand(128).astype(np.float64)
    return v / np.linalg.norm(v)


class TestFaceDatabase(unittest.TestCase):

    def setUp(self):
        # Use a temp file so tests are isolated
        self._tmp = tempfile.NamedTemporaryFile(suffix=".pickle", delete=False)
        self._tmp.close()
        os.unlink(self._tmp.name)   # Let FaceDatabase create it fresh
        self.db = FaceDatabase(self._tmp.name)

    def tearDown(self):
        if os.path.exists(self._tmp.name):
            os.unlink(self._tmp.name)

    def test_enroll_and_list(self):
        self.db.enroll("Alice", _random_enc())
        self.assertIn("Alice", self.db.list_people())
        self.assertEqual(self.db.person_count(), 1)

    def test_enroll_multiple_people(self):
        self.db.enroll("Alice", _random_enc())
        self.db.enroll("Bob", _random_enc())
        people = self.db.list_people()
        self.assertIn("Alice", people)
        self.assertIn("Bob", people)
        self.assertEqual(self.db.person_count(), 2)

    def test_enroll_multiple_encodings_same_person(self):
        for _ in range(3):
            self.db.enroll("Alice", _random_enc())
        encs, names = self.db.get_all_encodings_and_names()
        alice_count = names.count("Alice")
        self.assertEqual(alice_count, 3)

    def test_max_per_person_cap(self):
        for _ in range(10):
            result = self.db.enroll("Alice", _random_enc(), max_per_person=10)
            self.assertTrue(result)
        # 11th should fail
        result = self.db.enroll("Alice", _random_enc(), max_per_person=10)
        self.assertFalse(result)

    def test_remove_person(self):
        self.db.enroll("Alice", _random_enc())
        self.db.enroll("Bob", _random_enc())
        removed = self.db.remove_person("Alice")
        self.assertTrue(removed)
        self.assertNotIn("Alice", self.db.list_people())
        self.assertIn("Bob", self.db.list_people())

    def test_remove_nonexistent(self):
        result = self.db.remove_person("Ghost")
        self.assertFalse(result)

    def test_clear_all(self):
        self.db.enroll("Alice", _random_enc())
        self.db.enroll("Bob", _random_enc())
        self.db.clear_all()
        self.assertEqual(self.db.person_count(), 0)
        self.assertEqual(len(self.db.list_people()), 0)

    def test_persistence_across_instances(self):
        """Data written by one instance should be readable by another."""
        self.db.enroll("Alice", _random_enc())
        self.db.enroll("Bob", _random_enc())

        db2 = FaceDatabase(self._tmp.name)
        self.assertEqual(db2.person_count(), 2)
        self.assertIn("Alice", db2.list_people())
        self.assertIn("Bob", db2.list_people())

    def test_legacy_format_migration(self):
        """Old flat-list pickle format should be migrated transparently."""
        import pickle
        enc = _random_enc()
        legacy_data = {
            "encodings": [enc],
            "names": ["Legacy Person"],
        }
        with open(self._tmp.name, "wb") as f:
            pickle.dump(legacy_data, f)

        db = FaceDatabase(self._tmp.name)
        self.assertIn("Legacy Person", db.list_people())
        self.assertEqual(db.person_count(), 1)

    def test_blank_name_raises(self):
        with self.assertRaises(ValueError):
            self.db.enroll("", _random_enc())

    def test_get_encodings_empty_db(self):
        encs, names = self.db.get_all_encodings_and_names()
        self.assertEqual(encs, [])
        self.assertEqual(names, [])

    def test_encoding_count(self):
        self.db.enroll("A", _random_enc())
        self.db.enroll("A", _random_enc())
        self.db.enroll("B", _random_enc())
        self.assertEqual(self.db.encoding_count(), 3)


class TestFaceRecord(unittest.TestCase):

    def test_add_encoding(self):
        rec = FaceRecord(name="Test")
        enc = _random_enc()
        rec.add_encoding(enc)
        self.assertEqual(rec.encoding_count(), 1)
        self.assertTrue(np.allclose(rec.encodings[0], enc))


if __name__ == "__main__":
    unittest.main()
