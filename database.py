"""
database.py — Thread-safe face-encoding database with atomic persistence.

Stores and retrieves 128-D face embeddings (numpy arrays) alongside their
associated names. Persists to a local pickle file.  All public methods are
thread-safe via an internal RLock.
"""

import logging
import os
import pickle
import threading
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class FaceRecord:
    """A single enrolled person with one or more face encodings."""
    name: str
    encodings: List[np.ndarray] = field(default_factory=list)

    def add_encoding(self, enc: np.ndarray) -> None:
        self.encodings.append(enc)

    def encoding_count(self) -> int:
        return len(self.encodings)


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
class FaceDatabase:
    """
    Thread-safe, file-backed store for enrolled face encodings.

    The on-disk format is a dict ``{"records": [FaceRecord, ...]}``.
    Atomic writes (write-then-rename) protect against corruption on power loss.
    """

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._records: List[FaceRecord] = []
        self._lock = threading.RLock()
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _load(self) -> None:
        if not os.path.exists(self._db_path):
            logger.info("No database file found — starting fresh.")
            return
        try:
            with open(self._db_path, "rb") as fh:
                data = pickle.load(fh)

            # Support both old format {"encodings"/"names"} and new format
            if "records" in data:
                self._records = data["records"]
            elif "encodings" in data and "names" in data:
                self._migrate_legacy(data)
            else:
                raise ValueError("Unrecognised database format.")

            total = sum(r.encoding_count() for r in self._records)
            logger.info(
                "Database loaded: %d person(s), %d encoding(s).",
                len(self._records), total
            )
        except Exception as exc:
            logger.exception("Failed to load database from %s: %s", self._db_path, exc)

    def _migrate_legacy(self, data: dict) -> None:
        """Convert the old flat-list format to FaceRecord objects."""
        names = data["names"]
        encodings = data["encodings"]
        person_map: dict[str, FaceRecord] = {}
        for name, enc in zip(names, encodings):
            if name not in person_map:
                person_map[name] = FaceRecord(name=name)
            person_map[name].add_encoding(enc)
        self._records = list(person_map.values())
        logger.info("Migrated legacy database: %d person(s).", len(self._records))
        self._save()   # immediately save in new format

    def _save(self) -> None:
        """Atomically write the database to disk."""
        tmp_path = self._db_path + ".tmp"
        try:
            with open(tmp_path, "wb") as fh:
                pickle.dump({"records": self._records}, fh)
            os.replace(tmp_path, self._db_path)
            logger.debug("Database saved to %s.", self._db_path)
        except Exception as exc:
            logger.exception("Failed to save database: %s", exc)
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def enroll(self, name: str, encoding: np.ndarray,
               max_per_person: int = 10) -> bool:
        """
        Add a face encoding for *name*.

        Returns False if the person already has ``max_per_person`` encodings.
        """
        name = name.strip()
        if not name:
            raise ValueError("Name must not be blank.")
        with self._lock:
            record = self._find_record(name)
            if record is None:
                record = FaceRecord(name=name)
                self._records.append(record)
            if record.encoding_count() >= max_per_person:
                logger.warning(
                    "Enrollment rejected: %s already has %d encodings.",
                    name, record.encoding_count()
                )
                return False
            record.add_encoding(encoding)
            self._save()
            logger.info(
                "Enrolled encoding #%d for '%s'.", record.encoding_count(), name
            )
            return True

    def remove_person(self, name: str) -> bool:
        """Delete all encodings for *name*. Returns True if found."""
        name = name.strip()
        with self._lock:
            original_len = len(self._records)
            self._records = [r for r in self._records if r.name != name]
            if len(self._records) == original_len:
                return False
            self._save()
            logger.info("Removed profile '%s' from database.", name)
            return True

    def clear_all(self) -> None:
        """Wipe the entire database (in-memory and on-disk)."""
        with self._lock:
            self._records.clear()
            self._save()
            logger.warning("Database completely cleared.")

    def get_all_encodings_and_names(self):
        """
        Return flat lists ``(encodings, names)`` suitable for
        ``face_recognition.compare_faces``.
        """
        with self._lock:
            all_enc: List[np.ndarray] = []
            all_names: List[str] = []
            for record in self._records:
                for enc in record.encodings:
                    all_enc.append(enc)
                    all_names.append(record.name)
            return all_enc, all_names

    def list_people(self) -> List[str]:
        """Return sorted list of enrolled names."""
        with self._lock:
            return sorted({r.name for r in self._records})

    def person_count(self) -> int:
        with self._lock:
            return len(self._records)

    def encoding_count(self) -> int:
        with self._lock:
            return sum(r.encoding_count() for r in self._records)

    def _find_record(self, name: str) -> Optional[FaceRecord]:
        for r in self._records:
            if r.name == name:
                return r
        return None
