"""
access_log.py — Structured, rotating access-event logger.

Each event is written as a newline-delimited JSON (JSONL) record so the
log file can be streamed, grepped, and parsed by external tools without
loading the entire file.
"""

import json
import logging
import os
import threading
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional

from config import LOG_BACKUP_COUNT, LOG_FILE, LOG_MAX_BYTES

logger = logging.getLogger(__name__)


class AccessLogger:
    """
    Writes structured access events to a rotating JSONL file.

    Thread-safe — callers from the GUI thread and the Blynk thread can
    both call ``log_event`` safely.
    """

    def __init__(self, log_path: str = LOG_FILE):
        self._path = log_path
        self._lock = threading.Lock()
        self._handler = self._build_handler()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def log_event(
        self,
        event_type: str,
        *,
        name: Optional[str] = None,
        confidence: Optional[float] = None,
        source: str = "face",
        extra: Optional[dict] = None
    ) -> None:
        """
        Write a structured event record to the log file.

        Parameters
        ----------
        event_type : str
            One of: ACCESS_GRANTED, ACCESS_DENIED, REMOTE_UNLOCK,
            ENROLLMENT, REMOVAL, STARTUP, SHUTDOWN, ERROR.
        name : str | None
            Person name (if applicable).
        confidence : float | None
            Face-match confidence (0–1).
        source : str
            "face" | "blynk" | "admin" | "system".
        extra : dict | None
            Any additional key-value pairs to include.
        """
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "event": event_type,
            "source": source,
        }
        if name is not None:
            record["name"] = name
        if confidence is not None:
            record["confidence"] = round(confidence, 4)
        if extra:
            record.update(extra)

        line = json.dumps(record)
        with self._lock:
            try:
                self._handler.stream.write(line + "\n")
                self._handler.stream.flush()
                self._handler.doRollover() if self._should_rollover() else None
            except Exception as exc:
                logger.warning("AccessLogger write failed: %s", exc)

        logger.info("[ACCESS LOG] %s", line)

    def close(self) -> None:
        """Release underlying file handler resources."""
        with self._lock:
            try:
                self._handler.close()
            except Exception:
                pass

    def read_recent(self, n: int = 20) -> list:
        """Return the last *n* log records as parsed dicts."""
        records = []
        try:
            with self._lock:
                with open(self._path, "r") as fh:
                    lines = fh.readlines()
            for line in reversed(lines):
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
                if len(records) >= n:
                    break
        except FileNotFoundError:
            pass
        return records

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _build_handler(self) -> RotatingFileHandler:
        return RotatingFileHandler(
            self._path,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )

    def _should_rollover(self) -> bool:
        """Check if current log file exceeds max size."""
        try:
            return os.path.getsize(self._path) >= LOG_MAX_BYTES
        except OSError:
            return False
