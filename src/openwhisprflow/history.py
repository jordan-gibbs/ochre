"""Local dictation history (text only; never audio). Written before insertion so nothing is lost
if focus changed mid-dictation (the save-before-insert rule)."""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from openwhisprflow.config import data_dir

SCHEMA = """
CREATE TABLE IF NOT EXISTS dictations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created REAL NOT NULL,
  raw TEXT NOT NULL,
  text TEXT NOT NULL,
  app TEXT NOT NULL DEFAULT '',
  stt TEXT NOT NULL DEFAULT '',
  refiner TEXT NOT NULL DEFAULT '',
  duration_ms INTEGER NOT NULL DEFAULT 0,
  inserted INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS dictations_created ON dictations(created);
"""


@dataclass
class Entry:
    id: int
    created: float
    raw: str
    text: str
    app: str
    stt: str
    refiner: str
    duration_ms: int
    inserted: bool


class History:
    def __init__(self, path: Path | None = None, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self.path = path or data_dir() / "history.sqlite3"
        self._lock = threading.Lock()
        self._db: sqlite3.Connection | None = None
        if enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(self.path, check_same_thread=False)
            self._db.executescript(SCHEMA)

    def add(self, *, raw: str, text: str, app: str = "", stt: str = "", refiner: str = "",
            duration_ms: int = 0) -> int | None:
        if not self._db:
            return None
        with self._lock, self._db:
            cur = self._db.execute(
                "INSERT INTO dictations(created, raw, text, app, stt, refiner, duration_ms) VALUES (?,?,?,?,?,?,?)",
                (time.time(), raw, text, app, stt, refiner, duration_ms))
            return cur.lastrowid

    def mark_inserted(self, entry_id: int | None) -> None:
        if self._db and entry_id is not None:
            with self._lock, self._db:
                self._db.execute("UPDATE dictations SET inserted = 1 WHERE id = ?", (entry_id,))

    def query(self, q: str = "", limit: int = 50) -> list[Entry]:
        if not self._db:
            return []
        sql = "SELECT id, created, raw, text, app, stt, refiner, duration_ms, inserted FROM dictations"
        args: tuple = ()
        if q:
            sql += " WHERE text LIKE ? OR raw LIKE ?"
            args = (f"%{q}%", f"%{q}%")
        sql += " ORDER BY id DESC LIMIT ?"
        with self._lock:
            rows = self._db.execute(sql, (*args, max(1, min(limit, 500)))).fetchall()
        return [Entry(*r[:8], bool(r[8])) for r in rows]

    def prune(self, keep_days: int) -> None:
        if self._db and keep_days > 0:
            with self._lock, self._db:
                self._db.execute("DELETE FROM dictations WHERE created < ?", (time.time() - keep_days * 86400,))

    def clear(self) -> None:
        if self._db:
            with self._lock, self._db:
                self._db.execute("DELETE FROM dictations")

    def close(self) -> None:
        if self._db:
            self._db.close()
            self._db = None


def as_dicts(entries: list[Entry]) -> list[dict]:
    return [asdict(e) for e in entries]
