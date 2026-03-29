import sqlite3
import logging
import os
import numpy as np
from datetime import datetime
from config import SYSTEM_CONFIG


class DatabaseManager:
    def __init__(self):
        self.logger = logging.getLogger("DatabaseManager")
        self.db_path = SYSTEM_CONFIG['DATABASE_PATH']
        self._init_db()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    id         INTEGER PRIMARY KEY,
                    name       TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS fingerprint_templates (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id    INTEGER NOT NULL,
                    vector     BLOB NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                );

                CREATE TABLE IF NOT EXISTS access_logs (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id     INTEGER,
                    cnn_score   REAL,
                    hog_score   REAL,
                    granted     INTEGER NOT NULL,
                    conflict    INTEGER NOT NULL DEFAULT 0,
                    timestamp   TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                );
            """)
        self.logger.info(f"Database ready at {self.db_path}")

    # ------------------------------------------------------------------
    # Users
    # ------------------------------------------------------------------

    def add_user(self, user_id: int, name: str) -> bool:
        try:
            with self._get_conn() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO users (id, name, created_at) VALUES (?, ?, ?)",
                    (user_id, name, datetime.now().isoformat()),
                )
            self.logger.info(f"User added: id={user_id}, name={name}")
            return True
        except sqlite3.Error as e:
            self.logger.error(f"add_user failed: {e}")
            return False

    def get_user(self, user_id: int):
        try:
            with self._get_conn() as conn:
                row = conn.execute(
                    "SELECT * FROM users WHERE id = ?", (user_id,)
                ).fetchone()
            return dict(row) if row else None
        except sqlite3.Error as e:
            self.logger.error(f"get_user failed: {e}")
            return None

    def list_users(self):
        try:
            with self._get_conn() as conn:
                rows = conn.execute("SELECT * FROM users ORDER BY id").fetchall()
            return [dict(r) for r in rows]
        except sqlite3.Error as e:
            self.logger.error(f"list_users failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Fingerprint templates
    # ------------------------------------------------------------------

    def add_template(self, user_id: int, vector: np.ndarray) -> bool:
        try:
            blob = vector.astype(np.float32).tobytes()
            with self._get_conn() as conn:
                conn.execute(
                    "INSERT INTO fingerprint_templates (user_id, vector, created_at) VALUES (?, ?, ?)",
                    (user_id, blob, datetime.now().isoformat()),
                )
            self.logger.info(f"Template stored for user_id={user_id}, dim={vector.shape}")
            return True
        except sqlite3.Error as e:
            self.logger.error(f"add_template failed: {e}")
            return False

    def get_templates(self) -> list[tuple[int, np.ndarray]]:
        """Returns list of (user_id, vector) for all enrolled templates."""
        try:
            with self._get_conn() as conn:
                rows = conn.execute(
                    "SELECT user_id, vector FROM fingerprint_templates"
                ).fetchall()
            return [
                (row["user_id"], np.frombuffer(row["vector"], dtype=np.float32))
                for row in rows
            ]
        except sqlite3.Error as e:
            self.logger.error(f"get_templates failed: {e}")
            return []

    def delete_templates(self, user_id: int) -> bool:
        try:
            with self._get_conn() as conn:
                conn.execute(
                    "DELETE FROM fingerprint_templates WHERE user_id = ?", (user_id,)
                )
            self.logger.info(f"Templates deleted for user_id={user_id}")
            return True
        except sqlite3.Error as e:
            self.logger.error(f"delete_templates failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Access logs
    # ------------------------------------------------------------------

    def log_access(
        self,
        user_id,
        cnn_score: float,
        hog_score: float,
        granted: bool,
        conflict: bool = False,
    ) -> bool:
        try:
            with self._get_conn() as conn:
                conn.execute(
                    """INSERT INTO access_logs
                       (user_id, cnn_score, hog_score, granted, conflict, timestamp)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        user_id,
                        cnn_score,
                        hog_score,
                        int(granted),
                        int(conflict),
                        datetime.now().isoformat(),
                    ),
                )
            status = "GRANTED" if granted else "DENIED"
            flag = " [CONFLICT]" if conflict else ""
            self.logger.info(
                f"Access {status}{flag} — user={user_id}, "
                f"cnn={cnn_score:.3f}, hog={hog_score:.3f}"
            )
            return True
        except sqlite3.Error as e:
            self.logger.error(f"log_access failed: {e}")
            return False

    def get_logs(self, limit: int = 100):
        try:
            with self._get_conn() as conn:
                rows = conn.execute(
                    """SELECT l.id, l.timestamp, l.cnn_score, l.hog_score,
                              l.granted, l.conflict, u.name AS user_name
                       FROM access_logs l
                       LEFT JOIN users u ON l.user_id = u.id
                       ORDER BY l.id DESC
                       LIMIT ?""",
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.Error as e:
            self.logger.error(f"get_logs failed: {e}")
            return []
