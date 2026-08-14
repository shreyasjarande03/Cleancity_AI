from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "cleancity.db"


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL,
            phone TEXT,
            latitude REAL,
            longitude REAL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS complaints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            complaint_id TEXT UNIQUE NOT NULL,
            user_id INTEGER NOT NULL,
            image_url TEXT,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            waste_type TEXT,
            severity TEXT,
            status TEXT NOT NULL DEFAULT 'Pending',
            created_at TEXT NOT NULL,
            collector_id INTEGER,
            before_image TEXT,
            after_image TEXT,
            verified INTEGER DEFAULT 0
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS collectors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            phone TEXT,
            latitude REAL,
            longitude REAL,
            availability TEXT NOT NULL DEFAULT 'Available'
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS resolutions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            complaint_id TEXT NOT NULL,
            before_image TEXT,
            after_image TEXT,
            completed_time TEXT,
            verified INTEGER DEFAULT 0
        )
        """
    )

    conn.execute(
        "INSERT OR IGNORE INTO users (id, name, email, password, role, phone, latitude, longitude) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (1, "Admin User", "admin@cleancity.ai", "admin123", "admin", "9876543210", 12.9716, 77.5946),
    )
    conn.execute(
        "INSERT OR IGNORE INTO users (id, name, email, password, role, phone, latitude, longitude) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (2, "Collector One", "collector@cleancity.ai", "collector123", "collector", "9876543211", 12.9725, 77.5955),
    )
    conn.execute(
        "INSERT OR IGNORE INTO users (id, name, email, password, role, phone, latitude, longitude) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (3, "Citizen User", "citizen@cleancity.ai", "citizen123", "citizen", "9876543212", 12.9698, 77.5932),
    )
    conn.execute(
        "INSERT OR IGNORE INTO collectors (id, name, phone, latitude, longitude, availability) VALUES (?, ?, ?, ?, ?, ?)",
        (1, "Collector A", "9876500001", 12.9722, 77.5940, "Available"),
    )
    conn.execute(
        "INSERT OR IGNORE INTO collectors (id, name, phone, latitude, longitude, availability) VALUES (?, ?, ?, ?, ?, ?)",
        (2, "Collector B", "9876500002", 12.9660, 77.6010, "Available"),
    )
    conn.execute(
        "INSERT OR IGNORE INTO collectors (id, name, phone, latitude, longitude, availability) VALUES (?, ?, ?, ?, ?, ?)",
        (3, "Collector C", "9876500003", 12.9738, 77.5968, "Available"),
    )
    conn.commit()
    conn.close()


def get_user_by_email(email: str):
    conn = get_connection()
    user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    conn.close()
    return dict(user) if user else None


def get_all_complaints():
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM complaints ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def save_complaint(payload: dict) -> dict:
    conn = get_connection()
    cursor = conn.execute(
        """
        INSERT INTO complaints (
            complaint_id, user_id, image_url, latitude, longitude,
            waste_type, severity, status, created_at, collector_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            payload["complaint_id"],
            payload["user_id"],
            payload.get("image_url", ""),
            payload["latitude"],
            payload["longitude"],
            payload.get("waste_type", "Mixed"),
            payload.get("severity", "Medium"),
            payload.get("status", "Pending"),
            payload["created_at"],
            payload.get("collector_id"),
        ),
    )
    conn.commit()
    complaint = conn.execute("SELECT * FROM complaints WHERE id = ?", (cursor.lastrowid,)).fetchone()
    conn.close()
    return dict(complaint)


def assign_collector(latitude: float, longitude: float):
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, name, latitude, longitude FROM collectors WHERE availability = 'Available'"
    ).fetchall()
    conn.close()

    if not rows:
        return None

    def distance(lat1, lon1, lat2, lon2):
        return ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5

    nearest = min(rows, key=lambda row: distance(latitude, longitude, row["latitude"], row["longitude"]))
    return dict(nearest)
