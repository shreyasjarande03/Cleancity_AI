from __future__ import annotations

import math
import sqlite3
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "cleancity.db"
DUPLICATE_RADIUS_KM = 0.05  # ~50 meters


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


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
            verified INTEGER DEFAULT 0,
            duplicate_of TEXT,
            report_count INTEGER DEFAULT 1,
            priority_score REAL DEFAULT 0.5,
            ai_confidence REAL,
            description TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS collectors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
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
            verified INTEGER DEFAULT 0,
            verification_score REAL
        )
        """
    )

    _migrate_columns(conn)
    _seed_data(conn)
    _reassign_orphaned_complaints(conn)
    conn.commit()
    conn.close()


def _migrate_columns(conn: sqlite3.Connection) -> None:
    existing = {row[1] for row in conn.execute("PRAGMA table_info(complaints)").fetchall()}
    migrations = {
        "duplicate_of": "ALTER TABLE complaints ADD COLUMN duplicate_of TEXT",
        "report_count": "ALTER TABLE complaints ADD COLUMN report_count INTEGER DEFAULT 1",
        "priority_score": "ALTER TABLE complaints ADD COLUMN priority_score REAL DEFAULT 0.5",
        "ai_confidence": "ALTER TABLE complaints ADD COLUMN ai_confidence REAL",
        "description": "ALTER TABLE complaints ADD COLUMN description TEXT",
    }
    collector_cols = {row[1] for row in conn.execute("PRAGMA table_info(collectors)").fetchall()}
    if "user_id" not in collector_cols:
        conn.execute("ALTER TABLE collectors ADD COLUMN user_id INTEGER")
    for column, sql in migrations.items():
        if column not in existing:
            conn.execute(sql)


def _seed_data(conn: sqlite3.Connection) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO users (id, name, email, password, role, phone, latitude, longitude) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (1, "Admin User", "admin@cleancity.ai", "admin123", "admin", "9876543210", 12.9716, 77.5946),
    )
    conn.execute(
        "INSERT OR IGNORE INTO users (id, name, email, password, role, phone, latitude, longitude) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (2, "Collector One", "collector@cleancity.ai", "collector123", "collector", "9876543211", 12.9725, 77.5955),
    )
    conn.execute(
        "INSERT OR IGNORE INTO users (id, name, email, password, role, phone, latitude, longitude) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (3, "Citizen User", "citizen@cleancity.ai", "citizen123", "citizen", "9876543212", 12.9698, 77.5932),
    )
    collectors = [
        (1, 2, "Collector A", "9876500001", 12.9722, 77.5940, "Available"),
        (2, None, "Collector B", "9876500002", 12.9660, 77.6010, "Available"),
        (3, None, "Collector C", "9876500003", 12.9738, 77.5968, "Available"),
        (4, None, "Collector D", "9876500004", 12.9685, 77.5980, "Available"),
        (5, None, "Collector E", "9876500005", 12.9750, 77.5920, "Available"),
    ]
    for row in collectors:
        conn.execute(
            "INSERT OR IGNORE INTO collectors (id, user_id, name, phone, latitude, longitude, availability) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            row,
        )
    conn.execute("UPDATE collectors SET user_id = 2 WHERE id = 1 AND user_id IS NULL")


def get_user_by_email(email: str) -> dict[str, Any] | None:
    conn = get_connection()
    user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    conn.close()
    return dict(user) if user else None


def get_user_by_id(user_id: int) -> dict[str, Any] | None:
    conn = get_connection()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return dict(user) if user else None


def get_collector_by_user_email(email: str) -> dict[str, Any] | None:
    user = get_user_by_email(email)
    if not user or user["role"] != "collector":
        return None
    conn = get_connection()
    row = conn.execute("SELECT * FROM collectors WHERE user_id = ?", (user["id"],)).fetchone()
    conn.close()
    if row:
        return dict(row)
    conn = get_connection()
    row = conn.execute("SELECT * FROM collectors ORDER BY id LIMIT 1").fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_complaints() -> list[dict[str, Any]]:
    conn = get_connection()
    rows = conn.execute("SELECT * FROM complaints ORDER BY priority_score DESC, created_at DESC").fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_complaints_for_citizen(user_id: int) -> list[dict[str, Any]]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM complaints WHERE user_id = ? ORDER BY created_at DESC",
        (user_id,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_complaints_for_collector(collector_id: int) -> list[dict[str, Any]]:
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT * FROM complaints
        WHERE status NOT IN ('Verified')
          AND (collector_id = ? OR collector_id IS NULL)
        ORDER BY priority_score DESC, created_at ASC
        """,
        (collector_id,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_complaint_by_id(complaint_id: str) -> dict[str, Any] | None:
    conn = get_connection()
    row = conn.execute("SELECT * FROM complaints WHERE complaint_id = ?", (complaint_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def find_duplicate_complaint(latitude: float, longitude: float) -> dict[str, Any] | None:
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT * FROM complaints
        WHERE status IN ('Pending', 'Assigned', 'In Progress')
        AND duplicate_of IS NULL
        """
    ).fetchall()
    conn.close()

    for row in rows:
        dist = haversine_km(latitude, longitude, row["latitude"], row["longitude"])
        if dist <= DUPLICATE_RADIUS_KM:
            return dict(row)
    return None


def increment_duplicate_report(complaint_id: str) -> dict[str, Any]:
    conn = get_connection()
    conn.execute(
        "UPDATE complaints SET report_count = report_count + 1 WHERE complaint_id = ?",
        (complaint_id,),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM complaints WHERE complaint_id = ?", (complaint_id,)).fetchone()
    conn.close()
    return dict(row)


def save_complaint(payload: dict[str, Any]) -> dict[str, Any]:
    conn = get_connection()
    cursor = conn.execute(
        """
        INSERT INTO complaints (
            complaint_id, user_id, image_url, latitude, longitude,
            waste_type, severity, status, created_at, collector_id,
            before_image, ai_confidence, description, priority_score,
            duplicate_of, report_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            payload.get("before_image", payload.get("image_url", "")),
            payload.get("ai_confidence"),
            payload.get("description", ""),
            payload.get("priority_score", 0.5),
            payload.get("duplicate_of"),
            payload.get("report_count", 1),
        ),
    )
    conn.commit()
    complaint = conn.execute("SELECT * FROM complaints WHERE id = ?", (cursor.lastrowid,)).fetchone()
    conn.close()
    return dict(complaint)


def update_complaint_status(complaint_id: str, status: str, **extra: Any) -> dict[str, Any] | None:
    conn = get_connection()
    fields = ["status = ?"]
    values: list[Any] = [status]
    for key, value in extra.items():
        fields.append(f"{key} = ?")
        values.append(value)
    values.append(complaint_id)
    conn.execute(f"UPDATE complaints SET {', '.join(fields)} WHERE complaint_id = ?", values)
    conn.commit()
    row = conn.execute("SELECT * FROM complaints WHERE complaint_id = ?", (complaint_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def _nearest_collector(rows, latitude: float, longitude: float):
    return min(rows, key=lambda row: haversine_km(latitude, longitude, row["latitude"], row["longitude"]))


def assign_collector(latitude: float, longitude: float) -> dict[str, Any] | None:
    conn = get_connection()
    linked = conn.execute(
        """
        SELECT id, name, latitude, longitude FROM collectors
        WHERE availability = 'Available' AND user_id IS NOT NULL
        """
    ).fetchall()
    rows = linked or conn.execute(
        "SELECT id, name, latitude, longitude FROM collectors WHERE availability = 'Available'"
    ).fetchall()
    conn.close()

    if not rows:
        return None
    return dict(_nearest_collector(rows, latitude, longitude))


def _reassign_orphaned_complaints(conn: sqlite3.Connection) -> None:
    """Give open tasks to collectors who can log in so the dashboard is not empty."""
    linked = conn.execute(
        "SELECT id, latitude, longitude FROM collectors WHERE user_id IS NOT NULL"
    ).fetchall()
    if not linked:
        return

    orphaned = conn.execute(
        """
        SELECT c.complaint_id, c.latitude, c.longitude
        FROM complaints c
        LEFT JOIN collectors col ON col.id = c.collector_id
        WHERE c.status IN ('Pending', 'Assigned', 'In Progress')
          AND (c.collector_id IS NULL OR col.user_id IS NULL)
        """
    ).fetchall()

    for row in orphaned:
        nearest = _nearest_collector(linked, row["latitude"], row["longitude"])
        conn.execute(
            """
            UPDATE complaints
            SET collector_id = ?,
                status = CASE WHEN status = 'Pending' THEN 'Assigned' ELSE status END
            WHERE complaint_id = ?
            """,
            (nearest["id"], row["complaint_id"]),
        )


def save_resolution(payload: dict[str, Any]) -> dict[str, Any]:
    conn = get_connection()
    cursor = conn.execute(
        """
        INSERT INTO resolutions (complaint_id, before_image, after_image, completed_time, verified, verification_score)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            payload["complaint_id"],
            payload.get("before_image"),
            payload.get("after_image"),
            payload.get("completed_time"),
            payload.get("verified", 0),
            payload.get("verification_score"),
        ),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM resolutions WHERE id = ?", (cursor.lastrowid,)).fetchone()
    conn.close()
    return dict(row)


def get_admin_stats() -> dict[str, int]:
    complaints = get_all_complaints()
    return {
        "total": len(complaints),
        "pending": sum(1 for c in complaints if c["status"] == "Pending"),
        "assigned": sum(1 for c in complaints if c["status"] == "Assigned"),
        "in_progress": sum(1 for c in complaints if c["status"] == "In Progress"),
        "cleaned": sum(1 for c in complaints if c["status"] in {"Cleaned", "Verified"}),
        "high_priority": sum(1 for c in complaints if c["severity"] in {"High", "Critical"}),
        "duplicates_merged": sum(1 for c in complaints if c.get("duplicate_of")),
    }


def get_hotspot_clusters() -> list[dict[str, Any]]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT latitude, longitude, severity, waste_type, created_at FROM complaints WHERE duplicate_of IS NULL"
    ).fetchall()
    conn.close()

    if not rows:
        return []

    grid: dict[tuple[int, int], dict[str, Any]] = {}
    cell_size = 0.005  # ~500m cells

    for row in rows:
        key = (int(row["latitude"] / cell_size), int(row["longitude"] / cell_size))
        if key not in grid:
            grid[key] = {
                "lat_sum": 0.0,
                "lon_sum": 0.0,
                "count": 0,
                "high_severity": 0,
            }
        grid[key]["lat_sum"] += row["latitude"]
        grid[key]["lon_sum"] += row["longitude"]
        grid[key]["count"] += 1
        if row["severity"] in {"High", "Critical"}:
            grid[key]["high_severity"] += 1

    hotspots = []
    for data in grid.values():
        if data["count"] >= 1:
            hotspots.append(
                {
                    "latitude": data["lat_sum"] / data["count"],
                    "longitude": data["lon_sum"] / data["count"],
                    "complaint_count": data["count"],
                    "risk_score": min(1.0, (data["count"] * 0.2) + (data["high_severity"] * 0.15)),
                    "label": "High" if data["count"] >= 3 else "Medium" if data["count"] >= 2 else "Low",
                }
            )

    hotspots.sort(key=lambda h: h["risk_score"], reverse=True)
    return hotspots[:10]
