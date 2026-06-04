"""
engine/database.py
SQLite database operations for JobPilot.
"""
import sqlite3
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

DB_PATH = Path(__file__).parent.parent.parent / "data" / "jobpilot.db"


def get_connection() -> sqlite3.Connection:
    # Auto-extract zipped DB if it exists (for Streamlit Cloud deployment)
    zip_path = DB_PATH.parent / "jobpilot.db.zip"
    if not DB_PATH.exists() and zip_path.exists():
        import zipfile
        print("Extracting jobpilot.db from zip...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DB_PATH.parent)
            
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_db():
    """Create all tables if they don't exist."""
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS jobs (
            id              TEXT PRIMARY KEY,
            title           TEXT NOT NULL,
            company         TEXT,
            location        TEXT,
            description     TEXT,
            salary_min      REAL,
            salary_max      REAL,
            experience_level TEXT,
            work_type       TEXT,
            skills          TEXT,
            apply_url       TEXT,
            source          TEXT DEFAULT 'Kaggle',
            date_posted     TEXT,
            date_ingested   TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS feedback (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT NOT NULL,
            job_id      TEXT NOT NULL,
            action      TEXT NOT NULL,
            timestamp   TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS user_profiles (
            session_id      TEXT PRIMARY KEY,
            raw_text        TEXT,
            skills          TEXT,
            experience_years INTEGER,
            education       TEXT,
            target_roles    TEXT,
            preferences     TEXT,
            created_at      TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS streaming_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            source      TEXT,
            jobs_fetched INTEGER,
            jobs_new    INTEGER,
            timestamp   TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_jobs_title    ON jobs(title);
        CREATE INDEX IF NOT EXISTS idx_jobs_source   ON jobs(source);
        CREATE INDEX IF NOT EXISTS idx_fb_session    ON feedback(session_id);
    """)
    conn.commit()
    conn.close()


def get_job_count() -> int:
    conn = get_connection()
    count = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    conn.close()
    return count


def get_all_jobs_for_indexing() -> pd.DataFrame:
    """Load all jobs needed to build the FAISS index."""
    conn = get_connection()
    df = pd.read_sql(
        "SELECT id, title, company, location, description, skills, experience_level "
        "FROM jobs",
        conn,
    )
    conn.close()
    return df


def get_jobs_by_ids(job_ids: list) -> pd.DataFrame:
    if not job_ids:
        return pd.DataFrame()
    conn = get_connection()
    placeholders = ",".join("?" * len(job_ids))
    df = pd.read_sql(
        f"SELECT * FROM jobs WHERE id IN ({placeholders})",
        conn,
        params=job_ids,
    )
    conn.close()
    return df


def insert_jobs_batch(jobs: list) -> int:
    """Insert new jobs (from streaming). Returns count of truly new rows."""
    if not jobs:
        return 0
    conn = get_connection()
    new_count = 0
    for job in jobs:
        cursor = conn.execute(
            """INSERT OR IGNORE INTO jobs
               (id, title, company, location, description,
                salary_min, salary_max, apply_url, source, date_posted)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                str(job.get("id", "")),
                job.get("title", ""),
                job.get("company", ""),
                job.get("location", ""),
                job.get("description", ""),
                job.get("salary_min"),
                job.get("salary_max"),
                job.get("apply_url", ""),
                job.get("source", "Adzuna"),
                job.get("date_posted", datetime.now().isoformat()),
            ),
        )
        new_count += cursor.rowcount
    conn.commit()
    conn.close()
    return new_count


def save_feedback(session_id: str, job_id: str, action: str):
    conn = get_connection()
    conn.execute(
        "INSERT INTO feedback (session_id, job_id, action) VALUES (?,?,?)",
        (session_id, job_id, action),
    )
    conn.commit()
    conn.close()


def find_startup_like_job_id() -> Optional[str]:
    """Return one job that looks like a tiny/startup company for demo feedback."""
    conn = get_connection()
    row = conn.execute(
        """
        SELECT id FROM jobs
        WHERE lower(title || ' ' || company || ' ' || coalesce(description, ''))
              LIKE '%startup%'
           OR lower(title || ' ' || company || ' ' || coalesce(description, ''))
              LIKE '%early-stage%'
           OR lower(title || ' ' || company || ' ' || coalesce(description, ''))
              LIKE '%seed%'
           OR lower(title || ' ' || company || ' ' || coalesce(description, ''))
              LIKE '%stealth%'
        LIMIT 1
        """
    ).fetchone()
    conn.close()
    return str(row["id"]) if row else None


def get_feedback(session_id: str) -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql(
        "SELECT job_id, action, timestamp FROM feedback "
        "WHERE session_id=? ORDER BY timestamp",
        conn,
        params=(session_id,),
    )
    conn.close()
    return df


def save_user_profile(session_id: str, profile: dict):
    conn = get_connection()
    conn.execute(
        """INSERT OR REPLACE INTO user_profiles
           (session_id, raw_text, skills, experience_years,
            education, target_roles, preferences)
           VALUES (?,?,?,?,?,?,?)""",
        (
            session_id,
            profile.get("raw_text", ""),
            json.dumps(profile.get("skills", [])),
            profile.get("experience_years", 0),
            profile.get("education", ""),
            json.dumps(profile.get("target_roles", [])),
            json.dumps(profile.get("preferences", {})),
        ),
    )
    conn.commit()
    conn.close()


def log_streaming(source: str, fetched: int, new: int):
    conn = get_connection()
    conn.execute(
        "INSERT INTO streaming_log (source, jobs_fetched, jobs_new) VALUES (?,?,?)",
        (source, fetched, new),
    )
    conn.commit()
    conn.close()


def get_streaming_stats() -> dict:
    conn = get_connection()
    row = conn.execute(
        "SELECT SUM(jobs_fetched) as total_fetched, "
        "SUM(jobs_new) as total_new, "
        "MAX(timestamp) as last_fetch "
        "FROM streaming_log"
    ).fetchone()
    conn.close()
    return {
        "total_fetched": row["total_fetched"] or 0,
        "total_new": row["total_new"] or 0,
        "last_fetch": row["last_fetch"] or "Never",
    }


def get_analytics_data() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM jobs", conn)
    conn.close()
    return df


def load_csv_to_db(csv_path: str, replace: bool = True):
    """
    Load the jobs CSV snapshot into SQLite.
    Handles LinkedIn Kaggle dataset column naming.
    """
    print(f"Loading {csv_path} into database ...")
    df = pd.read_csv(csv_path, low_memory=False)

    # Map LinkedIn columns → our schema
    col_map = {
        "job_id": "id",
        "title": "title",
        "company_name": "company",
        "location": "location",
        "description": "description",
        "max_salary": "salary_max",
        "min_salary": "salary_min",
        "formatted_experience_level": "experience_level",
        "work_type": "work_type",
        "skills_desc": "skills",
        "job_posting_url": "apply_url",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    needed = ["id", "title", "company", "location", "description",
              "salary_min", "salary_max", "experience_level",
              "work_type", "skills", "apply_url", "source", "date_posted"]
    for col in needed:
        if col not in df.columns:
            df[col] = None

    df["id"]          = df["id"].astype(str)
    df["source"]      = "LinkedIn/Kaggle"
    df["description"] = df["description"].fillna("").astype(str)
    df["title"]       = df["title"].fillna("Unknown").astype(str)
    df["company"]     = df["company"].fillna("Unknown").astype(str)

    conn = get_connection()
    if replace:
        conn.execute("DELETE FROM jobs")
        conn.execute("DELETE FROM streaming_log")
        conn.commit()
    df[needed].to_sql("jobs", conn, if_exists="append", index=False, chunksize=2000)
    conn.close()
    print(f"✅ Loaded {len(df):,} jobs into database.")
