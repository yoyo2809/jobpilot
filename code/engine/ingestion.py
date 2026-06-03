"""
engine/ingestion.py
Streaming job ingestion pipeline — Core Capability #1.

Simulates a Kafka-style producer/consumer:
  - Producer: Adzuna REST API (polls for new postings)
  - Consumer: Dedup + insert into SQLite
  - Channel:  threading.Queue (represents Kafka topic)
"""
from __future__ import annotations

import hashlib
import queue
import threading
import time
from datetime import datetime
from typing import Optional

import requests
import streamlit as st

from engine import database as db

# ── Adzuna API ────────────────────────────────────────────────────────────────

ADZUNA_BASE = "https://api.adzuna.com/v1/api/jobs"


def _adzuna_creds() -> tuple[str, str]:
    app_id  = st.secrets.get("ADZUNA_APP_ID", "")
    app_key = st.secrets.get("ADZUNA_APP_KEY", "")
    return app_id, app_key


def fetch_adzuna(
    what: str = "data scientist",
    where: str = "us",
    country: str = "us",
    results_per_page: int = 50,
    page: int = 1,
) -> list[dict]:
    """
    Fetch job postings from Adzuna API.
    Returns list of normalised job dicts.
    """
    app_id, app_key = _adzuna_creds()
    if not app_id or not app_key:
        return []

    url    = f"{ADZUNA_BASE}/{country}/search/{page}"
    params = {
        "app_id":           app_id,
        "app_key":          app_key,
        "results_per_page": results_per_page,
        "what":             what,
        "where":            where,
        "content-type":     "application/json",
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return [_normalise_adzuna(item) for item in data.get("results", [])]
    except Exception as e:
        print(f"Adzuna fetch error: {e}")
        return []


def _normalise_adzuna(item: dict) -> dict:
    """Map Adzuna JSON → our unified job schema."""
    sal   = item.get("salary_min"), item.get("salary_max")
    desc  = item.get("description", "")
    loc   = item.get("location", {})
    loc_s = ", ".join(loc.get("area", []))

    raw_id = str(item.get("id", "")) or (item.get("redirect_url", "") + desc[:40])
    job_id = hashlib.md5(raw_id.encode()).hexdigest()

    return {
        "id":          job_id,
        "title":       item.get("title", ""),
        "company":     item.get("company", {}).get("display_name", ""),
        "location":    loc_s,
        "description": desc,
        "salary_min":  sal[0],
        "salary_max":  sal[1],
        "apply_url":   item.get("redirect_url", ""),
        "source":      "Adzuna",
        "date_posted": item.get("created", datetime.now().isoformat()),
    }


# ── Deduplication ─────────────────────────────────────────────────────────────

def _existing_ids() -> set[str]:
    """Fast set of all job IDs currently in SQLite."""
    from engine.database import get_connection
    conn = get_connection()
    rows = conn.execute("SELECT id FROM jobs").fetchall()
    conn.close()
    return {str(r[0]) for r in rows}


def deduplicate(jobs: list[dict]) -> list[dict]:
    existing = _existing_ids()
    return [j for j in jobs if j["id"] not in existing]


# ── Producer / Consumer (Kafka-style) ─────────────────────────────────────────

_job_queue: queue.Queue = queue.Queue(maxsize=500)
_consumer_active = False


def _producer_worker(query: str, stop_event: threading.Event):
    """
    Background thread: polls Adzuna every 60s and pushes to queue.
    Represents a Kafka Producer.
    """
    while not stop_event.is_set():
        try:
            jobs = fetch_adzuna(what=query, results_per_page=50)
            for job in jobs:
                try:
                    _job_queue.put_nowait(job)
                except queue.Full:
                    pass
        except Exception as e:
            print(f"Producer error: {e}")
        stop_event.wait(60)  # poll every 60 seconds


def _consumer_worker(stop_event: threading.Event):
    """
    Background thread: drains queue → dedup → SQLite.
    Represents a Kafka Consumer.
    """
    while not stop_event.is_set() or not _job_queue.empty():
        try:
            job   = _job_queue.get(timeout=2)
            deduped = deduplicate([job])
            if deduped:
                n = db.insert_jobs_batch(deduped)
                if n:
                    db.log_streaming("Adzuna", 1, n)
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Consumer error: {e}")


class StreamingPipeline:
    """
    Manages the producer + consumer threads.
    Call start() once; stop() at shutdown.
    """

    def __init__(self):
        self._stop      = threading.Event()
        self._producer  = None
        self._consumer  = None
        self.running    = False

    def start(self, query: str = "data scientist machine learning"):
        if self.running:
            return
        self._stop.clear()
        self._producer = threading.Thread(
            target=_producer_worker, args=(query, self._stop), daemon=True
        )
        self._consumer = threading.Thread(
            target=_consumer_worker, args=(self._stop,), daemon=True
        )
        self._producer.start()
        self._consumer.start()
        self.running = True
        print("✅ Streaming pipeline started")

    def stop(self):
        self._stop.set()
        self.running = False
        print("⏹ Streaming pipeline stopped")


# ── One-shot manual fetch (for Streamlit button) ──────────────────────────────

def manual_fetch(query: str = "data scientist machine learning",
                 max_pages: int = 3) -> dict:
    """
    Synchronous fetch used when the user clicks 'Fetch New Jobs' in the UI.
    Returns stats dict, including inserted job IDs so the UI can add new
    records to the in-memory FAISS index without a full rebuild.
    """
    all_jobs  = []
    for page in range(1, max_pages + 1):
        jobs = fetch_adzuna(what=query, results_per_page=50, page=page)
        all_jobs.extend(jobs)

    new_jobs = deduplicate(all_jobs)
    inserted = db.insert_jobs_batch(new_jobs)
    db.log_streaming("Adzuna (manual)", len(all_jobs), inserted)

    return {
        "fetched": len(all_jobs),
        "new":     inserted,
        "dupes":   len(all_jobs) - len(new_jobs),
        "new_job_ids": [str(j["id"]) for j in new_jobs[:inserted]],
    }
