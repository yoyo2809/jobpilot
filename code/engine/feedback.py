"""
engine/feedback.py
Adaptive learning from user accept/reject/skip signals.
Records signals in SQLite and computes per-session feedback scores.
"""
import pandas as pd
from engine import database as db


ACTION_WEIGHTS = {
    "accept": +1.0,
    "skip":    0.0,
    "reject": -2.0,
}


def record(session_id: str, job_id: str, action: str):
    """Persist a user feedback action."""
    db.save_feedback(session_id, job_id, action)


def get_session_summary(session_id: str) -> dict:
    """
    Return stats about the current session's feedback.
    Used for UI display.
    """
    fb = db.get_feedback(session_id)
    if fb.empty:
        return {"accepted": 0, "rejected": 0, "skipped": 0, "total": 0}

    counts = fb["action"].value_counts()
    return {
        "accepted": int(counts.get("accept", 0)),
        "rejected": int(counts.get("reject", 0)),
        "skipped":  int(counts.get("skip", 0)),
        "total":    len(fb),
    }


def compute_job_scores(session_id: str) -> dict[str, float]:
    """
    Build a dict {job_id: cumulative_score} from all feedback.
    Scores are raw (not normalised); used by ranking.py.
    """
    fb = db.get_feedback(session_id)
    if fb.empty:
        return {}

    scores: dict[str, float] = {}
    for _, row in fb.iterrows():
        jid = str(row["job_id"])
        scores[jid] = scores.get(jid, 0.0) + ACTION_WEIGHTS.get(row["action"], 0.0)
    return scores


def company_penalties(session_id: str) -> dict[str, float]:
    """
    Derive per-company negative signals from reject actions.
    Used to down-weight jobs from companies the user has rejected.
    """
    fb = db.get_feedback(session_id)
    if fb.empty:
        return {}

    rejected_ids = fb[fb["action"] == "reject"]["job_id"].tolist()
    if not rejected_ids:
        return {}

    jobs_df = db.get_jobs_by_ids(rejected_ids)
    if jobs_df.empty:
        return {}

    penalties: dict[str, float] = {}
    company_counts = jobs_df["company"].value_counts()
    for company, count in company_counts.items():
        penalties[str(company)] = -0.5 * count   # -0.5 per rejection

    return penalties
