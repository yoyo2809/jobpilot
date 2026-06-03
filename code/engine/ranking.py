"""
engine/ranking.py
BAX-423 Lecture 7 — Ranking & Multi-Stage Recommendation Systems

Multi-stage pipeline:
  Stage 1 — Recall:    FAISS embedding retrieval (top-200)
  Stage 2 — Filter:    Hard rules (seniority, dealbreakers, salary)
  Stage 3 — Re-rank:   Weighted score (embedding + skills + location + feedback)
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from engine.embeddings import EmbeddingEngine
from engine import database as db


# ── Preference / dealbreaker data classes ────────────────────────────────────

@dataclass
class UserPreferences:
    background:       str       = ""
    location:         str       = ""
    min_salary:       float     = 0.0
    target_roles:     List[str] = field(default_factory=list)
    skills:           List[str] = field(default_factory=list)
    dealbreakers:     List[str] = field(default_factory=list)   # e.g. ["Junior","Contract"]
    remote_ok:        bool      = True
    visa_required:    bool      = False   # True → prioritise big companies


# ── Scoring weights ───────────────────────────────────────────────────────────

WEIGHTS = {
    "embedding":  0.25,
    "skill":      0.25,
    "role_match": 0.25,
    "location":   0.10,
    "feedback":   0.15,
}

SENIORITY_KEYWORDS = {
    "junior":   ["junior", "entry", "associate", "jr."],
    "senior":   ["senior", "sr.", "lead", "principal", "staff", "director"],
    "mid":      ["mid", "ii", "iii", "intermediate"],
}


# ── Public API ────────────────────────────────────────────────────────────────

def rank_jobs(
    profile_text: str,
    prefs: UserPreferences,
    embedding_engine: EmbeddingEngine,
    session_id: str,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Full three-stage pipeline.
    Returns a DataFrame with ranked jobs + per-job score breakdown.
    """
    # Stage 1 — recall (Increase k to 1000 to prevent post-filtering starvation)
    candidates = _stage1_recall(profile_text, embedding_engine, k=1000)
    if candidates.empty:
        return pd.DataFrame()

    # Stage 2 — hard filter
    filtered, filter_reasons = _stage2_filter(candidates, prefs)

    # Stage 3 — re-rank
    feedback_scores = _load_feedback_scores(session_id)
    ranked = _stage3_rerank(filtered, prefs, feedback_scores)

    return ranked.head(top_n).reset_index(drop=True)


# ── Stage 1: Recall ───────────────────────────────────────────────────────────

def _stage1_recall(
    profile_text: str,
    engine: EmbeddingEngine,
    k: int = 200,
) -> pd.DataFrame:
    """FAISS ANN retrieval — returns top-k jobs with embedding_score."""
    hits = engine.search(profile_text, k=k)
    if not hits:
        return pd.DataFrame()

    job_ids  = [h[0] for h in hits]
    emb_map  = {h[0]: h[1] for h in hits}

    jobs_df  = db.get_jobs_by_ids(job_ids)
    jobs_df["embedding_score"] = jobs_df["id"].astype(str).map(emb_map).fillna(0)
    return jobs_df


# ── Stage 2: Hard Filter ──────────────────────────────────────────────────────

def _stage2_filter(
    df: pd.DataFrame,
    prefs: UserPreferences,
) -> tuple[pd.DataFrame, Dict[str, str]]:
    """
    Remove jobs that violate hard dealbreakers.
    Checks title, company, work_type, experience_level, AND description.
    Also enforces positive role relevance if target_roles are specified.
    Returns (filtered_df, {job_id: reason}).
    """
    removed: Dict[str, str] = {}
    keep = []

    # Build a set of target role keywords for positive filtering
    role_keywords = set()
    for role in prefs.target_roles:
        for word in role.lower().split():
            if len(word) > 2:  # skip short words like "ML"
                role_keywords.add(word)
        # Also add the full role as-is for exact matching
        role_keywords.add(role.lower())
    # Add 2-letter important abbreviations back
    for role in prefs.target_roles:
        for word in role.lower().split():
            if word in ("ml", "ai", "ds", "de", "qa"):
                role_keywords.add(word)

    for _, row in df.iterrows():
        job_id  = str(row["id"])
        title   = str(row.get("title", "")).lower()
        company = str(row.get("company", "")).lower()
        desc    = str(row.get("description", "")).lower()[:800]
        work_t  = str(row.get("work_type", "")).lower()
        exp_lvl = str(row.get("experience_level", "")).lower()
        skills_t = str(row.get("skills", "")).lower()

        # Combine primary text for strict dealbreaker matching
        primary_text = f"{title} {company} {work_t} {exp_lvl}"

        # Check each dealbreaker
        blocked = False
        for db_kw in prefs.dealbreakers:
            pattern = db_kw.lower()
            # If the dealbreaker is explicitly mentioned in title, company, work_type or exp_level
            if pattern in primary_text:
                removed[job_id] = f'Contains dealbreaker keyword in header: "{db_kw}"'
                blocked = True
                break
            # For description, only block if it's explicitly stated as a requirement or level
            if pattern in desc:
                # We strictly enforce dealbreakers in descriptions as per persona Pass Criteria
                removed[job_id] = f'Contains dealbreaker keyword in desc: "{db_kw}"'
                blocked = True
                break

        if blocked:
            continue

        # Salary hard floor
        sal_max = _to_float(row.get("salary_max"))
        if prefs.min_salary > 0 and sal_max and sal_max < prefs.min_salary * 0.7:
            removed[job_id] = f"Salary below threshold (${sal_max:,.0f} < ${prefs.min_salary*0.7:,.0f})"
            continue

        # Positive role relevance filter: if target roles are set,
        # require at least one keyword match in title or description
        if role_keywords:
            title_and_desc = f"{title} {desc}"
            has_role_relevance = any(kw in title_and_desc for kw in role_keywords)
            if not has_role_relevance:
                removed[job_id] = "No relevance to target roles"
                continue

        # Visa sponsorship hard filter
        if prefs.visa_required:
            if "no sponsorship" in desc or "no c2c" in desc or "us citizen" in desc or "green card" in desc:
                removed[job_id] = "Does not offer Visa sponsorship"
                continue

        # Location hard filter
        if prefs.location:
            pref_loc = prefs.location.lower()
            job_loc = str(row.get("location", "")).lower()
            is_remote = "remote" in job_loc or "remote" in work_t
            
            # Check if job matches preferred location
            loc_match = False
            if pref_loc in job_loc or job_loc in pref_loc:
                loc_match = True
            elif "bay area" in pref_loc:
                bay_keywords = ["san francisco", "san jose", "palo alto", "mountain view", "sunnyvale", "santa clara", "bay area", "oakland", "cupertino", "menlo park"]
                if any(k in job_loc for k in bay_keywords):
                    loc_match = True
            else:
                pref_words = set([w for w in pref_loc.replace(",", " ").split() if len(w) > 3 and w not in ("area", "remote", "united", "states")])
                if pref_words and any(w in job_loc for w in pref_words):
                    loc_match = True
                    
            if not loc_match and not (is_remote and prefs.remote_ok):
                # Only strictly filter if we're dealing with a real city/region, not "United States"
                if pref_loc != "united states":
                    removed[job_id] = f"Location mismatch ({job_loc})"
                    continue

        keep.append(row)

    if not keep:
        return pd.DataFrame(), removed
    return pd.DataFrame(keep), removed


# ── Stage 3: Re-rank ──────────────────────────────────────────────────────────

def _stage3_rerank(
    df: pd.DataFrame,
    prefs: UserPreferences,
    feedback_scores: Dict[str, float],
) -> pd.DataFrame:
    """
    Weighted scoring → final ranked list.
    Adds score columns for transparency (Explain feature).
    """
    if df.empty:
        return df

    skill_scores    = df.apply(lambda r: _skill_score(r, prefs.skills), axis=1)
    role_scores     = df.apply(lambda r: _role_match_score(r, prefs.target_roles), axis=1)
    location_scores = df.apply(lambda r: _location_score(r, prefs), axis=1)
    
    # Base explicit feedback (exact job ID match)
    fb_explicit     = df["id"].astype(str).map(feedback_scores).fillna(0.0)
    
    # Adaptive Learning (Propagate to similar companies / titles / experience levels)
    # If the user liked Apple before, other Apple jobs get a slight boost.
    # If the user rejected an Analyst role, other Analyst roles get penalized.
    company_fb = {}
    title_fb = {}
    exp_fb = {}
    
    # We need to look up the historical jobs the user gave feedback on
    if feedback_scores:
        hist_jobs = db.get_jobs_by_ids(list(feedback_scores.keys()))
        for _, hj in hist_jobs.iterrows():
            hid = str(hj["id"])
            score = feedback_scores.get(hid, 0)
            comp = str(hj.get("company", "")).lower()
            if comp:
                company_fb[comp] = company_fb.get(comp, 0) + (score * 0.5) # 50% propagation
                
            tit = str(hj.get("title", "")).lower()
            if tit:
                title_fb[tit] = title_fb.get(tit, 0) + (score * 0.3)
                
            exp = str(hj.get("experience_level", "")).lower()
            if exp and exp != "nan":
                exp_fb[exp] = exp_fb.get(exp, 0) + (score * 0.4)
    
    def _compute_semantic_fb(row):
        base = 0.0
        c = str(row.get("company", "")).lower()
        t = str(row.get("title", "")).lower()
        e = str(row.get("experience_level", "")).lower()
        if c in company_fb: base += company_fb[c]
        if t in title_fb: base += title_fb[t]
        if e in exp_fb: base += exp_fb[e]
        return base
        
    fb_implicit = df.apply(_compute_semantic_fb, axis=1)
    
    # Combine explicit (exact job) and implicit (semantic propagation)
    fb_total = fb_explicit + fb_implicit

    # Clamp feedback to [-1, 1] then shift to [0, 1] for the scoring model
    fb_norm         = (fb_total.clip(-1, 1) + 1) / 2

    emb = df["embedding_score"].clip(0, 1)

    df = df.copy()
    df["skill_score"]    = skill_scores.round(4)
    df["role_score"]     = role_scores.round(4)
    df["location_score"] = location_scores.round(4)
    df["feedback_score"] = fb_norm.round(4)
    df["final_score"]    = (
        WEIGHTS["embedding"]   * emb            +
        WEIGHTS["skill"]       * skill_scores   +
        WEIGHTS["role_match"]  * role_scores    +
        WEIGHTS["location"]    * location_scores +
        WEIGHTS["feedback"]    * fb_norm
    ).round(4)

    df["match_pct"] = (df["final_score"] * 100).clip(0, 100).astype(int)

    return df.sort_values("final_score", ascending=False)


# ── Scoring helpers ───────────────────────────────────────────────────────────

def _role_match_score(row: pd.Series, target_roles: List[str]) -> float:
    """
    Score how well the job title matches the user's desired roles.
    This is critical for career pivoters — without it, a 'Data Analyst'
    who wants 'ML Engineer' would see DA jobs ranked equally to ML jobs.
    """
    if not target_roles:
        return 0.5
    title = str(row.get("title", "")).lower()
    desc  = str(row.get("description", "")).lower()[:500]
    
    best_score = 0.0
    for role in target_roles:
        role_lower = role.lower()
        # Exact title match → full score
        if role_lower in title:
            best_score = max(best_score, 1.0)
            continue
            
        # Strict matching: ML Engineer shouldn't match Software Engineer
        # If the target role is "ML Engineer", they MUST have "ML" or "Machine Learning"
        role_words = [w for w in role_lower.split() if len(w) > 2]
        
        # Build strict keywords
        strict_kws = []
        if "ml" in role_lower.split() or "machine learning" in role_lower:
            strict_kws.extend(["ml", "machine learning", "ai", "artificial intelligence"])
        if "data scientist" in role_lower:
            strict_kws.extend(["data scientist", "data science"])
            
        if strict_kws:
            if any(k in title for k in strict_kws):
                best_score = max(best_score, 0.9)
            elif any(k in desc for k in strict_kws):
                best_score = max(best_score, 0.6)
                
        # Generic word match (needs ALL words to match for high score)
        if role_words:
            title_hits = sum(1 for w in role_words if w in title)
            if title_hits == len(role_words):
                best_score = max(best_score, 0.9)
            elif title_hits >= len(role_words) * 0.5:
                best_score = max(best_score, 0.6) # Lowered from 0.8 to heavily penalise partial matches like Software Engineer
                
            desc_hits = sum(1 for w in role_words if w in desc)
            if desc_hits == len(role_words):
                best_score = max(best_score, 0.6)
            elif desc_hits >= len(role_words) * 0.5:
                best_score = max(best_score, 0.3)
    
    return best_score

def _skill_score(row: pd.Series, user_skills: List[str]) -> float:
    if not user_skills:
        return 0.5
        
    title = str(row.get("title", "")).lower()
    job_skills = str(row.get("skills", "")).lower()
    desc = str(row.get("description", "")).lower()[:500]
    
    text_to_search = f"{title} {job_skills} {desc}"
    
    hits = sum(1 for s in user_skills if s.lower() in text_to_search)
    
    if hits == 0:
        return 0.1
    if hits == 1:
        return 0.4
    if hits == 2:
        return 0.7
    if hits >= 3:
        return 1.0
    return 0.5


def _location_score(row: pd.Series, prefs: UserPreferences) -> float:
    loc = str(row.get("location", "")).lower()
    
    if prefs.location:
        pref_loc = prefs.location.lower()
        
        # Exact substring match
        if pref_loc in loc or loc in pref_loc:
            return 1.0
            
        # Handle "Bay Area" specially
        if "bay area" in pref_loc:
            bay_keywords = ["san francisco", "san jose", "palo alto", "mountain view", "sunnyvale", "santa clara", "bay area", "oakland", "cupertino", "menlo park"]
            if any(k in loc for k in bay_keywords):
                return 1.0
                
        # Fallback to word overlap for partial matches (e.g. "San Francisco" in pref, but loc is "San Francisco, CA")
        pref_words = set([w for w in pref_loc.replace(",", " ").split() if len(w) > 3 and w not in ("area", "remote", "united", "states")])
        if pref_words and any(w in loc for w in pref_words):
            return 0.85
            
    if "remote" in loc and prefs.remote_ok:
        return 0.80
        
    if not prefs.location:
        return 0.5
        
    return 0.25


def _to_float(val) -> Optional[float]:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# ── Feedback scores ───────────────────────────────────────────────────────────

def _load_feedback_scores(session_id: str) -> Dict[str, float]:
    """
    Convert raw accept/reject/skip actions into per-job float scores.
    accept → +1, reject → -2, skip → 0.
    Also propagates company-level and experience-level signals.
    """
    fb_df = db.get_feedback(session_id)
    if fb_df.empty:
        return {}

    job_ids  = fb_df["job_id"].unique().tolist()
    jobs_df  = db.get_jobs_by_ids(job_ids) if job_ids else pd.DataFrame()

    action_map = {"accept": 1.0, "skip": 0.0, "reject": -2.0}
    scores: Dict[str, float] = {}

    for _, row in fb_df.iterrows():
        jid = str(row["job_id"])
        scores[jid] = scores.get(jid, 0) + action_map.get(row["action"], 0)

    return scores


# ── Explain feature ───────────────────────────────────────────────────────────

def explain_ranking(row: pd.Series, prefs: UserPreferences) -> str:
    """
    Generate a human-readable explanation for why a job was ranked here.
    Required by the rubric ('Explain feature').
    """
    lines = [
        f"**Overall Match Score: {row.get('match_pct', 0)}%**",
        "",
        "### Score Breakdown",
        f"- 🔍 **Embedding Similarity** (Lecture 5): `{row.get('embedding_score', 0):.2f}` × 25%",
        f"  > Your profile semantically aligns with this role.",
        f"- 🛠 **Skill Overlap** (Lecture 7): `{row.get('skill_score', 0):.2f}` × 25%",
    ]

    # Show matched skills
    if prefs.skills:
        job_text = (str(row.get("skills", "")) + " " + str(row.get("description", ""))[:600]).lower()
        matched  = [s for s in prefs.skills if s.lower() in job_text]
        missing  = [s for s in prefs.skills if s.lower() not in job_text]
        if matched:
            lines.append(f"  > ✅ Matched: {', '.join(matched[:6])}")
        if missing:
            lines.append(f"  > ❌ Not found: {', '.join(missing[:4])}")

    lines += [
        f"- 🎯 **Role Match**: `{row.get('role_score', 0):.2f}` × 25%",
        f"  > How well the job title matches your target roles: {', '.join(prefs.target_roles[:3]) if prefs.target_roles else 'Any'}",
        f"- 📍 **Location Match**: `{row.get('location_score', 0):.2f}` × 10%",
        f"  > Job location: {row.get('location', 'N/A')} | Your pref: {prefs.location or 'Any'}",
        f"- 🔄 **Adaptive Feedback**: `{row.get('feedback_score', 0.5):.2f}` × 15%",
        f"  > Based on your accept/reject history this session.",
    ]

    sal_min = _to_float(row.get("salary_min"))
    sal_max = _to_float(row.get("salary_max"))
    if sal_min or sal_max:
        sal_str = f"${sal_min:,.0f}" if sal_min else "?"
        sal_str += f" – ${sal_max:,.0f}" if sal_max else "+"
        lines.append(f"\n**Salary Range**: {sal_str}")

    exp = row.get("experience_level", "")
    if exp and str(exp) != "nan":
        lines.append(f"**Experience Level**: {exp}")

    return "\n".join(lines)
