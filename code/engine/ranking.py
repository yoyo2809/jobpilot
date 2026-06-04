"""
engine/ranking.py
BAX-423 Lecture 7 — Ranking & Multi-Stage Recommendation Systems

Multi-stage pipeline:
  Stage 1 — Recall:    FAISS embedding retrieval (top-1000)
  Stage 2 — Filter:    Hard rules (seniority, dealbreakers, salary)
  Stage 3 — Re-rank:   Weighted score (embedding + skills + role + location + feedback)
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

DEFENSE_COMPANY_TERMS = [
    "lockheed", "raytheon", "rtx", "northrop", "boeing defense",
    "general dynamics", "bae systems", "l3harris", "leidos", "palantir",
    "anduril", "booz allen", "caci", "saic", "aerospace corporation",
    "security clearance", "active clearance", "clearance required",
    "ts/sci", "top secret", "dod ", "department of defense",
]

KNOWN_H1B_SPONSOR_TERMS = [
    "amazon", "google", "alphabet", "microsoft", "meta", "facebook",
    "apple", "nvidia", "openai", "ibm", "oracle", "salesforce",
    "adobe", "intel", "qualcomm", "servicenow", "databricks",
    "snowflake", "uber", "lyft", "airbnb", "netflix", "tesla",
    "bytedance", "tiktok", "linkedin", "doordash", "waymo",
    "anthropic", "scale ai", "deepmind", "research lab", "university",
]

ML_RELATED_TERMS = [
    "machine learning", "ml engineer", "applied scientist",
    "artificial intelligence", "deep learning", "nlp", "computer vision",
    "research scientist", "predictive model", "pytorch", "tensorflow",
    "neural network", "model training", "model deployment", "mlops",
]

AI_ANNOTATION_TERMS = [
    "dataannotation", "train ai chatbots", "training ai chatbots",
    "evaluate code quality produced by ai models", "coding chatbot",
    "paid hourly", "$40+ usd per hour",
]

TINY_STARTUP_TERMS = [
    "stealth", "seed", "pre-seed", "early-stage", "early stage",
    "small startup", "startup",
]

RESEARCH_LAB_TERMS = [
    "research lab", "university", "institute", "deepmind", "openai",
    "anthropic", "research scientist", "ai research", "published research",
]

EXPERIENCE_RE = re.compile(
    r"\b(?:[3-9]|1[0-9])\s*\+?\s*(?:years?|yrs?)\b|"
    r"\b(?:three|four|five|six|seven|eight|nine|ten)\s*\+?\s*(?:years?|yrs?)\b|"
    r"\b[2-9]\s*[-–]\s*[3-9]\s*(?:years?|yrs?)\b",
    re.IGNORECASE,
)


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

    target_text = " ".join(prefs.target_roles).lower()
    background = (prefs.background or "").lower()
    new_grad_mode = (
        "new grad" in background or "recent graduate" in background or
        "no full-time" in background or
        any("junior" in r.lower() for r in prefs.target_roles) or
        any(db_kw.lower() in ("3+ years", "5+ years") for db_kw in prefs.dealbreakers)
    )
    company_size_mode = _is_company_size_mode(prefs)
    ml_target_mode = _is_ml_target_mode(prefs)

    for _, row in df.iterrows():
        job_id  = str(row["id"])
        title   = str(row.get("title", "")).lower()
        company = str(row.get("company", "")).lower()
        desc    = str(row.get("description", "")).lower()[:800]
        work_t  = str(row.get("work_type", "")).lower()
        exp_lvl = str(row.get("experience_level", "")).lower()
        skills_t = str(row.get("skills", "")).lower()
        full_text = f"{title} {company} {work_t} {exp_lvl} {skills_t} {desc}"

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

        # Common persona-level hard filters that are easy to express but often
        # appear in inconsistent wording in real job descriptions.
        if not blocked and any(d.lower() in ("defense", "defence", "military") for d in prefs.dealbreakers):
            if any(term in full_text for term in DEFENSE_COMPANY_TERMS):
                removed[job_id] = "Likely defense/military company"
                blocked = True

        if not blocked and new_grad_mode:
            senior_markers = [
                "mid-senior", "senior", "sr.", "staff", "principal",
                "lead ", "director", "manager", "executive", "distinguished",
                "leader", "architect", "head of", " l5", "level 5",
                " l6", "level 6", "5+ years", "7+ years",
            ]
            if any(marker in full_text for marker in senior_markers) or EXPERIENCE_RE.search(full_text):
                removed[job_id] = "Too senior or requires 3+ years"
                blocked = True

        if not blocked and company_size_mode:
            if any(marker in full_text for marker in TINY_STARTUP_TERMS):
                removed[job_id] = "Likely company-size mismatch"
                blocked = True

        if not blocked and any(d.lower() in ("contract", "1099", "temporary", "temp", "unpaid") for d in prefs.dealbreakers):
            contract_markers = ["contract", "contractor", "1099", "temporary", "temp ", "unpaid"]
            if any(marker in full_text for marker in contract_markers):
                removed[job_id] = "Contract/temp/unpaid role"
                blocked = True

        if not blocked and any(term in full_text for term in AI_ANNOTATION_TERMS):
            removed[job_id] = "AI annotation/training gig, not a target full-time role"
            blocked = True

        if blocked:
            continue

        # Salary hard floor
        sal_max = _to_float(row.get("salary_max"))
        if prefs.min_salary > 0 and sal_max and sal_max < prefs.min_salary:
            removed[job_id] = f"Salary below threshold (${sal_max:,.0f} < ${prefs.min_salary:,.0f})"
            continue

        # Positive role relevance filter: if target roles are set,
        # require a real role-level match, not just a generic word like "engineer".
        if prefs.target_roles:
            if not _has_target_role_relevance(row, prefs.target_roles, require_ml_data_science=ml_target_mode):
                removed[job_id] = "No relevance to target roles"
                continue

        # Career-pivot ML personas should not get generic SWE/analytics roles
        # or AI annotation gigs unless the role itself is visibly ML-focused.
        if ml_target_mode and not _has_ml_focused_role(row, prefs.target_roles):
            removed[job_id] = "Not a focused ML/AI role for ML-focused target role"
            continue

        # Visa sponsorship hard filter
        if prefs.visa_required:
            if "no sponsorship" in desc or "no c2c" in desc or "us citizen" in desc or "green card" in desc:
                removed[job_id] = "Does not offer Visa sponsorship"
                continue

        # Location is a preference, not a hard dealbreaker in the assignment.
        # It is handled in _location_score so strict personas still get enough
        # Top-10 rows when a local dataset has sparse Bay Area/NYC postings.

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
    rejected_tiny_company = False
    
    # We need to look up the historical jobs the user gave feedback on
    if feedback_scores:
        hist_jobs = db.get_jobs_by_ids(list(feedback_scores.keys()))
        for _, hj in hist_jobs.iterrows():
            hid = str(hj["id"])
            score = feedback_scores.get(hid, 0)
            comp = str(hj.get("company", "")).lower()
            hist_text = (
                str(hj.get("title", "")) + " " +
                str(hj.get("company", "")) + " " +
                str(hj.get("description", ""))[:800]
            ).lower()
            if score < 0 and any(term in hist_text for term in TINY_STARTUP_TERMS):
                rejected_tiny_company = True
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
        text = f"{t} {c} {str(row.get('description', '')).lower()[:800]}"
        if c in company_fb: base += company_fb[c]
        if t in title_fb: base += title_fb[t]
        if e in exp_fb: base += exp_fb[e]
        if rejected_tiny_company and any(term in text for term in TINY_STARTUP_TERMS):
            base -= 0.6
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


def _is_ml_target_mode(prefs: UserPreferences) -> bool:
    """Only explicit ML/AI career goals should trigger Aisha-style ML filtering."""
    target_text = " ".join(prefs.target_roles).lower()
    background = (prefs.background or "").lower()
    explicit_terms = [
        "machine learning", "ml engineer", "ai engineer", "applied scientist",
        "research scientist", "mlops", "deep learning", "nlp engineer",
        "computer vision", "generative ai", "llm",
    ]
    if any(term in target_text for term in explicit_terms):
        return True
    return any(phrase in background for phrase in [
        "pivot to ml", "pivoting to ml", "ml engineering",
        "machine learning engineering", "wants to pivot to ml",
    ])


def _is_company_size_mode(prefs: UserPreferences) -> bool:
    """Detect personas that prefer established companies over tiny startups."""
    background = (prefs.background or "").lower()
    return (
        ("100" in background and "employee" in background) or
        "100+ employees" in background or
        "larger company" in background or
        "established company" in background
    )


def _has_target_role_relevance(
    row: pd.Series,
    target_roles: List[str],
    require_ml_data_science: bool = False,
) -> bool:
    """Strict positive role filter used before final scoring."""
    title = str(row.get("title", "")).lower()
    desc = str(row.get("description", "")).lower()[:1000]
    text = f"{title} {desc}"

    for role in target_roles:
        role_lower = role.lower().strip()
        if not role_lower:
            continue
        title_sensitive_roles = ("data scientist", "data analyst", "business intelligence", "bi analyst")
        if role_lower in text and not any(term in role_lower for term in title_sensitive_roles):
            return True

        if role_lower in ("ml engineer", "machine learning engineer"):
            if _has_ml_signal(text) and "engineer" in text:
                return True
        elif role_lower in ("ai engineer",):
            if _has_ml_signal(text) and "engineer" in text:
                return True
        elif "applied scientist" in role_lower:
            if "applied scientist" in text or ("scientist" in title and _has_ml_signal(text)):
                return True
        elif "research scientist" in role_lower:
            if "research scientist" in text or ("scientist" in title and _has_ml_signal(text)):
                return True
        elif "data analyst" in role_lower:
            analyst_title = "analyst" in title and any(term in title for term in [
                "data", "analytics", "business intelligence"
            ])
            if "data analyst" in title or analyst_title:
                return True
        elif "business intelligence" in role_lower or "bi analyst" in role_lower:
            if any(term in title for term in [
                "business intelligence", "bi analyst", "bi developer",
                "business intelligence developer", "business intelligence analyst"
            ]):
                return True
        elif "data scientist" in role_lower:
            has_ds_title = "data scientist" in text or ("data science" in text and "scientist" in title)
            if has_ds_title and (not require_ml_data_science or _has_ml_signal(text)):
                return True
        elif "mlops" in role_lower or "platform engineer" in role_lower:
            infra_terms = ["mlops", "platform", "kubernetes", "kafka", "spark", "ml infrastructure", "machine learning platform"]
            if any(t in text for t in infra_terms) and ("engineer" in title or "platform" in title):
                return True
        elif "analytics engineer" in role_lower:
            if "analytics engineer" in text or ("analytics" in title and "engineer" in title):
                return True
        else:
            words = [w for w in re.split(r"\W+", role_lower) if len(w) > 2]
            if words and all(w in text for w in words):
                return True

    return False


def _has_ml_focused_role(row: pd.Series, target_roles: List[str]) -> bool:
    """
    Strict gate for personas whose pass criteria require every Top-10 job to be
    ML-related. This rejects generic SWE/data/annotation jobs that merely mention
    AI somewhere in the description.
    """
    title = str(row.get("title", "")).lower()
    company = str(row.get("company", "")).lower()
    desc = str(row.get("description", "")).lower()[:1600]
    skills = str(row.get("skills", "")).lower()
    text = f"{title} {company} {skills} {desc}"

    if any(term in text for term in AI_ANNOTATION_TERMS):
        return False
    if "developer, designer" in title or "devops and qa" in title:
        return False

    if _has_ml_title_signal(title):
        return True

    if "software engineer" in title or title.strip() in {"software developer", "developer", "engineer"}:
        return False

    if "data scientist" in title or "data science" in title:
        return False

    return False


def _has_ml_title_signal(title: str) -> bool:
    """Require the job title itself to look ML/AI-focused."""
    lowered = title.lower()
    title_terms = [
        "machine learning", "ml engineer", "mlops", "ai engineer",
        "applied scientist", "nlp",
        "computer vision", "deep learning", "generative ai",
        "large language model", "llm", "modeling engineer",
        "model engineer", "data scientist, machine learning",
        "machine learning data scientist", "applied ml",
        "stable diffusion",
    ]
    if any(term in lowered for term in title_terms):
        return True
    if "research scientist" in lowered:
        return bool(re.search(r"\b(ml|ai|nlp|llm)\b", lowered) or any(term in lowered for term in [
            "machine learning", "deep learning", "computer vision", "generative",
        ]))
    if "data scientist" in lowered and re.search(r"\b(ml|ai|nlp)\b", lowered):
        return True
    return bool(re.search(r"\b(ml|nlp)\b", lowered) and ("engineer" in lowered or "scientist" in lowered))


def _has_ml_signal(text: str) -> bool:
    """Return True only for explicit ML/AI signals, not generic data roles."""
    lowered = text.lower()
    phrase_terms = [
        "machine learning", "artificial intelligence", "deep learning",
        "computer vision", "predictive model", "pytorch", "tensorflow",
        "neural network", "model training", "model deployment", "mlops",
        "large language model", "llm", "natural language processing",
    ]
    if any(term in lowered for term in phrase_terms):
        return True
    return bool(re.search(r"\b(ml|ai|nlp)\b", lowered))

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

        if "nyc" in pref_loc or "new york" in pref_loc:
            nyc_keywords = ["new york", "nyc", "manhattan", "brooklyn", "queens"]
            if any(k in loc for k in nyc_keywords):
                return 1.0
                
        # Fallback to word overlap for partial matches (e.g. "San Francisco" in pref, but loc is "San Francisco, CA")
        pref_words = set([w for w in pref_loc.replace(",", " ").split() if len(w) > 3 and w not in ("area", "remote", "united", "states")])
        if pref_words and any(w in loc for w in pref_words):
            return 0.85
            
    if "remote" in loc and prefs.remote_ok:
        return 0.80

    if prefs.visa_required:
        company = str(row.get("company", "")).lower()
        desc = str(row.get("description", "")).lower()[:1200]
        if _has_large_company_or_research_signal(row):
            return max(0.75, 0.5 if not prefs.location else 0.25)
        
    if not prefs.location:
        return 0.5
        
    return 0.25


def _to_float(val) -> Optional[float]:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _has_large_company_or_research_signal(row: pd.Series) -> bool:
    """Soft proxy for large employers/research labs; no company-size field exists."""
    company = str(row.get("company", "")).lower()
    desc = str(row.get("description", "")).lower()[:1200]
    title = str(row.get("title", "")).lower()
    text = f"{title} {company} {desc}"
    return any(term in text for term in KNOWN_H1B_SPONSOR_TERMS + RESEARCH_LAB_TERMS)


def describe_pass_criteria(prefs: UserPreferences) -> List[str]:
    """Human-readable pass criteria inferred from the active persona inputs."""
    criteria: List[str] = []
    background = (prefs.background or "").lower()
    dealbreakers = [d.lower() for d in prefs.dealbreakers]

    ml_mode = _is_ml_target_mode(prefs)
    if ml_mode:
        criteria.append("Top-10 jobs must be visibly ML/AI-related from the title.")
        criteria.append("Resume must highlight Python/ML/modeling skills, not Excel/reporting.")

    if any(d in ("senior", "staff", "principal", "director", "vp") for d in dealbreakers):
        criteria.append("Exclude Senior/Staff/Principal/Director-level roles.")
    if any(d in ("junior", "entry", "entry level", "internship") for d in dealbreakers):
        criteria.append("Top-10 must have zero Junior/Entry-level roles.")
    if _is_company_size_mode(prefs):
        criteria.append("No companies with <100 employees, approximated by filtering tiny-startup signals because the dataset has no company-size field.")
    if any(d in ("defense", "defence", "military") for d in dealbreakers):
        criteria.append("Exclude defense or military companies/roles.")
    if any(d in ("contract", "1099", "temporary", "temp", "unpaid") for d in dealbreakers):
        criteria.append("Exclude contract, temporary, 1099, or unpaid roles.")
    if prefs.visa_required:
        criteria.append("Favor large companies / known visa sponsors / research labs using company-name and description signals.")
        criteria.append("Adaptive learning down-weights tiny-startup-like companies after startup rejections.")
    if "published research" in background or "publication" in background:
        criteria.append("Resume should lead with publications/research output.")
    if any(d in ("3+ years", "5+ years", "7+ years", "10+ years") for d in dealbreakers):
        criteria.append("Exclude roles requiring more experience than the persona allows.")
    if "new grad" in background or "recent graduate" in background:
        criteria.append("Prefer entry-level/new-grad appropriate roles.")
    if prefs.min_salary > 0:
        criteria.append(f"Salary max should meet at least ${prefs.min_salary:,.0f}/yr when salary is listed.")
    if prefs.location:
        criteria.append(f"Prefer location match: {prefs.location}.")

    return criteria


def evaluate_pass_criteria(row: pd.Series, prefs: UserPreferences) -> List[tuple[str, bool]]:
    """Per-job pass/fail checks shown in the explanation panel."""
    title = str(row.get("title", "")).lower()
    company = str(row.get("company", "")).lower()
    exp_lvl = str(row.get("experience_level", "")).lower()
    work_t = str(row.get("work_type", "")).lower()
    desc = str(row.get("description", "")).lower()[:1200]
    text = f"{title} {company} {exp_lvl} {work_t} {desc}"
    checks: List[tuple[str, bool]] = []

    ml_mode = _is_ml_target_mode(prefs)
    if ml_mode:
        checks.append(("ML/AI-focused title", _has_ml_focused_role(row, prefs.target_roles)))

    dealbreakers = [d.lower() for d in prefs.dealbreakers]
    if any(d in ("senior", "staff", "principal", "director", "vp") for d in dealbreakers):
        senior_terms = ["senior", "sr.", "staff", "principal", "director", "vp", "lead "]
        checks.append(("No Senior/Staff-level signal", not any(term in text for term in senior_terms)))
    if any(d in ("junior", "entry", "entry level", "internship") for d in dealbreakers):
        junior_terms = ["junior", "jr.", "entry", "entry level", "intern", "internship", "new grad"]
        checks.append(("No Junior/Entry-level signal", not any(term in text for term in junior_terms)))
    if _is_company_size_mode(prefs):
        checks.append(("No tiny-startup signal (<100 employees proxy)", not any(term in text for term in TINY_STARTUP_TERMS)))
    if any(d in ("defense", "defence", "military") for d in dealbreakers):
        checks.append(("No defense/military signal", not any(term in text for term in DEFENSE_COMPANY_TERMS + ["defense", "defence", "military"])))
    if any(d in ("contract", "1099", "temporary", "temp", "unpaid") for d in dealbreakers):
        checks.append(("No contract/temp/unpaid signal", not any(term in text for term in ["contract", "1099", "temporary", "temp ", "unpaid"])))
    if prefs.visa_required:
        checks.append(("Large company / research lab / visa-sponsor signal", _has_large_company_or_research_signal(row)))
    if any(d in ("3+ years", "5+ years", "7+ years", "10+ years") for d in dealbreakers):
        checks.append(("Experience requirement is not too high", not EXPERIENCE_RE.search(text)))

    return checks


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
        "### Pass Criteria Check",
    ]

    criteria = describe_pass_criteria(prefs)
    if criteria:
        lines.append("This ranking is being filtered against:")
        lines.extend(f"- {criterion}" for criterion in criteria[:8])
    checks = evaluate_pass_criteria(row, prefs)
    if checks:
        lines.append("")
        for label, passed in checks:
            status = "PASS" if passed else "FAIL"
            lines.append(f"- **{status}** — {label}")

    lines += [
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
