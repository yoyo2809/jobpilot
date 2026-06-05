"""
engine/benchmark.py
BAX-423 Technique Benchmarking.

Compares three retrieval approaches for brief.pdf:
  Baseline  — BM25 keyword search (rank-bm25)
  L5        — Sentence-embedding FAISS retrieval
  L7        — Multi-stage re-ranking on top of L5

Metrics: Precision@K, NDCG@K
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import pandas as pd
from rank_bm25 import BM25Okapi

from engine import database as db
from engine.embeddings import EmbeddingEngine
from engine.ranking import (
    UserPreferences,
    rank_jobs,
    _has_large_company_or_research_signal,
    _has_ml_focused_role,
    TINY_STARTUP_TERMS,
)


# ── NDCG helper ───────────────────────────────────────────────────────────────

def dcg(relevances: List[int]) -> float:
    return sum(
        rel / math.log2(i + 2)
        for i, rel in enumerate(relevances)
    )


def ndcg_at_k(ranked_ids: List[str], relevant_ids: set, k: int = 10) -> float:
    top_k = ranked_ids[:k]
    rels  = [1 if jid in relevant_ids else 0 for jid in top_k]
    ideal = sorted(rels, reverse=True)
    d     = dcg(rels)
    i     = dcg(ideal)
    return d / i if i > 0 else 0.0


def precision_at_k(ranked_ids: List[str], relevant_ids: set, k: int = 10) -> float:
    top_k   = ranked_ids[:k]
    hits    = sum(1 for jid in top_k if jid in relevant_ids)
    return hits / k


# ── BM25 Baseline ─────────────────────────────────────────────────────────────

class BM25Retriever:
    def __init__(self):
        self.corpus: List[str]  = []
        self.ids: List[str]     = []
        self.bm25: BM25Okapi | None = None

    def fit(self, df: pd.DataFrame):
        self.ids    = list(df["id"].astype(str))
        self.corpus = [
            (str(r.get("title", "")) + " " +
             str(r.get("skills", "") or "") + " " +
             str(r.get("description", "") or "")[:400]).lower().split()
            for _, r in df.iterrows()
        ]
        self.bm25 = BM25Okapi(self.corpus)

    def search(self, query: str, k: int = 10) -> List[str]:
        if self.bm25 is None:
            return []
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.ids[i] for i in ranked[:k]]


# ── Benchmark runner ──────────────────────────────────────────────────────────

PERSONA_QUERIES = {
    "Aisha (Career Pivoter)": {
        "query": "machine learning engineer Python scikit-learn data scientist applied scientist",
        "background": "Data Analyst, 3 years at a mid-size retail company. Wants to pivot to ML Engineering.",
        "skills": ["Python", "SQL", "pandas", "scikit-learn", "PyTorch"],
        "dealbreakers": ["Senior", "Staff", "Defense", "Military"],
        "target_roles": ["Machine Learning Engineer", "Applied Scientist", "Data Scientist"],
        "location": "Bay Area",
        "min_salary": 140000,
    },
    "Marcus (New Grad)": {
        "query": "entry level data analyst BI analyst junior data scientist analytics engineer SQL Python Tableau",
        "background": "Recent MSBA graduate, UC Davis. No full-time experience. 2 analytics internships. Prefers tech or healthcare.",
        "skills": ["Python", "R", "SQL", "Tableau", "PySpark"],
        "dealbreakers": ["3+ years", "4+ years", "5+ years", "Contract", "Temporary", "Unpaid"],
        "target_roles": ["Data Analyst", "BI Analyst", "Junior Data Scientist", "Analytics Engineer"],
        "location": "United States",
        "min_salary": 80000,
    },
    "Priya (Experienced Niche)": {
        "query": "ML platform engineer MLOps engineer senior machine learning engineer Kafka Spark Kubernetes AWS TensorFlow",
        "background": "Senior Software Engineer, 7 years in fintech. Wants ML/AI infrastructure. Prefers NYC or remote and companies with 100+ employees. No companies with <100 employees.",
        "skills": ["Java", "Python", "Kubernetes", "Kafka", "Spark", "TensorFlow", "AWS"],
        "dealbreakers": ["Junior", "Entry"],
        "target_roles": ["ML Platform Engineer", "MLOps Engineer", "Senior ML Engineer"],
        "location": "NYC",
        "min_salary": 200000,
    },
    "Kenji (International/Visa)": {
        "query": "research scientist ML engineer deep learning NLP PyTorch computer vision",
        "background": "International MS Computer Science student on OPT. Needs H-1B sponsorship within one year. Published research.",
        "skills": ["Python", "C++", "PyTorch", "NLP", "Computer Vision"],
        "dealbreakers": ["Contract", "1099", "Temporary"],
        "target_roles": ["Research Scientist", "Machine Learning Engineer"],
        "location": "United States",
        "min_salary": 120000,
        "visa_required": True,
    },
}


def run_benchmark(
    engine: EmbeddingEngine,
    k: int = 10,
) -> pd.DataFrame:
    """
    Run all three approaches on all four personas.
    Returns a summary DataFrame for display in the brief.
    """
    if not engine.index_ready():
        engine.load_index()

    jobs_df  = db.get_all_jobs_for_indexing()
    bm25_ret = BM25Retriever()
    bm25_ret.fit(jobs_df)

    rows = []
    for persona, cfg in PERSONA_QUERIES.items():
        query = cfg["query"]
        prefs = UserPreferences(
            background    = cfg.get("background", ""),
            location     = cfg["location"],
            min_salary   = cfg.get("min_salary", 0),
            skills       = cfg["skills"],
            target_roles = cfg.get("target_roles", []),
            dealbreakers = cfg["dealbreakers"],
            visa_required = cfg.get("visa_required", False),
        )

        # Baseline: BM25
        bm25_ids = bm25_ret.search(query, k=k)

        # L5: Embedding FAISS
        emb_hits = engine.search(query, k=50)
        emb_ids  = [h[0] for h in emb_hits[:k]]

        # L7: Multi-stage (full pipeline)
        ranked_df = rank_jobs(query, prefs, engine, session_id="benchmark", top_n=k)
        l7_ids    = list(ranked_df["id"].astype(str)) if not ranked_df.empty else []
        pass_result, pass_notes = _persona_pass_check(persona, ranked_df, cfg)

        # Define relevant set: jobs that contain at least 2 skill keywords
        relevant = _find_relevant(jobs_df, cfg["skills"], cfg["dealbreakers"])
        print(f"[{persona}] Relevant set size: {len(relevant)}")

        rows.append({
            "Persona":             persona,
            "BM25 P@10":          f"{precision_at_k(bm25_ids, relevant, k):.2f}",
            "BM25 NDCG@10":       f"{ndcg_at_k(bm25_ids, relevant, k):.2f}",
            "Embedding P@10":     f"{precision_at_k(emb_ids, relevant, k):.2f}",
            "Embedding NDCG@10":  f"{ndcg_at_k(emb_ids, relevant, k):.2f}",
            "Multi-Stage P@10":   f"{precision_at_k(l7_ids, relevant, k):.2f}",
            "Multi-Stage NDCG@10":f"{ndcg_at_k(l7_ids, relevant, k):.2f}",
            "Pass Criteria":      "PASS" if pass_result else "FAIL",
            "Pass Notes":         pass_notes,
        })

    return pd.DataFrame(rows)


def _persona_pass_check(persona: str, ranked_df: pd.DataFrame, cfg: dict) -> tuple[bool, str]:
    """Deterministically inspect Top-10 for the persona-specific rubric rules."""
    if ranked_df.empty:
        return False, "No ranked jobs returned"

    top = ranked_df.head(10).copy()

    def clean_col(name: str) -> pd.Series:
        if name not in top.columns:
            return pd.Series([""] * len(top), index=top.index)
        return top[name].fillna("").astype(str)

    text = (
        clean_col("title") + " " +
        clean_col("company") + " " +
        clean_col("experience_level") + " " +
        clean_col("work_type") + " " +
        clean_col("description").str[:1000]
    ).str.lower().fillna("")
    header_text = (
        clean_col("title") + " " +
        clean_col("company") + " " +
        clean_col("experience_level") + " " +
        clean_col("work_type")
    ).str.lower().fillna("")

    def has_any(terms: list[str]) -> bool:
        return any(text.str.contains(term, regex=False, na=False).any() for term in terms)

    if "Aisha" in persona:
        blocked = ["senior", "staff", "defense", "defence", "military"]
        ml_focused = all(_has_ml_focused_role(row, cfg.get("target_roles", [])) for _, row in top.iterrows())
        return (not has_any(blocked) and ml_focused,
                "No senior/staff/defense; every Top-10 row is a focused ML/AI role")

    if "Marcus" in persona:
        bad_level_terms = ["mid-senior", "senior", "sr.", "lead ", "principal", "staff", "director"]
        bad_work_terms = ["contract", "contractor", "temporary", "temp ", "1099", "unpaid"]
        bad_level = any(header_text.str.contains(term, regex=False, na=False).any() for term in bad_level_terms)
        bad_work = any(header_text.str.contains(term, regex=False, na=False).any() for term in bad_work_terms)
        strict_exp_patterns = [
            "3+ years experience", "3+ years of experience",
            "4+ years experience", "4+ years of experience",
            "5+ years experience", "5+ years of experience",
            "minimum 3 years", "minimum of 3 years",
            "at least 3 years", "requires 3 years",
            "required 3 years", "3 years required",
        ]
        bad_experience = any(
            any(pattern in t for pattern in strict_exp_patterns)
            for t in text.tolist()
        )
        return (not bad_level and not bad_work and not bad_experience,
                "No 3+ year requirement, senior-level header, contract/temp/1099, or unpaid role in Top-10")

    if "Priya" in persona:
        blocked = ["junior", "entry", "intern"]
        bad_level = any(header_text.str.contains(term, regex=False, na=False).any() for term in blocked)
        no_tiny_startup = not has_any(TINY_STARTUP_TERMS)
        return (not bad_level and no_tiny_startup,
                "No junior/entry-level header; no tiny-startup proxy terms in Top-10")

    if "Kenji" in persona:
        blocked = ["contract", "1099", "temporary", "temp ", "no sponsorship", "us citizen", "green card"]
        sponsor_or_research = any(_has_large_company_or_research_signal(row) for _, row in top.iterrows())
        return (not has_any(blocked) and sponsor_or_research,
                "No contract/temp/1099/no-sponsorship wording; Top-10 includes large-company/research-lab signal")

    return True, "Generic benchmark row"


def _find_relevant(jobs_df: pd.DataFrame, skills: List[str],
                   dealbreakers: List[str]) -> set:
    """Heuristically mark jobs as relevant for benchmark ground truth."""
    relevant = set()
    for _, row in jobs_df.iterrows():
        text = (str(row.get("title", "")) + " " +
                str(row.get("skills", "") or "") + " " +
                str(row.get("description", "") or "")[:500]).lower()
        # Must have ≥2 skill keywords and no dealbreakers
        skill_hits = sum(1 for s in skills if s.lower() in text)
        has_db     = any(d.lower() in text for d in dealbreakers)
        if skill_hits >= 2 and not has_db:
            relevant.add(str(row["id"]))
    return relevant
