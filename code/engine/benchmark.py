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
from engine.ranking import UserPreferences, rank_jobs


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
        "skills": ["Python", "SQL", "pandas", "scikit-learn", "PyTorch"],
        "dealbreakers": ["Senior", "Staff", "Defense"],
        "location": "United States",
    },
    "Marcus (New Grad)": {
        "query": "data analyst business intelligence junior entry level SQL Python",
        "skills": ["Python", "R", "SQL", "Tableau", "PySpark"],
        "dealbreakers": ["3+ years", "5+ years", "Contract"],
        "location": "United States",
    },
    "Priya (Experienced Niche)": {
        "query": "MLOps engineer ML platform Kafka Spark Kubernetes senior",
        "skills": ["Java", "Python", "Kubernetes", "Kafka", "Spark", "TensorFlow", "AWS"],
        "dealbreakers": ["Junior", "Entry"],
        "location": "United States",
    },
    "Kenji (International/Visa)": {
        "query": "research scientist ML engineer deep learning NLP PyTorch computer vision",
        "skills": ["Python", "C++", "PyTorch", "NLP", "Computer Vision"],
        "dealbreakers": ["Contract", "1099", "Temporary"],
        "location": "United States",
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
    jobs_df  = db.get_all_jobs_for_indexing()
    bm25_ret = BM25Retriever()
    bm25_ret.fit(jobs_df)

    rows = []
    for persona, cfg in PERSONA_QUERIES.items():
        query = cfg["query"]
        prefs = UserPreferences(
            location     = cfg["location"],
            skills       = cfg["skills"],
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

        # Define relevant set: jobs that contain at least 2 skill keywords
        relevant = _find_relevant(jobs_df, cfg["skills"], cfg["dealbreakers"])

        rows.append({
            "Persona":             persona,
            "BM25 P@10":          f"{precision_at_k(bm25_ids, relevant, k):.2f}",
            "BM25 NDCG@10":       f"{ndcg_at_k(bm25_ids, relevant, k):.2f}",
            "Embedding P@10":     f"{precision_at_k(emb_ids, relevant, k):.2f}",
            "Embedding NDCG@10":  f"{ndcg_at_k(emb_ids, relevant, k):.2f}",
            "Multi-Stage P@10":   f"{precision_at_k(l7_ids, relevant, k):.2f}",
            "Multi-Stage NDCG@10":f"{ndcg_at_k(l7_ids, relevant, k):.2f}",
        })

    return pd.DataFrame(rows)


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
