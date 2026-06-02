"""
engine/embeddings.py
BAX-423 Lecture 5 — Embeddings & Vector Semantics

Sentence-Transformer embeddings + FAISS ANN index for job retrieval.
This is the core Lecture 5 technique.
"""
import numpy as np
import faiss
import pickle
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

MODEL_NAME   = "all-MiniLM-L6-v2"   # 384-dim, fast, free
INDEX_PATH   = Path(__file__).parent.parent / "data" / "faiss_index.bin"
META_PATH    = Path(__file__).parent.parent / "data" / "index_metadata.pkl"
EMBED_DIM    = 384


def _build_search_text(row: pd.Series) -> str:
    """Combine job fields into one searchable string."""
    parts = [
        str(row.get("title", "")),
        str(row.get("company", "")),
        str(row.get("location", "")),
        str(row.get("skills", "") or ""),
        str(row.get("description", "") or "")[:500],  # truncate description
    ]
    return " ".join(p for p in parts if p and p != "nan")


class EmbeddingEngine:
    """
    Wraps sentence-transformers + FAISS for semantic job retrieval.
    BAX-423 Lecture 5 technique.
    """

    def __init__(self):
        self.model: SentenceTransformer | None = None
        self.index: faiss.Index | None = None
        self.job_ids: List[str] = []    # parallel list to index rows

    def load_model(self):
        if self.model is None:
            self.model = SentenceTransformer(MODEL_NAME)

    def encode(self, texts: List[str], batch_size: int = 64,
               show_progress: bool = False) -> np.ndarray:
        """Encode a list of texts → L2-normalised float32 embeddings."""
        self.load_model()
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,   # L2 norm → cosine via inner product
        )
        return embeddings.astype(np.float32)

    # ------------------------------------------------------------------
    # Index build / persist
    # ------------------------------------------------------------------

    def build_index(self, df: pd.DataFrame):
        """
        Build FAISS IndexFlatIP (exact inner-product = cosine for
        normalised vectors) from a jobs DataFrame.
        """
        print(f"Building FAISS index for {len(df):,} jobs ...")
        texts = [_build_search_text(row) for _, row in df.iterrows()]
        embeddings = self.encode(texts, batch_size=128, show_progress=True)

        index = faiss.IndexFlatIP(EMBED_DIM)
        index.add(embeddings)

        self.index   = index
        self.job_ids = list(df["id"].astype(str))

        self._save()
        print(f"✅ FAISS index built: {index.ntotal:,} vectors")

    def _save(self):
        INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(INDEX_PATH))
        with open(META_PATH, "wb") as f:
            pickle.dump(self.job_ids, f)

    def load_index(self) -> bool:
        """Load pre-built index from disk. Returns True if successful."""
        if INDEX_PATH.exists() and META_PATH.exists():
            self.index = faiss.read_index(str(INDEX_PATH))
            with open(META_PATH, "rb") as f:
                self.job_ids = pickle.load(f)
            return True
        return False

    def index_ready(self) -> bool:
        return self.index is not None and len(self.job_ids) > 0

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query_text: str, k: int = 200) -> List[Tuple[str, float]]:
        """
        Retrieve top-k job IDs most similar to query_text.
        Returns [(job_id, cosine_score), ...]
        """
        if not self.index_ready():
            return []
        query_emb = self.encode([query_text])
        scores, indices = self.index.search(query_emb, min(k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((self.job_ids[idx], float(score)))
        return results

    # ------------------------------------------------------------------
    # Incremental add (for streaming new jobs)
    # ------------------------------------------------------------------

    def add_jobs(self, new_jobs: pd.DataFrame):
        """Incrementally add new jobs to an existing index."""
        if not self.index_ready():
            return
        texts = [_build_search_text(row) for _, row in new_jobs.iterrows()]
        embs  = self.encode(texts)
        self.index.add(embs)
        self.job_ids.extend(new_jobs["id"].astype(str).tolist())
        self._save()


# Module-level singleton (loaded once per process by Streamlit cache)
_engine = EmbeddingEngine()


def get_engine() -> EmbeddingEngine:
    return _engine
