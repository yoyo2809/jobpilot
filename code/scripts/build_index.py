"""
scripts/build_index.py
Build the FAISS embedding index from jobs_snapshot.csv.

Run this ONCE after downloading data:
    python scripts/build_index.py

Output:
    data/faiss_index.bin
    data/index_metadata.pkl
"""
import sys
from pathlib import Path

# Make engine importable from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine import database as db
from engine.embeddings import EmbeddingEngine

CSV_PATH = Path(__file__).parent.parent.parent / "data" / "jobs_snapshot.csv"


def main():
    # 1. Load CSV into SQLite
    if CSV_PATH.exists():
        print(f"Loading {CSV_PATH} into SQLite ...")
        db.initialize_db()
        db.load_csv_to_db(str(CSV_PATH))
    else:
        print(f"⚠️  {CSV_PATH} not found. Run scripts/download_data.py first.")
        if db.get_job_count() == 0:
            sys.exit(1)
        print(f"   Using existing DB ({db.get_job_count():,} jobs).")

    # 2. Load jobs for embedding
    print("Loading jobs from DB ...")
    jobs_df = db.get_all_jobs_for_indexing()
    print(f"  {len(jobs_df):,} jobs to index")

    # 3. Build FAISS index
    engine = EmbeddingEngine()
    engine.build_index(jobs_df)

    print(f"\n✅ Index built successfully!")
    print(f"   data/faiss_index.bin  — FAISS binary")
    print(f"   data/index_metadata.pkl — job ID mapping")


if __name__ == "__main__":
    main()
