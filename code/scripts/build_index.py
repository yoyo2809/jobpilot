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
CSV_ZIP_PATH = Path(__file__).parent.parent.parent / "data" / "jobs_snapshot.zip"


def ensure_readable_csv() -> bool:
    """Use the zipped snapshot if the plain CSV is missing or corrupted."""
    import zipfile
    import pandas as pd

    if CSV_PATH.exists():
        try:
            pd.read_csv(CSV_PATH, nrows=5)
            return True
        except Exception as exc:
            print(f"⚠️  Existing CSV is not readable: {exc}")

    if CSV_ZIP_PATH.exists():
        print(f"Extracting readable snapshot from {CSV_ZIP_PATH} ...")
        with zipfile.ZipFile(CSV_ZIP_PATH, "r") as zf:
            members = [m for m in zf.namelist() if m.endswith(".csv")]
            if not members:
                return False
            CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(members[0]) as src, open(CSV_PATH, "wb") as dst:
                dst.write(src.read())
        return True
    return False


def main():
    # 1. Load CSV into SQLite
    if ensure_readable_csv():
        print(f"Loading {CSV_PATH} into SQLite ...")
        db.initialize_db()
        db.load_csv_to_db(str(CSV_PATH), replace=True)
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
