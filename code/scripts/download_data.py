"""
scripts/download_data.py
Download & preprocess job postings from Kaggle → data/jobs_snapshot.csv

Dataset: arshkon/linkedin-job-postings
  - CSV format, ~124k records, ~135MB download
  - Much more practical than techmap (7.4GB JSON)

Usage:
    python scripts/download_data.py
"""
import os
import sys
import shutil
from pathlib import Path

import pandas as pd
import kagglehub
import datetime

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent.parent
DATA_DIR    = ROOT / "data"
OUTPUT_CSV  = DATA_DIR / "jobs_snapshot.csv"

DATASET     = "arshkon/linkedin-job-postings"
TARGET_ROWS = 30_000   # Exactly 30k US jobs


def preprocess(csv_path: Path) -> pd.DataFrame:
    """Clean and standardise the raw LinkedIn CSV, isolating US English jobs."""
    print(f"Processing {csv_path} ...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  Raw rows: {len(df):,}")

    # ── Rename to our schema ──────────────────────────────────────────────────
    col_map = {
        "job_id":                    "id",
        "title":                     "title",
        "company_name":              "company",
        "location":                  "location",
        "description":               "description",
        "max_salary":                "salary_max",
        "min_salary":                "salary_min",
        "formatted_experience_level":"experience_level",
        "work_type":                 "work_type",
        "skills_desc":               "skills",
        "job_posting_url":           "apply_url",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # ── Clean ─────────────────────────────────────────────────────────────────
    df["id"]          = df["id"].astype(str)
    df["source"]      = "LinkedIn/Kaggle"
    df["title"]       = df["title"].fillna("Unknown").astype(str).str.strip()
    df["company"]     = df["company"].fillna("Unknown").astype(str).str.strip()
    df["location"]    = df["location"].fillna("").astype(str)
    df["description"] = df["description"].fillna("").astype(str)
    df["date_posted"] = datetime.datetime.now().isoformat()

    # Keep only rows with an English description (>100 chars)
    df = df[df["description"].str.len() > 100]

    # Filter for US locations heuristically (2-letter state codes, or 'United States')
    us_states = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY']
    
    # We create a regex pattern that looks for State Code at the end of the string, or "United States"
    # Example matches: "San Francisco, CA" or "United States"
    pattern = '|'.join([f' {s}$' for s in us_states] + [f' {s},' for s in us_states] + ['United States'])
    
    # Apply location filter
    print("  Applying US location filter...")
    df = df[df["location"].str.contains(pattern, na=False, regex=True)]

    # ── Sample exactly 30,000 ─────────────────────────────────────────────────
    if len(df) > TARGET_ROWS:
        df = df.sample(TARGET_ROWS, random_state=42)
    elif len(df) < TARGET_ROWS:
        print(f"⚠️ Warning: Only found {len(df)} US jobs, which is less than requested 30,000.")

    print(f"  After cleaning: {len(df):,} US rows")
    
    # Ensure required columns exist
    out_cols = ["id","title","company","location","description",
               "salary_min","salary_max","experience_level",
               "work_type","skills","apply_url","source","date_posted"]
    for col in out_cols:
        if col not in df.columns:
            df[col] = None
            
    return df[out_cols]


def main():
    DATA_DIR.mkdir(exist_ok=True)
    
    print(f"Downloading {DATASET} using kagglehub...")
    # kagglehub handles auth via KAGGLE_API_TOKEN environment variable automatically
    dataset_path = kagglehub.dataset_download(DATASET)
    print(f"Dataset downloaded to: {dataset_path}")

    csv_path = Path(dataset_path) / "postings.csv"
    if not csv_path.exists():
        print(f"❌ Could not find postings.csv in {dataset_path}")
        sys.exit(1)

    df = preprocess(csv_path)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Saved {len(df):,} US jobs → {OUTPUT_CSV}")
    print(f"   File size: {OUTPUT_CSV.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
