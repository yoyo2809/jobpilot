"""
scripts/fetch_live_jobs.py
Fetch 500+ live jobs from Adzuna API to supplement the offline dataset.
"""
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine import ingestion
from engine import database as db

def main():
    print("Fetching live jobs from Adzuna...")
    # Fetch 15 pages of 50 results each = 750 jobs
    # Use general tech search terms to get a wide variety
    stats = ingestion.manual_fetch(query="data software engineer manager", max_pages=15)
    
    print("\n✅ Adzuna Fetch Complete")
    print(f"Total fetched: {stats['fetched']}")
    print(f"New inserts:   {stats['new']}")
    print(f"Duplicates:    {stats['dupes']}")
    
    count = db.get_job_count()
    print(f"\nTotal jobs in database now: {count:,}")

if __name__ == "__main__":
    main()
