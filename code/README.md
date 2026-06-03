# JobPilot
**BAX-423 Final Project · Option B · Smart Job Matcher**

JobPilot is a Streamlit app for job ingestion, semantic retrieval, multi-stage ranking, adaptive feedback, analytics, and tailored resume generation.

## Run Locally

From the submitted project root:

```bash
cd code
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Fill `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your_gemini_key"
ADZUNA_APP_ID = "your_adzuna_app_id"
ADZUNA_APP_KEY = "your_adzuna_app_key"
```

Initialize data and index:

```bash
python3 scripts/build_index.py
streamlit run app.py
```

If `data/jobs_snapshot.csv` is missing or corrupted, `scripts/build_index.py` automatically restores it from `data/jobs_snapshot.zip`.

## Streamlit Cloud

Deploy from the GitHub repo with:

```text
Main file path: code/app.py
```

Add the same three secrets in Streamlit Cloud settings.

## What To Verify

Data ingestion:
Click **Fetch New Jobs** in the sidebar. The app calls Adzuna, deduplicates incoming jobs against SQLite, inserts new rows, and adds them to the FAISS index for the current session.

Profile and preferences:
Use either resume upload or manual structured inputs for background, target roles, skills, location, salary, visa need, and dealbreakers.

Ranking:
Click **Find Matches**. The pipeline is:

```text
FAISS candidate generation -> hard filters -> weighted re-ranking -> feedback adaptation
```

Explainability:
Open **Why ranked here?** under a job card to see embedding, skill, role, location, and feedback scores.

Evaluation:
Use the **Benchmark** tab. It reports Precision@10, NDCG@10, and a deterministic Pass/Fail check for the four required personas.

Download:
Use **Download Top Jobs** to export CSV, Excel, or JSON with title, company, location, salary, link, match score, and full job description text.

Resume generation:
Click **Generate Resume** on any selected job. Gemini generates a Markdown resume based on the candidate profile plus the selected job description.

## Project Structure

```text
code/
  app.py
  engine/
    database.py
    embeddings.py
    ranking.py
    profile.py
    feedback.py
    ingestion.py
    benchmark.py
    resume_gen.py
  pages/
  scripts/
  requirements.txt
data/
  jobs_snapshot.zip
  jobpilot.db.zip
  faiss_index.bin
  index_metadata.pkl
brief.pdf
prompts.md
```

## Notes

The app uses `sentence-transformers/all-MiniLM-L6-v2`; the first run may need internet access to download the model unless it is already cached. API secrets are intentionally not committed.
