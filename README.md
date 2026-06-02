# JobPilot 🚀
**AI-powered Job Matcher & Resume Builder · BAX-423 Final Project · Option B**

> Built by [Your Name] · UC Davis MSBA · Spring 2026

## Live App
🔗 **[https://jobpilot-bax423.streamlit.app](https://jobpilot-bax423.streamlit.app)** *(add URL after deployment)*

---

## Quick Start (Local)

### 1. Install dependencies
```bash
pip3 install -r requirements.txt
```

### 2. Set up API keys
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit .streamlit/secrets.toml and add your keys
```

### 3. Download job data (~2 min)
```bash
python3 scripts/download_data.py
```

### 4. Build FAISS search index (~3 min, one-time)
```bash
python3 scripts/build_index.py
```

### 5. Run the app
```bash
/Users/yoyo/Library/Python/3.9/bin/streamlit run app.py
# or: python3 -m streamlit run app.py
```

Open http://localhost:8501

---

## Architecture

```
User Resume (PDF/DOCX)
        ↓
  Gemini 1.5 Flash
  (Profile Extraction)
        ↓
Multi-Stage Ranking Pipeline
  ├── Stage 1: FAISS Embedding Retrieval  ← BAX-423 Lecture 5
  │            (sentence-transformers, all-MiniLM-L6-v2)
  ├── Stage 2: Hard Filters
  │            (seniority, dealbreakers, salary floor)
  └── Stage 3: Weighted Re-rank           ← BAX-423 Lecture 7
               (embedding 40% + skills 30% + location 15% + feedback 15%)
        ↓
  Adaptive Learning
  (accept/reject/skip → re-weight)
        ↓
  Tailored Resume Generation (Gemini)
```

## BAX-423 Techniques Used

| Technique | Lecture | Implementation |
|-----------|---------|----------------|
| Embedding-Based Retrieval | Lecture 5 | `sentence-transformers` + `FAISS IndexFlatIP` |
| Multi-Stage Ranking | Lecture 7 | 3-stage pipeline (recall → filter → re-rank) |
| Streaming Ingestion | Lecture 3 | Kafka-style producer/consumer (`threading.Queue`) |

## Data Sources

- **Primary**: LinkedIn Job Postings (Kaggle, `arshkon/linkedin-job-postings`) — 25,000 job snapshot
- **Live**: Adzuna REST API — real-time postings ingestion

## Project Structure

```
jobpilot/
├── app.py                    # Main Streamlit app
├── pages/
│   └── 1_📊_Analytics.py    # Job market analytics dashboard
├── engine/
│   ├── database.py           # SQLite layer
│   ├── embeddings.py         # FAISS + sentence-transformers (Lecture 5)
│   ├── ranking.py            # Multi-stage ranking (Lecture 7)
│   ├── profile.py            # Resume parser (Gemini)
│   ├── resume_gen.py         # Resume generator (Gemini)
│   ├── feedback.py           # Adaptive learning
│   ├── ingestion.py          # Streaming pipeline (Lecture 3)
│   └── benchmark.py          # BM25 vs Embedding vs Re-rank benchmark
├── scripts/
│   ├── download_data.py      # Kaggle data download + preprocess
│   └── build_index.py        # FAISS index builder
├── data/
│   ├── jobs_snapshot.csv     # 25k job offline snapshot
│   ├── faiss_index.bin       # Pre-built FAISS index
│   └── index_metadata.pkl    # Job ID mapping
├── .streamlit/secrets.toml   # API keys (not committed)
├── requirements.txt
└── prompts.md
```

## Test Personas

| Persona | Background | Key Pass Criteria |
|---------|-----------|-------------------|
| Aisha | ML Engineer career pivoter | No Senior/Staff, all ML-related |
| Marcus | New grad MSBA | No 3+ yr requirements, leads with education |
| Priya | Senior SWE → MLOps | No Junior titles, Kafka framed as ML infra |
| Kenji | International student (visa) | No contract roles, favors large companies |

## Submission

```bash
zip -r LastName_FirstName_BAX423_Final.zip code/ data/ brief.pdf prompts.md
```
