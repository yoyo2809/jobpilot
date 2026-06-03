# JobPilot 🚀
**AI-powered Job Matcher & Resume Builder · BAX-423 Final Project · Option B**

> Built by [Your Name] · UC Davis MSBA · Spring 2026

## 🌟 Live App
🔗 **[https://jobpilot-bax423.streamlit.app](https://jobpilot-bax423.streamlit.app)** *(Replace with your URL if deploying to Streamlit Cloud)*

---

## 💻 System Requirements
- **Python**: 3.9 or higher (Tested on Python 3.9.6)
- **OS**: macOS / Windows / Linux

## 🛠️ Quick Start (Local Setup)

### 1. Install Dependencies
It is highly recommended to use a virtual environment to avoid conflicts.
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure API Keys
The system requires API keys for LLM parsing and Real-Time job fetching.
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```
Open `.streamlit/secrets.toml` in your IDE and populate it with your credentials:
```toml
# Google Gemini API (Required for Resume Parsing)
GEMINI_API_KEY = "your_google_api_key_here"

# Adzuna API (Required for Real-time Job Streaming)
ADZUNA_APP_ID = "your_adzuna_id_here"
ADZUNA_APP_KEY = "your_adzuna_key_here"
```

### 3. Initialize Data Pipeline (~2 min)
This script downloads the offline Kaggle dataset and initializes the SQLite database to solve the cold-start problem.
```bash
python3 scripts/download_data.py
```

### 4. Build FAISS Vector Index (~3 min, one-time)
This script converts 30,000+ job descriptions into 384-dimensional dense vectors using HuggingFace's `all-MiniLM-L6-v2`.
```bash
python3 scripts/build_index.py
```

### 5. Launch the Application
```bash
cd code
streamlit run app.py
```
The application will automatically open at `http://localhost:8501`.

---

## 🔍 How to Grade / Verify Rubric Requirements

To assist the TA/Professor in verifying the BAX-423 techniques, please follow these steps in the UI:

1. **Streaming Pipeline (15 Pts)**: Expand the sidebar and click **"Fetch Real-Time Jobs"**. The app uses Producer/Consumer multi-threading to pull live data from Adzuna, deduplicate using MD5 hashes, and insert it into SQLite.
2. **Dense Embeddings vs. Sparse (20 Pts)**: Upload a sample resume. Click the `🔍 Why ranked here?` expander under any job card to see the L2 FAISS distance score powered by `all-MiniLM-L6-v2`.
3. **Multi-Stage Ranking & Benchmarks (15 Pts)**: Navigate to the **"Benchmark"** tab. Here you will find the offline evaluation results (NDCG@10) demonstrating how our Multi-Stage pipeline outperforms BM25 across 4 diverse Personas.
4. **Adaptive Learning (15 Pts)**: On the main dashboard, locate a job you dislike and click the `👎 Pass` button. Observe the metrics panel at the top. The system applies a Semantic Penalty Array to penalize similar jobs, dynamically updating the *Hit Rate*.

---

## 🏗️ Architecture

```text
User Resume (PDF)
        ↓
  Gemini 2.5 Flash
  (Profile Extraction JSON)
        ↓
Multi-Stage Ranking Pipeline
  ├── Stage 1: FAISS Embedding Retrieval  ← BAX-423 Lecture 5
  ├── Stage 2: Hard Filters (Location/Visa/Salary)
  └── Stage 3: Weighted Re-rank           ← BAX-423 Lecture 7
               (Semantic + Skills + Constraints)
        ↓
  Adaptive Learning
  (Implicit Feedback Array → Penalty)     ← BAX-423 Lecture 8
```

## 📂 Project Structure

```text
jobpilot/
├── app.py                    # Main Streamlit Frontend
├── engine/
│   ├── database.py           # SQLite persistence layer
│   ├── embeddings.py         # FAISS + sentence-transformers (L5)
│   ├── ranking.py            # Multi-stage ranking & weights (L7)
│   ├── profile.py            # Resume parser (Gemini)
│   ├── feedback.py           # Adaptive semantic learning (L8)
│   └── ingestion.py          # Adzuna Streaming pipeline (L3)
├── scripts/
│   ├── download_data.py      # Kaggle offline data processor
│   └── build_index.py        # FAISS index builder
├── data/                     # Local storage (SQLite, FAISS bin)
├── .streamlit/secrets.toml   # API keys (not committed)
├── requirements.txt          # Python dependencies
└── brief.pdf                 # Final Technical Brief (Max 4 Pages)
```

## 👨‍💻 Test Personas Evaluated
| Persona | Background | Key Edge Case Handled |
|---------|-----------|-------------------|
| **Aisha** | Marketing → Data Analytics | Dense Embeddings fixed vocabulary gap |
| **Marcus**| New grad, zero experience | Adaptive Learner penalized Senior titles |
| **Priya** | Specialized niche stack | L3 Re-Ranker fixed embedding over-generalization |
| **Kenji** | Needs H1-B Visa sponsorship | Hard Filters eradicated false-positive matches |

## 📦 Final Submission
```bash
zip -r LastName_FirstName_BAX423_Final.zip code/ data/ brief.pdf
```
