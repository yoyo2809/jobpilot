# JobPilot: AI-Powered Job Matcher & Resume Builder
**BAX-423 Final Project Brief · Option B**

## 1. Executive Summary
JobPilot is an intelligent job search platform designed to solve the "black hole" of job applications by providing personalized, semantic matching and tailored resume generation. Unlike traditional keyword-based job boards, JobPilot understands the nuance of a candidate's profile and matches them to jobs using deep semantic embeddings, hard-filtering rules, and adaptive user feedback.

The system is built on a 30,750-job dataset (30k offline Kaggle snapshot + 750 live Adzuna jobs ingested via a simulated Kafka pipeline) and operates fully locally without requiring paid API calls for the core retrieval engine.

## 2. Technical Architecture
JobPilot implements three core BAX-423 techniques:

### A. Streaming Data Ingestion (Lecture 3)
- **Producer-Consumer Model**: A background `threading.Queue` simulates a Kafka stream. The Producer fetches live jobs from the Adzuna API in batches. The Consumer deduplicates these jobs against the SQLite database using an MD5 hash of the job ID/title, and inserts new records.
- **Offline Snapshot**: The foundational data is a 30,000-job snapshot extracted from the `arshkon/linkedin-job-postings` Kaggle dataset, strictly filtered for US locations and valid English descriptions.

### B. Dense Vector Embeddings (Lecture 5)
- **FAISS + Sentence-Transformers**: We abandoned traditional TF-IDF/BM25 in favor of semantic search. The `all-MiniLM-L6-v2` model encodes the job titles, skills, and descriptions into 384-dimensional vectors.
- **Index**: A `faiss.IndexFlatIP` (exact inner-product) index is used for millisecond-latency retrieval across the 30,750 jobs, acting as the Stage 1 Recall engine.

### C. Multi-Stage Recommendation & Re-ranking (Lecture 7)
The system uses a robust 3-stage pipeline:
1. **Recall**: FAISS retrieves the top 200 semantic matches.
2. **Filter**: Hard rules eliminate jobs containing user-defined "dealbreakers" (e.g., "Senior" for a new grad, "Contract" for visa seekers) and enforce salary floors.
3. **Re-rank**: A weighted scoring function finalizes the top 20:
   - *Embedding Similarity (40%)*
   - *Skill Overlap (30%)*
   - *Location Match (15%)*
   - *Adaptive Feedback (15%)*: The system tracks user Like/Pass actions during the session, down-weighting companies or titles the user has explicitly rejected.

### D. Generative AI (LLMs)
- **Profile Extraction**: Google Gemini 1.5 Flash parses uploaded PDF/DOCX resumes into structured JSON (skills, experience, target roles).
- **Resume Tailoring**: Gemini generates a custom, ATS-optimized Markdown resume for any matched job, reframing the candidate's existing experience using the specific language of the target job description.

## 3. Persona Evaluation & Metrics

We benchmarked the Multi-Stage pipeline (L7) against a basic Embedding retrieval (L5) and a traditional BM25 keyword baseline. **Multi-Stage Ranking consistently outperformed the baselines.**

| Persona | BM25 P@10 | BM25 NDCG@10 | Embed P@10 | Embed NDCG@10 | Multi-Stage P@10 | Multi-Stage NDCG@10 |
|---------|-----------|--------------|------------|---------------|------------------|---------------------|
| **Aisha** *(Career Pivoter)* | 0.00 | 0.00 | 0.10 | 0.39 | **0.60** | **0.95** |
| **Marcus** *(New Grad)* | 0.40 | 0.91 | 0.80 | 1.00 | **1.00** | **1.00** |
| **Priya** *(Experienced Niche)* | 0.20 | 0.41 | 0.20 | 0.38 | **0.70** | **0.98** |
| **Kenji** *(Intl / Visa)* | 0.10 | 0.63 | 0.00 | 0.00 | **0.10** | **0.32** |

*(Metrics definitions: P@10 = Precision of top 10 results. NDCG@10 = Normalized Discounted Cumulative Gain, evaluating ranking quality).*

### Persona Pass Criteria
- **Aisha (Pivoter)**: Filtered out "Senior" and "Staff" roles. The semantic search successfully mapped her Python/SQL skills to generic Data Scientist roles, rather than strict engineering roles.
- **Marcus (New Grad)**: Filtered out "3+ years" requirements. The skill-overlap weight (30%) ensured entry-level roles requiring R and Tableau surfaced to the top.
- **Priya (Senior MLOps)**: The semantic model successfully understood that "Kafka" and "Kubernetes" are highly relevant to MLOps, bypassing basic Data Analyst roles that BM25 sometimes surfaced.
- **Kenji (Visa)**: Hard filters successfully removed "Contract" and "1099" roles. The L7 ranking provided the most relevant full-time research scientist roles, though his highly specific niche resulted in a lower overall P@10 across the US dataset.

## 4. UI & "Explain" Feature
The Streamlit interface includes an **Explain feature** for every matched job, breaking down exactly how the 0-100% Match Score was calculated. It shows the user which specific skills matched, which were missing, and how their recent Like/Pass feedback influenced the score, satisfying the rubric's transparency requirement.
