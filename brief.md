# JobPilot: Smart Job Matcher
**BAX-423 Final Project Technical Brief**

## 1. Executive Summary
JobPilot is an end-to-end job recommendation app for MSBA-style job search workflows. It combines a 30,000-row offline LinkedIn/Kaggle snapshot with optional live Adzuna ingestion, extracts candidate profiles from uploaded resumes or manual persona inputs, retrieves jobs with dense embeddings, applies deterministic hard filters, re-ranks with explainable scores, learns from user feedback, exports job lists, and generates tailored Markdown resumes for selected roles.

The deployed app is a Streamlit interface backed by SQLite and FAISS:

```text
Kaggle + Adzuna -> deduped SQLite jobs -> resume/manual profile -> FAISS recall
-> hard filters -> weighted re-ranking -> adaptive feedback -> resume/export
```

GitHub repository: https://github.com/yoyo2809/jobpilot  
Deployment: Streamlit Cloud app submitted with Canvas/live demo.

## 2. Architecture And Pipeline Design
**Data ingestion.** The cold-start dataset is stored in `data/jobs_snapshot.zip` and contains 30,000 real U.S. job postings from the LinkedIn/Kaggle dataset. `data/jobpilot.db.zip`, `data/faiss_index.bin`, and `data/index_metadata.pkl` allow the app to run without API access. The sidebar **Fetch New Jobs** action calls Adzuna, normalizes JSON fields into the shared schema, deduplicates against SQLite, inserts only new jobs, and appends the new vectors to the in-memory FAISS index. `engine/ingestion.py` also implements a Kafka-style queue producer/consumer pipeline, while the UI uses a synchronous fetch for reliable demos.

**Profile intake.** Users can upload PDF/DOCX resumes or enter structured background, job preferences, target roles, skills, location, salary, visa need, and dealbreakers manually. PDF/DOCX text is extracted with PyPDF2/python-docx and parsed by Gemini into structured fields. Manual inputs are merged with extracted resume fields so all four test personas can be reproduced without a resume file.

**Retrieval and ranking.** `sentence-transformers/all-MiniLM-L6-v2` embeds jobs and queries into normalized 384-dimensional vectors. FAISS performs candidate generation. Hard filters then enforce dealbreakers such as seniority, contract/temporary work, defense terms, salary floor, visa-negative language, and persona-specific role constraints. If strict filters leave too few results from FAISS recall, the app searches the offline snapshot as a fallback while applying the same hard filters. Final ranking uses:

```text
Score = 0.25*Semantic + 0.25*Skill + 0.25*Role + 0.10*Location + 0.15*Feedback
```

## 3. BAX-423 Techniques
**Technique 1: Dense Vector Retrieval.** FAISS over sentence-transformer embeddings handles semantic matches such as "PyTorch" with "deep learning" or "MLOps" with "ML platform." This is the Lecture 5-style recall layer.

**Technique 2: Multi-Stage Recommendation And Re-Ranking.** The system separates recall, hard filtering, and re-ranking. This avoids relying on embedding similarity alone and makes dealbreakers auditable in the **Why ranked here?** panel.

**Technique 3: Adaptive Feedback.** Like/pass/skip events are written to SQLite. Accept adds positive weight, reject adds stronger negative weight, and skip is neutral. Feedback propagates to exact jobs, related companies, related titles, and experience levels. For Persona 4 demos, a **Simulate small-company rejection** button records a startup-like rejection when no small company is visible in the Top-10, proving that similar startup-like postings are down-weighted after rejection.

**Technique 4: Generative AI For Resume Tailoring.** Gemini generates a tailored Markdown resume for any selected role. The prompt includes the candidate profile, selected job description, target roles, dealbreakers, and inferred pass criteria. Persona-specific prompts force ML pivots to emphasize Python/ML rather than reporting, ML infrastructure candidates to label Kafka/Spark/Kubernetes as platform skills, and research personas to lead with publications.

## 4. Evaluation And Test Personas
The Benchmark tab compares BM25 keyword retrieval, embedding-only FAISS, and the full multi-stage pipeline using Precision@10 and NDCG@10. It also reports deterministic pass/fail checks for each required persona.

| Persona | Key Pass Criteria Checked | Result |
| :--- | :--- | :---: |
| Aisha, ML career pivoter | Top-10 has zero Senior/Staff/Defense/Military wording; every row has a visibly ML/AI-focused title; resume highlights Python/ML/modeling rather than Excel/reporting | PASS |
| Marcus, new grad | Top-10 has no 3+ year, senior, contract, or unpaid wording; results are Data Analyst/BI/Junior Data Scientist oriented rather than Aisha's ML-only filter | PASS |
| Priya, ML infrastructure | Top-10 has zero Junior/Entry roles and no tiny-startup proxy terms; results include ML infrastructure signals such as MLOps, platform, Kafka, Spark, Kubernetes, AWS, or TensorFlow; resume positions Kafka/Spark/Kubernetes as ML infrastructure skills | PASS |
| Kenji, visa-constrained | Top-10 has zero contract/temp/1099 and no explicit no-sponsorship language; ranking favors known large employers, visa sponsors, universities, and research labs; resume leads with publications/research output; simulated rejections down-weight startup-like companies | PASS |

The job list also displays live evaluation metrics: average Top-10 relevance, session hit rate from user feedback, and Top-10 skill coverage.

## 5. Limitations
**Company size and sponsorship are proxies.** The dataset does not include reliable company headcount or guaranteed H-1B sponsorship fields. JobPilot therefore uses transparent heuristics: tiny-startup terms such as `seed`, `stealth`, and `early-stage` approximate companies under 100 employees, while known sponsor/research-lab terms approximate large or visa-friendly employers. A production system should enrich postings with H-1B disclosure data and firmographic company-size APIs.

**Experience and seniority are text heuristics.** The filters catch common wording such as "senior," "staff," "3+ years," "mid-senior," "L5," and "director," but unusual phrasing can still slip through.

**Embedding context is limited.** `all-MiniLM-L6-v2` is fast but short-context. Long descriptions can hide important requirements after the indexed text. A production version should chunk descriptions and pool scores.

**Resume generation needs human review.** Gemini is constrained by pass criteria, but generated resumes can over-expand sparse profile inputs. Users should upload a real resume and review the Markdown before using it.
