# prompts.md — Key AI Prompts Used in JobPilot

> BAX-423 Final Project · Option B · JobPilot  
> AI tools used: Google Gemini 1.5 Flash (via API), Antigravity IDE (Claude Sonnet)

---

## 1. Resume / Profile Extraction

**Prompt** (used in `engine/profile.py`):
```
You are a resume parser. Extract structured information from the resume below.

Return ONLY valid JSON with exactly these fields:
{
  "name": "Full Name",
  "skills": ["Python", "SQL", "Tableau"],
  "experience_years": 3,
  "education": "MS Business Analytics, UC Davis 2024",
  "target_roles": ["Data Analyst", "Business Intelligence Analyst"],
  "current_title": "Data Analyst",
  "location": "San Francisco, CA",
  "summary": "One sentence summary of the candidate."
}

Resume:
{resume_text}
```

**Purpose**: Convert unstructured PDF/DOCX text into a structured profile dict for skill matching and ranking.  
**Modification**: Truncated resume to 4000 characters; added JSON fence-stripping to handle markdown code blocks in Gemini output.

---

## 2. Tailored Resume Generation

**Prompt** (used in `engine/resume_gen.py`):
```
You are an expert resume writer and career coach.

Given the candidate's profile and the specific job description, write a complete,
tailored resume in Markdown format optimised for this role.

Rules:
- Lead with the candidate's strongest relevant experience for THIS specific role.
- Reframe existing experience using the language of the job description.
- Include a Professional Summary (3 sentences) mentioning the company name.
- List top 6-8 relevant skills first.
- Use strong action verbs and quantify achievements where possible.
- Do NOT invent qualifications the candidate doesn't have.

Candidate Profile:
{profile_summary}

Target Job:
Title: {job_title}
Company: {company}
...
```

**Purpose**: Generate a role-specific tailored resume that matches the job description language (ATS optimisation).  
**Modification**: Added explicit rule "Do NOT invent qualifications" to prevent hallucination; limited job description to 3000 chars to stay within token limits.

---

## 3. Cover Letter Generation (Bonus)

**Prompt** (used in `engine/resume_gen.py`):
```
Write a concise, compelling cover letter (3 paragraphs) for:
Candidate: {name}
Skills: {skills}
Applying to: {title} at {company}
Description: {description}
```

**Purpose**: Bonus feature — generates a 3-paragraph cover letter alongside the resume.  
**Modification**: Added "3 paragraphs" constraint to keep it concise.

---

## 4. Development Assistance (Antigravity IDE)

**Prompts used for code generation**:

- *"Build a multi-stage ranking pipeline using Lecture 5 embeddings and Lecture 7 re-ranking. Stage 1: FAISS recall. Stage 2: hard dealbreaker filter. Stage 3: weighted score combining embedding similarity, skill overlap, location match, and adaptive feedback."*  
  **Purpose**: Generated the core `engine/ranking.py` module.

- *"Create a streaming ingestion module that simulates Kafka producer/consumer using Python's threading.Queue. Producer fetches from Adzuna API. Consumer deduplicates and inserts into SQLite."*  
  **Purpose**: Generated `engine/ingestion.py`.

- *"Write a benchmark that compares BM25 baseline vs sentence-embedding FAISS retrieval vs multi-stage re-ranking using NDCG@10 and Precision@10 across 4 test personas."*  
  **Purpose**: Generated `engine/benchmark.py`.

- *"Design a Streamlit app with dark sidebar, job cards with match score badges, like/pass/skip feedback buttons, explain expander, and generate resume action."*  
  **Purpose**: Generated `app.py` UI layout and custom CSS.

---

## Notes on AI Tool Usage

All AI-generated code was reviewed, tested, and modified before use. Key modifications:
- Fixed FAISS index loading/saving to handle cold-start (no pre-built index)
- Adjusted ranking weights (40/30/15/15) based on experimentation with test personas
- Added error handling for Gemini API failures with graceful fallbacks
- Fixed SQLite thread safety (`check_same_thread=False`)
