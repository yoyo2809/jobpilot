# prompts.md — Key AI Prompts Used in JobPilot

> BAX-423 Final Project · Option B · JobPilot  
> AI tools used: Google Gemini via API, ChatGPT/Codex for implementation review and debugging.

## 1. Resume / Profile Extraction

Prompt used in `code/engine/profile.py`:

```text
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

Purpose: convert unstructured PDF/DOCX text into a structured profile used by ranking and resume generation.

Key modifications: resume text is truncated to 4000 characters, markdown JSON fences are stripped, and a basic regex fallback extracts common skills when Gemini fails.

## 2. Tailored Resume Generation

Prompt used in `code/engine/resume_gen.py`:

```text
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
- Omit irrelevant or basic skills (e.g., Excel, reporting) if the target role is advanced.
- Must satisfy the Pass Criteria below.
- If the Pass Criteria mention ML infrastructure, the Top Skills section MUST contain
  a bullet titled exactly "ML Infrastructure & Platform Engineering".
- If the Pass Criteria mention publications, the first major section after Professional
  Summary MUST be "Selected Publications & Research".

Candidate Profile:
{profile_summary}

Pass Criteria:
{pass_criteria}

Target Job:
Title: {job_title}
Company: {company}
Location: {location}
Description:
{job_description}
```

Purpose: generate a role-specific Markdown resume using the selected job description plus the current persona constraints.

Key modifications:
- Aisha: resume must highlight Python/ML/modeling and not lead with Excel/reporting.
- Priya: Kafka, Spark, Kubernetes, AWS, and TensorFlow are positioned as ML infrastructure / ML platform engineering skills.
- Kenji: publications/research output must appear immediately after the professional summary.
- Dealbreakers are passed into the prompt so the resume does not frame the candidate toward contract, defense, or seniority-mismatched roles.

## 3. Ranking And Recommendation Design Prompts

Implementation prompts used during development:

```text
Build a multi-stage ranking pipeline using embedding retrieval and re-ranking.
Stage 1: FAISS recall over job postings.
Stage 2: deterministic hard filters for dealbreakers, salary, seniority, visa constraints,
contract/temp terms, and persona-specific pass criteria.
Stage 3: weighted scoring combining semantic similarity, skill overlap, role match,
location preference, and adaptive feedback.
```

Current scoring weights:

```text
Score = 0.25*Embedding + 0.25*Skill + 0.25*Role + 0.10*Location + 0.15*Feedback
```

Purpose: implement Lecture 5 dense retrieval and Lecture 7 multi-stage recommendation.

Important modifications after testing:
- Added strict ML-only title filtering for the Aisha persona.
- Separated Data Analyst/BI/New Grad personas from ML-only filtering.
- Added a fallback over the offline snapshot when strict hard filters starve FAISS recall.
- Added transparent company-size proxy logic because the dataset has no real headcount field.
- Added known sponsor / research-lab soft signals for the visa-constrained persona.

## 4. Streaming Ingestion Prompt

```text
Create a streaming ingestion module that simulates a Kafka producer/consumer
using Python's threading.Queue. The producer fetches from Adzuna. The consumer
deduplicates against SQLite and inserts only new jobs.
```

Purpose: satisfy the live/current job ingestion requirement while keeping the demo reliable. The UI uses a one-shot Adzuna fetch button and the module also contains a Kafka-style queue pipeline.

## 5. Benchmark Prompt

```text
Write a benchmark comparing BM25 keyword retrieval, sentence-transformer FAISS retrieval,
and the full multi-stage recommender. Use Precision@10 and NDCG@10 across the four
required personas, and include deterministic pass/fail checks for each persona's rubric.
```

Purpose: show the impact of course techniques and produce a pass/fail table for the technical brief.

## 6. AI-Assisted Debugging Prompts

Prompts used during final review included:

```text
Audit the project against the final exam requirements. Identify mismatches between
the pass criteria, Streamlit UI, technical brief, benchmark checks, and implementation.
```

```text
For each test persona, inspect whether Top-10 recommendations satisfy the stated
pass criteria and modify ranking/resume generation without breaking the other personas.
```

Purpose: align the final code, benchmark, explanations, prompt constraints, and documentation before submission.

## 7. Persona-Specific Refinement Prompts

Prompts used to tighten the four required persona demos:

```text
Review Persona 1 Aisha against this pass criteria:
Top-10 has zero Senior/Staff roles, zero defense companies, and all jobs are ML-related.
Identify why generic Software Engineer, DataAnnotation, senior, staff, principal, director,
lead, manager, L5/L6, defense, military, or clearance-related jobs might still appear.
Modify the hard filters and role relevance logic so Aisha's Top-10 remains ML-focused
without breaking non-ML personas.
```

Purpose: force Aisha's career-pivot recommendations to satisfy the ML-only and no-senior constraints.

```text
Review Persona 2 Marcus as a new MSBA graduate.
The previous test was still returning ML-focused results after changing the inputs.
Find any caching or preference-signature issue that could preserve stale rankings.
Ensure Marcus sees Data Analyst, BI Analyst, Junior Data Scientist, or Analytics Engineer
roles and excludes 3+ years, 5+ years, senior, contract-only, and unpaid jobs.
The generated resume should lead with MSBA education and projects, not full-time work history.
```

Purpose: separate the broad new-grad search from Aisha's ML-only filter and enforce education-first resume generation.

```text
Review Persona 3 Priya:
Top-10 must have zero Junior roles and zero tiny startups.
Because the dataset has no structured employee-count field, propose transparent proxy
rules for company-size constraints. Also ensure the generated resume positions Kafka,
Spark, Kubernetes, AWS, and TensorFlow as ML infrastructure / ML platform skills rather
than generic backend skills.
```

Purpose: document the company-size proxy design and improve ML infrastructure framing.

```text
Review Persona 4 Kenji:
Top-10 must have zero contract/temp/1099 roles, should favor large companies or research labs,
and resume generation should lead with publications. The dataset lacks guaranteed H-1B
sponsorship fields, so design honest proxy logic using explicit no-sponsorship language,
known large employers, universities, and research-lab terms. Also add a demo mechanism
to show small-company rejection feedback when no small company appears in the visible Top-10.
```

Purpose: make the visa-constrained demo transparent about sponsorship proxies while still satisfying the rubric behavior.

## 8. Final Submission Audit And Packaging Prompts

Prompts used during the final pre-submission review:

```text
Read the final exam PDF, the JobPilot project brief, and the current repository.
Check whether the project satisfies all required deliverables:
working app, public hosted URL, GitHub repository, code folder with README and requirements,
data folder with offline snapshot, brief.pdf under 4 pages, prompts.md, and all six
core capabilities. List only issues that should be fixed before submission.
```

Purpose: verify the project against the official requirements rather than only against implementation assumptions.

```text
Inspect the final ZIP file. It must be named LastName_FirstName_BAX423_Final.zip and
its root must directly contain code/, data/, brief.pdf, prompts.md, and requirements.txt.
Flag nested top-level folders, __MACOSX files, .DS_Store files, pycache files, temporary
test scripts, and any real secrets.toml file.
```

Purpose: ensure the Canvas upload has the required structure and does not leak API keys.

```text
Check the Streamlit deployment and documentation. Confirm the live URL is present in
brief.md / brief.pdf and README, confirm the app is public, and explain what works
without local secrets versus what requires Streamlit Cloud secrets.
```

Purpose: distinguish safe local offline demo behavior from hosted Gemini/Adzuna API functionality.

```text
Explain how constraints that are not explicit database columns are implemented.
For company size, H-1B sponsorship, defense/military, contract/temp, and seniority,
identify whether the project uses structured fields or text-based proxy rules, and
rewrite the brief so it is transparent about those limitations.
```

Purpose: avoid overstating the dataset and make the limitations clear for grading and live demo questions.
