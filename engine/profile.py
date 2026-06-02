"""
engine/profile.py
Parse uploaded resume (PDF/DOCX) and extract structured profile
using Gemini 1.5 Flash.
"""
import io
import json
import re
from typing import Optional

import streamlit as st

try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

try:
    from docx import Document as DocxDocument
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

import google.generativeai as genai


def _get_gemini():
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set in .streamlit/secrets.toml")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")


# ── Text extraction ───────────────────────────────────────────────────────────

def extract_text_from_pdf(file_bytes: bytes) -> str:
    if not HAS_PYPDF2:
        return ""
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    pages  = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def extract_text_from_docx(file_bytes: bytes) -> str:
    if not HAS_DOCX:
        return ""
    doc   = DocxDocument(io.BytesIO(file_bytes))
    paras = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paras)


def extract_text(uploaded_file) -> str:
    """Auto-detect PDF/DOCX and extract text."""
    name  = uploaded_file.name.lower()
    data  = uploaded_file.read()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(data)
    elif name.endswith(".docx"):
        return extract_text_from_docx(data)
    else:
        return data.decode("utf-8", errors="ignore")


# ── LLM profile extraction ────────────────────────────────────────────────────

PROFILE_PROMPT = """You are a resume parser. Extract structured information from the resume below.

Return ONLY valid JSON with exactly these fields:
{{
  "name": "Full Name",
  "skills": ["Python", "SQL", "Tableau"],
  "experience_years": 3,
  "education": "MS Business Analytics, UC Davis 2024",
  "target_roles": ["Data Analyst", "Business Intelligence Analyst"],
  "current_title": "Data Analyst",
  "location": "San Francisco, CA",
  "summary": "One sentence summary of the candidate."
}}

Resume:
{resume_text}"""


def parse_profile(resume_text: str) -> dict:
    """
    Call Gemini to extract a structured profile from raw resume text.
    Falls back to a minimal dict on error.
    """
    if not resume_text.strip():
        return _empty_profile()

    try:
        model    = _get_gemini()
        prompt   = PROFILE_PROMPT.format(resume_text=resume_text[:4000])
        response = model.generate_content(prompt)
        raw      = response.text.strip()

        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        profile          = json.loads(raw)
        profile["raw_text"] = resume_text
        return profile

    except Exception as e:
        st.warning(f"Profile parse warning: {e}. Using basic extraction.")
        return _basic_extract(resume_text)


def _empty_profile() -> dict:
    return {
        "name": "User", "skills": [], "experience_years": 0,
        "education": "", "target_roles": [], "current_title": "",
        "location": "", "summary": "No resume uploaded.",
        "raw_text": "",
    }


def _basic_extract(text: str) -> dict:
    """Fallback: simple keyword extraction without LLM."""
    common_skills = [
        "Python", "SQL", "R", "Java", "Scala", "Spark", "Kafka",
        "Tableau", "PowerBI", "Excel", "pandas", "NumPy", "scikit-learn",
        "TensorFlow", "PyTorch", "AWS", "GCP", "Azure", "Docker",
        "Kubernetes", "Airflow", "dbt", "Snowflake", "PostgreSQL",
        "MySQL", "MongoDB", "NLP", "Machine Learning", "Deep Learning",
        "PySpark", "MLflow", "FastAPI", "Flask", "React",
    ]
    found_skills = [s for s in common_skills if s.lower() in text.lower()]
    return {
        "name": "User",
        "skills": found_skills,
        "experience_years": 0,
        "education": "",
        "target_roles": [],
        "current_title": "",
        "location": "",
        "summary": text[:200],
        "raw_text": text,
    }
