"""
engine/resume_gen.py
Gemini-powered tailored resume generator.
Signature deliverable: 'Generate Resume' action per job.
"""
import re
import io
import streamlit as st
import google.generativeai as genai


RESUME_PROMPT = """You are an expert resume writer and career coach.

Given the candidate's profile and the specific job description, write a complete,
tailored resume in Markdown format optimised for this role.

Rules:
- Lead with the candidate's strongest relevant experience for THIS specific role.
- Reframe existing experience using the language of the job description.
- Include a Professional Summary (3 sentences) mentioning the company name.
- List top 6-8 relevant skills first.
- Use strong action verbs and quantify achievements where possible.
- Do NOT invent qualifications the candidate doesn't have.
- Format with proper Markdown headers (# for name, ## for sections).

Candidate Profile:
{profile_summary}

Target Job:
Title: {job_title}
Company: {company}
Location: {location}
Description:
{job_description}

Write the complete tailored resume now:"""


def _get_gemini():
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-flash-latest")


def generate_resume(profile: dict, job: dict) -> str:
    """
    Generate a Markdown-formatted tailored resume.
    Returns the Markdown string.
    """
    profile_summary = _format_profile_summary(profile)
    description     = str(job.get("description", ""))[:3000]

    prompt = RESUME_PROMPT.format(
        profile_summary  = profile_summary,
        job_title        = job.get("title", ""),
        company          = job.get("company", ""),
        location         = job.get("location", ""),
        job_description  = description,
    )

    try:
        model    = _get_gemini()
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"# Resume Generation Error\n\nError: {e}\n\nPlease try again."


def _format_profile_summary(profile: dict) -> str:
    lines = [
        f"Name: {profile.get('name', 'Candidate')}",
        f"Current Title: {profile.get('current_title', 'N/A')}",
        f"Experience: {profile.get('experience_years', 0)} years",
        f"Education: {profile.get('education', 'N/A')}",
        f"Skills: {', '.join(profile.get('skills', [])[:15])}",
        f"Target Roles: {', '.join(profile.get('target_roles', [])[:5])}",
        f"Location: {profile.get('location', 'N/A')}",
        "",
        "Resume Text:",
        profile.get("raw_text", "")[:2000],
    ]
    return "\n".join(lines)


def resume_to_bytes(markdown_text: str) -> bytes:
    """Return the resume as UTF-8 bytes for Streamlit download button."""
    return markdown_text.encode("utf-8")


def generate_cover_letter(profile: dict, job: dict) -> str:
    """Bonus: cover letter generator."""
    prompt = f"""Write a concise, compelling cover letter (3 paragraphs) for:

Candidate: {profile.get('name', 'Candidate')}
Skills: {', '.join(profile.get('skills', [])[:10])}
Experience: {profile.get('experience_years', 0)} years

Applying to:
Title: {job.get('title', '')} at {job.get('company', '')}
Location: {job.get('location', '')}
Description: {str(job.get('description', ''))[:1500]}

Write a professional, specific cover letter now:"""

    try:
        model    = _get_gemini()
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Cover letter generation failed: {e}"
