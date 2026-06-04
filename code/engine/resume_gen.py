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
- Omit irrelevant or basic skills (e.g., Excel, reporting) if the target role is advanced (e.g., ML/Data Science).
- Must satisfy the Pass Criteria below. If the criteria mention ML, emphasize Python, ML, modeling, scikit-learn, data pipelines, and learning trajectory.
- Do not lead with Excel, dashboards, reporting, Tableau, or generic business analysis when the pass criteria require ML.
- Format with proper Markdown headers (# for name, ## for sections).

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
        pass_criteria    = _format_pass_criteria(profile),
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


def _format_pass_criteria(profile: dict) -> str:
    target_roles = " ".join(profile.get("target_roles", [])).lower()
    dealbreakers = [str(d).lower() for d in profile.get("dealbreakers", [])]
    criteria = []

    ml_mode = any(term in target_roles for term in [
        "ml engineer", "machine learning", "ai engineer",
        "applied scientist", "research scientist", "mlops",
        "deep learning", "nlp engineer", "computer vision",
    ])
    if ml_mode:
        criteria.append("- Resume highlights Python/ML/modeling/scikit-learn experience first.")
        criteria.append("- Resume does not lead with Excel, dashboarding, reporting, or generic analyst work.")
        criteria.append("- Summary positions the candidate as pivoting toward ML engineering without inventing production ML experience.")

    infra_mode = any(term in target_roles for term in [
        "mlops", "platform engineer", "ml platform", "machine learning platform",
        "infrastructure", "senior engineer",
    ])
    profile_text = (
        " ".join(profile.get("skills", [])) + " " +
        " ".join(profile.get("target_roles", [])) + " " +
        str(profile.get("raw_text", ""))
    ).lower()
    infra_skills = [skill for skill in ["Kafka", "Spark", "Kubernetes", "AWS", "TensorFlow"] if skill.lower() in profile_text]
    if infra_mode and infra_skills:
        criteria.append(f"- Position {', '.join(infra_skills)} as ML infrastructure / ML platform engineering skills.")
        criteria.append("- Emphasize production systems, data pipelines, model-serving platforms, reliability, and cross-functional engineering impact.")

    if any(d in ("senior", "staff", "principal", "director", "vp") for d in dealbreakers):
        criteria.append("- Tone should fit entry-to-mid level roles, not senior/staff leadership roles.")
    if any(d in ("defense", "defence", "military") for d in dealbreakers):
        criteria.append("- Do not frame the candidate toward defense or military work.")
    if any(d in ("contract", "1099", "temporary", "temp", "unpaid") for d in dealbreakers):
        criteria.append("- Do not frame the resume toward contract, temporary, 1099, or unpaid roles.")

    if not criteria:
        criteria.append("- Align the resume with the stated target roles, skills, location, salary, and dealbreakers.")
    return "\n".join(criteria)


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
