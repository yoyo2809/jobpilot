"""
app.py — JobPilot Main Streamlit Application
BAX-423 Final Project · Option B · Smart Job Matcher & Resume Builder
"""
import sys
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Make engine importable ────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from engine import database as db
db.initialize_db()
from engine.embeddings import EmbeddingEngine, get_engine
from engine.ranking import UserPreferences, rank_jobs, explain_ranking
from engine import feedback as fb_engine
from engine import profile as profile_mod
from engine import resume_gen
from engine import ingestion

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="JobPilot — AI Job Matcher",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    color: white;
}
.main-header h1 { font-size: 2.6rem; font-weight: 700; margin: 0; }
.main-header p  { font-size: 1.1rem; opacity: 0.9; margin: 0.4rem 0 0 0; }

.job-card {
    background: white;
    border: 1px solid #e8ecf0;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,.06);
    transition: box-shadow .2s;
}
.job-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,.12); }

.match-badge {
    display: inline-block;
    padding: .3rem .8rem;
    border-radius: 20px;
    font-size: .85rem;
    font-weight: 600;
    color: white;
}
.badge-high  { background: linear-gradient(90deg,#43e97b,#38f9d7); color:#1a4a2e; }
.badge-mid   { background: linear-gradient(90deg,#f6d365,#fda085); color:#5c3510; }
.badge-low   { background: linear-gradient(90deg,#a8a8a8,#d3d3d3); color:#333; }

.skill-chip {
    display: inline-block;
    background: #f0f4ff;
    color: #4f46e5;
    border-radius: 6px;
    padding: .15rem .55rem;
    font-size: .78rem;
    margin: .1rem;
    font-weight: 500;
}
.skill-chip-match { background: #d1fae5; color: #065f46; }

.stat-box {
    background: linear-gradient(135deg,#f8f9ff,#eef1ff);
    border-left: 4px solid #667eea;
    border-radius: 8px;
    padding: .9rem 1.2rem;
    margin-bottom: .6rem;
}

section[data-testid="stSidebar"] { background: #0f0e17; color: #fffffe; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p { color: #fffffe !important; }
section[data-testid="stSidebar"] .stMarkdown { color: #a7a9be; }
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "profile" not in st.session_state:
    st.session_state.profile = None
if "ranked_jobs" not in st.session_state:
    st.session_state.ranked_jobs = None
if "prefs" not in st.session_state:
    st.session_state.prefs = None


# ── Load engine (cached across reruns) ───────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI engine …")
def load_engine() -> EmbeddingEngine:
    engine = get_engine()
    if not engine.load_index():
        # Auto-build if index doesn't exist
        st.info("Building search index for the first time … (~2 min)")
        jobs_df = db.get_all_jobs_for_indexing()
        if not jobs_df.empty:
            engine.build_index(jobs_df)
    return engine


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🚀 JobPilot</h1>
  <p>AI-powered job matcher, ranker & resume builder · BAX-423 Final Project</p>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📋 Your Profile")

    uploaded = st.file_uploader(
        "Upload Resume (PDF or DOCX)",
        type=["pdf", "docx"],
        help="We extract your skills, experience, and target roles automatically.",
    )

    if uploaded:
        with st.spinner("Parsing resume with Gemini AI …"):
            raw_text = profile_mod.extract_text(uploaded)
            profile  = profile_mod.parse_profile(raw_text)
            st.session_state.profile = profile
            db.save_user_profile(st.session_state.session_id, profile)
            st.session_state.ranked_jobs = None   # trigger re-rank

    if st.session_state.profile:
        p = st.session_state.profile
        st.success(f"✅ Profile loaded: **{p.get('name','User')}**")
        
        if p.get("target_roles"):
            st.markdown("**🎯 Target Roles:**")
            st.markdown(" ".join(f"`{r}`" for r in p["target_roles"][:5]))
            
        if p.get("skills"):
            st.markdown("**🛠 Skills detected:**")
            st.markdown(" ".join(f"`{s}`" for s in p["skills"][:12]))

    st.divider()
    st.markdown("## ⚙️ Preferences")

    manual_background = st.text_area("✍️ Or paste Background/Resume here", placeholder="e.g. Data Analyst for 3 years, wanting to pivot to ML Engineering...")
    manual_roles_str = st.text_input("🎯 Target Roles (comma-separated)", placeholder="e.g. Data Analyst, Software Engineer")
    manual_skills_str = st.text_input("🛠 Skills (comma-separated)", placeholder="e.g. Python, SQL, Machine Learning")
    location = st.text_input("📍 Preferred Location", placeholder="San Francisco, CA")
    min_salary = st.slider("💰 Min Salary ($/yr)", 0, 300_000, 80_000, 10_000,
                           format="$%d")
    remote_ok = st.toggle("Remote OK", value=True)
    visa_req  = st.toggle("Needs visa sponsorship", value=False)

    dealbreaker_options = [
        "Junior", "Entry Level", "Contract", "Internship",
        "Senior", "Staff", "Principal", "Director", "VP",
        "5+ years", "7+ years", "10+ years",
        "Defense", "Military"
    ]
    dealbreakers = st.multiselect(
        "🚫 Dealbreakers (exclude these)",
        dealbreaker_options,
    )
    custom_dbs = st.text_input("🚫 Custom Dealbreakers (comma-separated)", placeholder="e.g. defense, healthcare")

    # Combine extracted from PDF (if any) + manual inputs
    extracted_roles = st.session_state.profile.get("target_roles", []) if st.session_state.profile else []
    if manual_roles_str:
        extracted_roles.extend([r.strip() for r in manual_roles_str.split(",") if r.strip()])
        
    extracted_skills = st.session_state.profile.get("skills", []) if st.session_state.profile else []
    if manual_skills_str:
        extracted_skills.extend([s.strip() for s in manual_skills_str.split(",") if s.strip()])

    if custom_dbs:
        dealbreakers.extend([d.strip() for d in custom_dbs.split(",") if d.strip()])

    prefs = UserPreferences(
        background    = manual_background,
        location      = location,
        min_salary    = float(min_salary),
        skills        = list(set(extracted_skills)),
        target_roles  = list(set(extracted_roles)),
        dealbreakers  = dealbreakers,
        remote_ok     = remote_ok,
        visa_required = visa_req,
    )
    st.session_state.prefs = prefs

    if st.button("🔍 Find Matches", type="primary", use_container_width=True):
        st.session_state.ranked_jobs = None
        st.rerun()

    st.divider()
    st.markdown("## 🔄 Live Job Streaming")

    stream_query = st.text_input("Search query", value="data scientist machine learning")
    if st.button("⬇️ Fetch New Jobs", use_container_width=True):
        with st.spinner("Fetching 750 jobs from Adzuna (this takes ~15 seconds) …"):
            stats = ingestion.manual_fetch(query=stream_query, max_pages=15)
        st.success(f"✅ Fetched {stats['fetched']} | New: {stats['new']} | Dupes: {stats['dupes']}")
        st.session_state.ranked_jobs = None   # refresh results

    stream_stats = db.get_streaming_stats()
    st.markdown(f"""
<div class="stat-box">
🗄 <b>Total jobs:</b> {db.get_job_count():,}<br>
📡 <b>Streamed new:</b> {stream_stats['total_new']:,}<br>
🕐 <b>Last fetch:</b> {str(stream_stats['last_fetch'])[:16]}
</div>
""", unsafe_allow_html=True)

    st.divider()
    fb_stats = fb_engine.get_session_summary(st.session_state.session_id)
    st.markdown("## 🧠 Adaptive Learning")
    col1, col2, col3 = st.columns(3)
    col1.metric("✅ Liked",    fb_stats["accepted"])
    col2.metric("❌ Rejected", fb_stats["rejected"])
    col3.metric("⏭ Skipped",  fb_stats["skipped"])
    if fb_stats["total"] > 0:
        st.caption(f"Ranking adapts after each reaction. "
                   f"Rejected jobs/companies are down-weighted.")


# ════════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ════════════════════════════════════════════════════════════════════════════════
job_count = db.get_job_count()

if job_count == 0:
    st.warning("""
    ### 📂 No job data loaded yet

    Run these commands in your terminal to download and index the data:
    ```bash
    python scripts/download_data.py
    python scripts/build_index.py
    ```
    Or click **"Fetch New Jobs"** in the sidebar to pull data from Adzuna.
    """)
    st.stop()

engine = load_engine()


def _show_market_overview():
    """Quick stats when no profile is loaded."""
    df = db.get_analytics_data()
    if df.empty:
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Jobs", f"{len(df):,}")
    lvl = df["experience_level"].dropna()
    c2.metric("Experience Levels", lvl.nunique())
    locs = df["location"].dropna()
    c3.metric("Locations", locs.nunique())


if not st.session_state.profile:
    st.info("👆 Upload your resume in the sidebar to get personalised job matches.")
    _demo_tab, _about_tab = st.tabs(["📊 Market Overview", "ℹ️ About"])
    with _demo_tab:
        _show_market_overview()
    st.stop()


# ── Tab layout ────────────────────────────────────────────────────────────────
tab_match, tab_download, tab_bench = st.tabs([
    "🎯 Job Matches",
    "⬇️ Download Top Jobs",
    "📈 Benchmark",
])


# ── Helper: run ranking (cached in session) ───────────────────────────────────
def get_ranked_jobs() -> pd.DataFrame:
    if st.session_state.ranked_jobs is None:
        profile  = st.session_state.profile
        prefs    = st.session_state.prefs
        profile_text = profile.get("raw_text", "")[:800] if profile else ""
        target_roles = " ".join(profile.get("target_roles", [])) if profile else " ".join(prefs.target_roles)
        skills_text  = " ".join(profile.get("skills", [])[:10]) if profile else " ".join(prefs.skills[:10])
        
        query    = " ".join([
            profile_text,
            prefs.background[:800] if prefs.background else "",
            target_roles,
            skills_text,
        ])
        with st.spinner("🔍 Retrieving & ranking jobs …"):
            ranked = rank_jobs(
                profile_text = query,
                prefs        = prefs,
                embedding_engine = engine,
                session_id   = st.session_state.session_id,
                top_n        = 20,
            )
        st.session_state.ranked_jobs = ranked
    return st.session_state.ranked_jobs


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — JOB MATCHES
# ════════════════════════════════════════════════════════════════════════════════
with tab_match:
    ranked = get_ranked_jobs()

    if ranked is None or ranked.empty:
        st.warning("No jobs matched your profile. Try relaxing your dealbreakers.")
    else:
        n_shown = len(ranked)
        st.markdown(f"### Top {n_shown} Matches for **{st.session_state.profile.get('name','You')}**")
        st.caption(
            "Pipeline: FAISS Embedding Retrieval (Lecture 5) → "
            "Hard Filter → Weighted Re-ranking (Lecture 7) → Adaptive Feedback"
        )
        
        # --- Explicit Evaluation Metrics for Rubric #4 ---
        with st.container():
            st.markdown("##### 📊 Real-Time Ranking Evaluation Metrics (Rubric #4)")
            m1, m2, m3 = st.columns(3)
            
            # Metric 1: Avg Top-10 Relevancy
            top_10 = ranked.head(10)
            avg_rel = top_10["match_pct"].mean() if not top_10.empty else 0
            m1.metric("Avg Top-10 Relevance", f"{avg_rel:.1f}%")
            
            # Metric 2: Session Precision (from likes/rejects)
            fb = fb_engine.get_session_summary(st.session_state.session_id)
            total_fb = fb["accepted"] + fb["rejected"]
            prec = (fb["accepted"] / total_fb * 100) if total_fb > 0 else 0
            m2.metric("User Precision (Hit Rate)", f"{prec:.1f}%")
            
            # Metric 3: Skill Coverage in Top-10
            u_skills = st.session_state.prefs.skills if st.session_state.prefs else []
            if u_skills:
                found = set()
                for _, r in top_10.iterrows():
                    text = (str(r.get("skills","")) + " " + str(r.get("description",""))).lower()
                    for s in u_skills:
                        if s.lower() in text:
                            found.add(s.lower())
                cov = (len(found) / len(u_skills) * 100) if u_skills else 0
                m3.metric("Top-10 Skill Coverage", f"{cov:.1f}%")
            else:
                m3.metric("Top-10 Skill Coverage", "N/A")
        st.divider()

        for idx, (_, row) in enumerate(ranked.iterrows()):
            job_id  = str(row["id"])
            match   = int(row.get("match_pct", 0))
            salary_str = ""
            sal_min = row.get("salary_min")
            sal_max = row.get("salary_max")
            if pd.notna(sal_min) and sal_min:
                salary_str = f"${float(sal_min):,.0f}"
                if pd.notna(sal_max) and sal_max:
                    salary_str += f" – ${float(sal_max):,.0f}"

            badge_cls = "badge-high" if match >= 70 else ("badge-mid" if match >= 45 else "badge-low")

            # ── Job card ──────────────────────────────────────────────────────
            with st.container():
                st.markdown(f"""
<div class="job-card">
<div style="display:flex;justify-content:space-between;align-items:flex-start;">
<div>
<h4 style="margin:0;color:#1a1a2e;font-size:1.1rem;">{row.get('title','')}</h4>
<p style="margin:.2rem 0;color:#555;font-size:.9rem;">
🏢 {row.get('company','')} &nbsp;|&nbsp; 📍 {row.get('location','')}
{"&nbsp;|&nbsp; 💰 " + salary_str if salary_str else ""}
{"&nbsp;|&nbsp; 🏷 " + str(row.get('experience_level','')) if pd.notna(row.get('experience_level')) and row.get('experience_level') else ""}
</p>
</div>
<span class="match-badge {badge_cls}">{match}% match</span>
</div>
</div>
""", unsafe_allow_html=True)

                # Controls row
                c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1.5, 1.5])
                with c1:
                    if st.button("👍 Like",  key=f"like_{job_id}_{idx}"):
                        fb_engine.record(st.session_state.session_id, job_id, "accept")
                        st.session_state.ranked_jobs = None
                        st.rerun()
                with c2:
                    if st.button("👎 Pass",  key=f"pass_{job_id}_{idx}"):
                        fb_engine.record(st.session_state.session_id, job_id, "reject")
                        st.session_state.ranked_jobs = None
                        st.rerun()
                with c3:
                    if st.button("⏭ Skip",  key=f"skip_{job_id}_{idx}"):
                        fb_engine.record(st.session_state.session_id, job_id, "skip")
                        st.session_state.ranked_jobs = None
                        st.rerun()
                with c4:
                    gen_key = f"gen_{job_id}_{idx}"
                    if st.button("📝 Generate Resume", key=gen_key):
                        with st.spinner("Generating tailored resume with Gemini …"):
                            resume_md = resume_gen.generate_resume(
                                st.session_state.profile,
                                row.to_dict(),
                            )
                        st.session_state[f"resume_{job_id}"] = resume_md
                with c5:
                    apply_url = row.get("apply_url", "")
                    if apply_url and str(apply_url) not in ("nan", "None", ""):
                        st.link_button("🔗 Apply", apply_url)

                # Explain expander
                with st.expander("🔍 Why ranked here?"):
                    explanation = explain_ranking(row, st.session_state.prefs)
                    st.markdown(explanation)

                # Generated resume expander
                if f"resume_{job_id}" in st.session_state:
                    with st.expander("📄 Generated Resume", expanded=True):
                        resume_md = st.session_state[f"resume_{job_id}"]
                        st.markdown(resume_md)
                        st.download_button(
                            label       = "⬇️ Download Resume (.md)",
                            data        = resume_gen.resume_to_bytes(resume_md),
                            file_name   = f"resume_{str(row.get('company','job')).replace(' ','_')[:20]}.md",
                            mime        = "text/markdown",
                            key         = f"dl_resume_{job_id}_{idx}",
                        )

                st.markdown("---")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — DOWNLOAD TOP JOBS
# ════════════════════════════════════════════════════════════════════════════════
with tab_download:
    st.markdown("### ⬇️ Download Your Top Job Matches")
    ranked = get_ranked_jobs()

    if ranked is not None and not ranked.empty:
        export_cols = ["title","company","location","salary_min","salary_max",
                       "experience_level","work_type","match_pct","apply_url","description"]
        export_df = ranked[[c for c in export_cols if c in ranked.columns]].copy()
        export_df = export_df.rename(columns={"match_pct": "match_%", "apply_url": "link"})

        st.dataframe(export_df.drop(columns=["description"], errors="ignore"), use_container_width=True)

        # Excel download
        import io
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            export_df.to_excel(writer, index=False, sheet_name="Top Jobs")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button(
                "📥 Download as CSV",
                data      = export_df.to_csv(index=False).encode('utf-8'),
                file_name = "jobpilot_top_matches.csv",
                mime      = "text/csv",
                use_container_width=True
            )
        with c2:
            st.download_button(
                "📥 Download as Excel",
                data      = buf.getvalue(),
                file_name = "jobpilot_top_matches.xlsx",
                mime      = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

        with c3:
            st.download_button(
                "📥 Download as JSON",
                data      = export_df.to_json(orient="records", indent=2),
                file_name = "jobpilot_top_matches.json",
                mime      = "application/json",
                use_container_width=True
            )
    else:
        st.info("Match some jobs first (👍 or ⏭ on the Job Matches tab).")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — BENCHMARK
# ════════════════════════════════════════════════════════════════════════════════
with tab_bench:
    st.markdown("### 📈 BAX-423 Technique Benchmark")
    st.markdown("""
    Compares three retrieval approaches across the 4 test personas.
    - **Baseline**: BM25 keyword search (no ML)
    - **Embedding (L5)**: Sentence-transformer FAISS retrieval (Lecture 5)
    - **Multi-Stage (L7)**: Full re-ranking pipeline (Lecture 7)
    """)

    if st.button("▶️ Run Benchmark (takes ~30s)", type="primary"):
        from engine.benchmark import run_benchmark
        with st.spinner("Running benchmark across all 4 personas …"):
            bench_df = run_benchmark(engine, k=10)
        st.dataframe(bench_df, use_container_width=True)
        st.caption("Metrics: P@10 = Precision at 10  |  NDCG@10 = Normalised Discounted Cumulative Gain at 10")
        st.session_state["bench_results"] = bench_df

    if "bench_results" in st.session_state:
        import plotly.graph_objects as go
        bdf = st.session_state["bench_results"]
        personas = bdf["Persona"].tolist()

        fig = go.Figure()
        for label, col in [
            ("BM25 (Baseline)", "BM25 NDCG@10"),
            ("Embedding L5",    "Embedding NDCG@10"),
            ("Multi-Stage L7",  "Multi-Stage NDCG@10"),
        ]:
            fig.add_trace(go.Bar(
                name   = label,
                x      = personas,
                y      = bdf[col].astype(float).tolist(),
            ))
        fig.update_layout(
            title       = "NDCG@10 by Persona & Approach",
            barmode     = "group",
            yaxis_title = "NDCG@10",
            xaxis_tickangle = -20,
            height      = 420,
        )
        st.plotly_chart(fig, use_container_width=True)
