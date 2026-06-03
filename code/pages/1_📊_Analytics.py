"""
pages/1_📊_Analytics.py
Job Market Analytics Dashboard — Core Capability: Batch Analytics
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from engine import database as db

st.set_page_config(page_title="Job Market Analytics · JobPilot", page_icon="📊", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Job Market Analytics")
st.caption("Aggregate insights from the full job dataset · BAX-423 JobPilot")

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_data():
    return db.get_analytics_data()

df = load_data()
if df.empty:
    st.warning("No job data loaded. Run `python scripts/download_data.py` first.")
    st.stop()

# ── KPI row ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("📋 Total Jobs",      f"{len(df):,}")
c2.metric("🏢 Companies",       f"{df['company'].nunique():,}")
c3.metric("📍 Locations",       f"{df['location'].nunique():,}")

sal = pd.to_numeric(df["salary_max"], errors="coerce").dropna()
c4.metric("💰 Avg Max Salary",  f"${sal.mean():,.0f}" if not sal.empty else "N/A")

exp_counts = df["experience_level"].dropna()
top_exp = exp_counts.value_counts().index[0] if not exp_counts.empty else "N/A"
c5.metric("🎓 Top Level",       top_exp)

st.divider()

# ── Row 1: Top Skills + Salary Distribution ───────────────────────────────────
row1_left, row1_right = st.columns(2)

with row1_left:
    st.subheader("🛠 Top 20 Skills in Demand")
    # Extract skills from description text (keyword frequency)
    skill_keywords = [
        "Python","SQL","R","Java","Scala","Spark","Kafka","Tableau","PowerBI",
        "Excel","pandas","NumPy","scikit-learn","TensorFlow","PyTorch","AWS",
        "GCP","Azure","Docker","Kubernetes","Airflow","dbt","Snowflake",
        "PostgreSQL","MySQL","MongoDB","NLP","Machine Learning","Deep Learning",
        "PySpark","MLflow","FastAPI","Looker","Databricks","LLM","Generative AI",
        "Statistics","A/B Testing","Computer Vision","Reinforcement Learning",
    ]
    desc_corpus = " ".join(df["description"].fillna("").str.lower().tolist())
    skill_counts = {s: desc_corpus.count(s.lower()) for s in skill_keywords}
    skill_df = (
        pd.DataFrame(list(skill_counts.items()), columns=["Skill","Count"])
        .sort_values("Count", ascending=True)
        .tail(20)
    )
    fig_skills = px.bar(
        skill_df, x="Count", y="Skill", orientation="h",
        color="Count", color_continuous_scale="Viridis",
        labels={"Count": "# Mentions"},
    )
    fig_skills.update_layout(height=480, showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig_skills, use_container_width=True)

with row1_right:
    st.subheader("💰 Salary Distribution by Experience Level")
    sal_df = df[["experience_level","salary_max"]].copy()
    sal_df["salary_max"] = pd.to_numeric(sal_df["salary_max"], errors="coerce")
    sal_df = sal_df.dropna()
    if not sal_df.empty and sal_df["experience_level"].notna().any():
        fig_sal = px.box(
            sal_df, x="experience_level", y="salary_max",
            color="experience_level",
            labels={"salary_max": "Max Salary ($)", "experience_level": "Level"},
        )
        fig_sal.update_layout(height=480, showlegend=False,
                              xaxis_tickangle=-20)
        st.plotly_chart(fig_sal, use_container_width=True)
    else:
        st.info("No salary + experience data available in this snapshot.")

st.divider()

# ── Row 2: Locations + Work Types ─────────────────────────────────────────────
row2_left, row2_right = st.columns(2)

with row2_left:
    st.subheader("📍 Top 20 Job Locations")
    loc_df = (
        df["location"].dropna()
        .str.split(",").str[0].str.strip()
        .value_counts()
        .head(20)
        .reset_index()
    )
    loc_df.columns = ["Location", "Count"]
    fig_loc = px.bar(
        loc_df, x="Count", y="Location", orientation="h",
        color="Count", color_continuous_scale="Blues",
    )
    fig_loc.update_layout(height=420, showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig_loc, use_container_width=True)

with row2_right:
    st.subheader("🏢 Jobs by Work Type")
    wt = df["work_type"].dropna().value_counts().reset_index()
    wt.columns = ["Work Type", "Count"]
    if not wt.empty:
        fig_wt = px.pie(wt, names="Work Type", values="Count",
                        color_discrete_sequence=px.colors.sequential.Plasma_r)
        fig_wt.update_layout(height=420)
        st.plotly_chart(fig_wt, use_container_width=True)
    else:
        st.info("Work type data not available.")

st.divider()

# ── Row 3: Experience Level Distribution ──────────────────────────────────────
st.subheader("🎓 Jobs by Experience Level")
exp_df = df["experience_level"].dropna().value_counts().reset_index()
exp_df.columns = ["Level","Count"]
if not exp_df.empty:
    fig_exp = px.bar(exp_df, x="Level", y="Count",
                     color="Count", color_continuous_scale="Teal",
                     text="Count")
    fig_exp.update_traces(textposition="outside")
    fig_exp.update_layout(height=380, showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig_exp, use_container_width=True)

st.caption(f"Data snapshot: {len(df):,} jobs loaded from LinkedIn/Kaggle + Adzuna live feed.")
