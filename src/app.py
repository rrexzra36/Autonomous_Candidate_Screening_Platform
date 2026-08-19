"""
Autonomous Candidate Screening Platform - Streamlit HR Dashboard PoC
Run with: streamlit run src/app.py
"""

import streamlit as st
import json
import os
import pandas as pd
from anonymizer import BlindCVAnonymizer
from matcher import CandidateMatcherEngine

# Page Config
st.set_page_config(
    page_title="TalentAI - Candidate Screening Engine",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Sample Data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "sample_data")

@st.cache_data
def load_data():
    with open(os.path.join(DATA_DIR, "job_descriptions.json"), "r", encoding="utf-8") as f:
        jobs = json.load(f)
    with open(os.path.join(DATA_DIR, "sample_cvs.json"), "r", encoding="utf-8") as f:
        cvs = json.load(f)
    return jobs, cvs

jobs, raw_cvs = load_data()

# Header
st.title("🤖 Autonomous Candidate Screening Platform")
st.caption("AI-Powered Talent Acquisition Engine with Blind Anonymization & Explainable AI (XAI)")

st.markdown("---")

# Sidebar Controls
st.sidebar.header("⚙️ Screening Configuration")

selected_job_title = st.sidebar.selectbox(
    "Select Target Job Opening:",
    [j["title"] for j in jobs]
)
selected_job = next(j for j in jobs if j["title"] == selected_job_title)

min_score = st.sidebar.slider("Minimum Shortlist Score Threshold:", 0, 100, 60, 5)

enable_blind_cv = st.sidebar.toggle("🛡️ Enable Blind-CV Anonymization (Bias Shield)", value=True)

# Display Job Requirement Overview
with st.expander(f"📋 Active Job Opening Details: **{selected_job['title']}**", expanded=False):
    st.write(f"**Department:** {selected_job['department']}")
    st.write(f"**Min Education:** {selected_job['hard_requirements']['min_education']}")
    st.write(f"**Min Experience:** {selected_job['hard_requirements']['min_experience_years']} Years")
    st.write(f"**Key Skills:** {', '.join(selected_job['key_skills'])}")
    st.write(f"**Description:** {selected_job['description']}")

# Process CVs
matcher = CandidateMatcherEngine()
evaluated_results = []

for raw_cv in raw_cvs:
    cv_to_process = BlindCVAnonymizer.anonymize_cv(raw_cv) if enable_blind_cv else raw_cv
    eval_res = matcher.evaluate_candidate(cv_to_process, selected_job)
    eval_res["raw_cv"] = raw_cv
    eval_res["anonymized_cv"] = cv_to_process
    evaluated_results.append(eval_res)

# Sort Results by Score
evaluated_results.sort(key=lambda x: x["overall_score"], reverse=True)

# Main Dashboard Tabs
tab1, tab2, tab3 = st.tabs(["🏆 Candidate Leaderboard & XAI", "🛡️ Blind-CV Bias Shield Audit", "📊 Metrics & Analytics"])

with tab1:
    st.subheader("Candidate Evaluation & Ranking")
    
    # Filter by threshold
    filtered_list = [c for c in evaluated_results if c["overall_score"] >= min_score]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Applicants", len(evaluated_results))
    col2.metric("Shortlisted Candidates", len(filtered_list))
    col3.metric("Avg Match Score", f"{round(sum(c['overall_score'] for c in evaluated_results)/len(evaluated_results), 1)}%")

    st.markdown("### Ranked Candidates")
    
    for rank, item in enumerate(filtered_list, start=1):
        status_color = "🟢" if item["status"] == "SHORTLISTED" else ("🟡" if item["status"] == "CONSIDERATION" else "🔴")
        display_name = item["candidate_alias"] if enable_blind_cv else item["raw_cv"]["personal_info"]["full_name"]
        
        with st.container():
            st.markdown(f"#### #{rank} {status_color} {display_name} — **{item['overall_score']}% Match**")
            
            c1, c2, c3, c4 = st.columns([2, 2, 2, 3])
            c1.write(f"**Status:** `{item['status']}`")
            c2.write(f"**Skill Fit:** {item['score_breakdown']['skill_match']}%")
            c3.write(f"**Exp Fit:** {item['score_breakdown']['experience_depth']}%")
            
            with st.expander("🔍 Read Explainable AI (XAI) Justification & Interview Questions"):
                p1, p2 = st.columns(2)
                with p1:
                    st.markdown("**✅ Strengths (Pros):**")
                    for pro in item["justification"]["pros"]:
                        st.markdown(f"- {pro}")
                with p2:
                    st.markdown("**⚠️ Potential Gaps (Cons):**")
                    for con in item["justification"]["cons"]:
                        st.markdown(f"- {con}")
                
                st.markdown("**❓ Recommended Interview Questions:**")
                for q in item["justification"]["interview_questions"]:
                    st.markdown(f"1. *{q}*")
                
                b1, b2 = st.columns(2)
                b1.button(f"Approve for Interview", key=f"app_{item['cv_id']}")
                b2.button(f"Reject Candidate", key=f"rej_{item['cv_id']}")
            st.markdown("---")

with tab2:
    st.subheader("🛡️ Blind-CV Anonymization Demonstration")
    st.info("Demonstrating how PII (Name, Gender, Photo, Age, Institution Specific Prestige) is stripped to eliminate human bias before LLM scoring.")
    
    selected_cv_id = st.selectbox("Select Candidate CV to Audit:", [c["cv_id"] for c in raw_cvs])
    target_item = next(c for c in evaluated_results if c["cv_id"] == selected_cv_id)
    
    left_col, right_col = st.columns(2)
    with left_col:
        st.error("❌ Raw Unmasked CV Data (Vulnerable to Bias)")
        st.json(target_item["raw_cv"]["personal_info"])
    with right_col:
        st.success("✅ Anonymized Blind-CV Data (Sent to AI Engine)")
        st.json(target_item["anonymized_cv"]["personal_info"])

with tab3:
    st.subheader("📊 Screening Performance Metrics")
    df_chart = pd.DataFrame([
        {
            "Candidate": c["candidate_alias"] if enable_blind_cv else c["raw_cv"]["personal_info"]["full_name"],
            "Overall Score": c["overall_score"],
            "Skill Fit": c["score_breakdown"]["skill_match"],
            "Experience Fit": c["score_breakdown"]["experience_depth"]
        } for c in evaluated_results
    ])
    st.bar_chart(df_chart.set_index("Candidate"))

st.caption("Developed for AI Specialist Assessment | Autonomous Candidate Screening Platform v1.0.0")
