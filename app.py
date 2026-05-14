"""
AI Job Search & Resume Improvement Agent — Streamlit UI.

Run from this directory:
  pip install -r requirements.txt
  copy .env.example .env   # add GEMINI_API_KEY (Google AI Studio)
  streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

# OS / machine env wins over .env values (override=False).
load_dotenv(ROOT / ".env", override=False)
load_dotenv(override=False)

import streamlit as st

from agents import JobFinderAgent, RecommendationAgent, ResumeAnalyzerAgent
from services.env_config import adzuna_configured, gemini_api_key
from services.resume_text import extract_text


st.set_page_config(page_title="AI Job & Resume Agent", layout="wide")
st.title("AI Job Search & Resume Improvement Agent")
st.caption("Agents: Resume Analyzer · Job Finder · Recommendation (match + interview prep)")

resume_analyzer = ResumeAnalyzerAgent()
job_finder = JobFinderAgent()
recommendation = RecommendationAgent()

if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "jobs" not in st.session_state:
    st.session_state.jobs = []
if "matches" not in st.session_state:
    st.session_state.matches = None
if "selected_job" not in st.session_state:
    st.session_state.selected_job = None
if "interview" not in st.session_state:
    st.session_state.interview = None

with st.sidebar:
    st.header("Job search")
    query = st.text_input("Role / keywords", value="software engineer python")
    location = st.text_input("Location (optional)", value="")
    use_llm_rank = st.toggle("Use LLM for job matching (needs Gemini API key)", value=True)
    st.divider()
    gem_ok = bool(gemini_api_key())
    adz_ok = adzuna_configured()
    st.caption("API detection (keys are never shown)")
    st.write("Gemini (resume / match / interview):", "yes" if gem_ok else "no")
    st.write("Adzuna (local job ads):", "yes" if adz_ok else "no")
    if not gem_ok:
        st.info(
            "Set **`GEMINI_API_KEY`** or **`GOOGLE_API_KEY`** in Windows “Environment variables”, "
            "or in project **`.env`**, then **fully restart** the terminal (and Cursor) so Streamlit inherits it."
        )
    st.divider()
    st.markdown(
        "**Live jobs:** Remotive (remote, no key). **Local ads:** `ADZUNA_APP_ID` + `ADZUNA_APP_KEY`. "
        "**AI:** `GEMINI_API_KEY` (or `GOOGLE_API_KEY`)."
    )

uploaded = st.file_uploader("Upload resume (.pdf, .docx, .txt)", type=["pdf", "docx", "txt"])

col_a, col_b = st.columns(2)
with col_a:
    if st.button("1 · Analyze resume", use_container_width=True):
        if not uploaded:
            st.warning("Upload a resume first.")
        else:
            with st.spinner("Resume Analyzer running…"):
                st.session_state.resume_text = extract_text(uploaded)
                st.session_state.analysis = resume_analyzer.analyze(st.session_state.resume_text)
                st.session_state.matches = None
                st.session_state.interview = None
            st.success("Resume analyzed.")
with col_b:
    if st.button("2 · Find jobs", use_container_width=True):
        with st.spinner("Job Finder running…"):
            st.session_state.jobs = job_finder.search(query, location)
            st.session_state.matches = None
            st.session_state.interview = None
        st.success(f"Found {len(st.session_state.jobs)} listing(s).")
        srcs = sorted({j.raw.get("_source", "?") for j in st.session_state.jobs})
        if srcs:
            st.caption("Sources: " + ", ".join(srcs) + " — *Remotive* = live remote board (no key). *adzuna* = local ads (needs API keys). *sample* = demo fallback only.")

if st.button("3 · Match resume to jobs", use_container_width=True):
    if not st.session_state.resume_text or not st.session_state.analysis:
        st.warning("Run resume analysis first.")
    elif not st.session_state.jobs:
        st.warning("Find jobs first (or widen keywords).")
    else:
        with st.spinner("Recommendation Agent scoring…"):
            if use_llm_rank:
                ranked = recommendation.llm_rank_and_advice(
                    st.session_state.resume_text,
                    st.session_state.analysis,
                    st.session_state.jobs,
                )
                st.session_state.matches = ranked or recommendation.rank_jobs(
                    st.session_state.resume_text,
                    st.session_state.analysis,
                    st.session_state.jobs,
                )
            else:
                st.session_state.matches = recommendation.rank_jobs(
                    st.session_state.resume_text,
                    st.session_state.analysis,
                    st.session_state.jobs,
                )
        st.success("Matches ready.")

tab_resume, tab_jobs, tab_match, tab_interview = st.tabs(
    ["Resume analysis", "Job listings", "Match & gaps", "Interview prep"]
)

with tab_resume:
    if st.session_state.analysis:
        a = st.session_state.analysis
        st.subheader("Summary")
        st.write(a.summary or "—")
        st.subheader("Extracted skills")
        st.write(", ".join(a.skills) if a.skills else "—")
        if a.experience_highlights:
            st.subheader("Experience highlights")
            for x in a.experience_highlights:
                st.markdown(f"- {x}")
        if a.education_highlights:
            st.subheader("Education highlights")
            for x in a.education_highlights:
                st.markdown(f"- {x}")
        if a.suggested_improvements:
            st.subheader("Improvement ideas")
            for x in a.suggested_improvements:
                st.markdown(f"- {x}")
    else:
        st.info("Upload a resume and click **Analyze resume**.")

with tab_jobs:
    jobs = st.session_state.jobs
    if not jobs:
        st.info("Set keywords in the sidebar and click **Find jobs**.")
    else:
        st.caption(
            "Live data from [Remotive](https://remotive.com) (remote roles). "
            "Optional: set Adzuna keys in `.env` for country-specific boards. "
            "Per Remotive terms, listings link to their site."
        )
        for j in jobs:
            src = j.raw.get("_source", "")
            label = f"{j.title} — {j.company or 'Company TBC'} ({j.location or 'location n/a'})"
            if src:
                label = f"[{src}] {label}"
            with st.expander(label):
                if j.salary_hint:
                    st.caption(j.salary_hint)
                st.write(j.description or "No description.")
                if j.url:
                    st.markdown(f"[Open posting]({j.url})")

with tab_match:
    matches = st.session_state.matches
    if not matches:
        st.info("Run **Match resume to jobs** after analysis and job search.")
    else:
        for idx, m in enumerate(matches):
            j = m.job
            title = f"{j.title} @ {j.company or '—'} — match {m.match_score:.1f}/100"
            with st.expander(title):
                st.write(m.rationale)
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Aligned skills**")
                    st.write(", ".join(m.matched_skills) or "—")
                with c2:
                    st.markdown("**Gaps / posting keywords**")
                    st.write(", ".join(m.missing_skills) or "—")
                if st.button("Use this job for interview prep", key=f"pick-{idx}"):
                    st.session_state.selected_job = j
                    st.session_state.interview = None
                    st.success("Job selected. Open **Interview prep** tab and generate questions.")

with tab_interview:
    job = st.session_state.selected_job
    st.write(
        "Selected job:"
        f" **{job.title}** @ {job.company}"
        if job
        else "No job selected yet — pick one from **Match & gaps**."
    )
    if st.button("Generate interview questions", use_container_width=True):
        if not st.session_state.resume_text:
            st.warning("Analyze a resume first.")
        else:
            with st.spinner("Generating interview pack…"):
                st.session_state.interview = recommendation.interview_questions(
                    st.session_state.resume_text,
                    job,
                    st.session_state.analysis,
                )
            st.success("Done.")
    pack = st.session_state.interview
    if pack:
        st.subheader("Technical")
        for q in pack.technical_questions:
            st.markdown(f"- {q}")
        st.subheader("Behavioral")
        for q in pack.behavioral_questions:
            st.markdown(f"- {q}")
        if pack.sample_answers_outline:
            st.subheader("Answer outlines (STAR-style)")
            for q in pack.sample_answers_outline:
                st.markdown(f"- {q}")
