"""Recommendation Agent — scores fit, gaps, and interview prep."""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher

from models.schemas import InterviewPack, JobListing, JobMatch, ResumeAnalysis
from services import llm_client


def _norm_skill(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _token_overlap(resume_blob: str, job_blob: str) -> tuple[list[str], list[str]]:
    """Very light skill alignment using catalog + substring checks."""
    resume_l = resume_blob.lower()
    job_l = job_blob.lower()
    skills = llm_client.heuristic_skills(resume_blob + "\n" + job_blob)
    matched: list[str] = []
    missing: list[str] = []
    for s in skills:
        n = _norm_skill(s)
        in_cv = n in resume_l or any(n in resume_l for n in [n.replace(" ", ""), n.replace("-", "")])
        in_job = n in job_l or s.lower() in job_l
        if in_job and in_cv:
            matched.append(s)
        elif in_job and not in_cv:
            missing.append(s)
    # de-dupe preserve order
    def dedupe(xs: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for x in xs:
            k = x.lower()
            if k not in seen:
                seen.add(k)
                out.append(x)
        return out

    return dedupe(matched), dedupe(missing)


def _score_text_similarity(a: str, b: str) -> float:
    a2, b2 = a.lower()[:8000], b.lower()[:8000]
    return SequenceMatcher(None, a2, b2).ratio() * 100


class RecommendationAgent:
    role = "Recommendation"

    def rank_jobs(self, resume_text: str, analysis: ResumeAnalysis, jobs: list[JobListing]) -> list[JobMatch]:
        blob = resume_text + "\n" + " ".join(analysis.skills)
        matches: list[JobMatch] = []
        for job in jobs:
            matched, missing = _token_overlap(blob, f"{job.title}\n{job.description}")
            base = _score_text_similarity(blob, f"{job.title}\n{job.description}")
            bonus = min(len(matched) * 4, 40)
            score = max(0.0, min(100.0, base * 0.55 + bonus))
            rationale = (
                f"Lexical overlap {base:.1f}/100; {len(matched)} skill(s) aligned with the posting."
                if matched
                else f"Lexical overlap {base:.1f}/100; limited explicit skill overlap in text."
            )
            matches.append(
                JobMatch(
                    job=job,
                    match_score=round(score, 1),
                    matched_skills=matched,
                    missing_skills=missing,
                    rationale=rationale,
                )
            )
        matches.sort(key=lambda m: m.match_score, reverse=True)
        return matches

    def llm_rank_and_advice(self, resume_text: str, analysis: ResumeAnalysis, jobs: list[JobListing]) -> list[JobMatch] | None:
        """Optional richer ranking via LLM; returns None if unavailable."""
        if not llm_client.client():
            return None
        compact_jobs = []
        for j in jobs[:15]:
            compact_jobs.append(
                {
                    "title": j.title,
                    "company": j.company,
                    "location": j.location,
                    "description": j.description[:2500],
                }
            )
        data = llm_client.chat_json(
            system=(
                "You compare a resume to job postings. Return JSON: "
                '{ "matches": [ { "job_index": number, "match_score": 0-100, '
                '"matched_skills": [], "missing_skills": [], "rationale": "" } ] } '
                "job_index is 0-based into the provided jobs array."
            ),
            user=_json_payload(resume_text, analysis, compact_jobs),
        )
        if not data or "matches" not in data:
            return None
        out: list[JobMatch] = []
        for row in data["matches"]:
            idx = int(row.get("job_index", -1))
            if idx < 0 or idx >= len(jobs):
                continue
            out.append(
                JobMatch(
                    job=jobs[idx],
                    match_score=float(min(100, max(0, row.get("match_score", 0)))),
                    matched_skills=[str(x) for x in row.get("matched_skills", [])],
                    missing_skills=[str(x) for x in row.get("missing_skills", [])],
                    rationale=str(row.get("rationale", "")),
                )
            )
        out.sort(key=lambda m: m.match_score, reverse=True)
        return out or None

    def interview_questions(
        self,
        resume_text: str,
        job: JobListing | None,
        analysis: ResumeAnalysis | None = None,
    ) -> InterviewPack:
        job_ctx = ""
        if job:
            job_ctx = f"Role: {job.title} at {job.company}\nPosting:\n{job.description[:6000]}"
        skill_hint = ", ".join((analysis.skills if analysis else [])[:25])
        data = llm_client.chat_json(
            system=(
                "You are a hiring interviewer. Reply with JSON only. Keys: "
                "technical_questions (array of 6-8 strings): specific to this resume and target role; "
                "reference real tools from the resume. "
                "behavioral_questions (array of 5-7 strings): each must ask for a concrete past story and "
                "explicitly mention STAR (Situation, Task, Action, Result) or 'Tell me about a time when…'. "
                "No generic textbook questions only — tie themes to the job (ownership, deadlines, quality, teamwork). "
                "sample_answers_outline (array, SAME length as technical_questions): each item is ONE string containing "
                "3-5 bullet sentences separated by ' | ' telling the candidate how to answer that technical question "
                "using STAR and facts they should pull from their resume (not empty platitudes)."
            ),
            user=(
                f"Extracted skills (hints, may be incomplete):\n{skill_hint or '(none)'}\n\n"
                f"Resume:\n{resume_text[:8000]}\n\n{job_ctx}"
            ),
        )
        if data:
            tech = [str(x).strip() for x in data.get("technical_questions", []) if str(x).strip()]
            beh = [str(x).strip() for x in data.get("behavioral_questions", []) if str(x).strip()]
            outlines = [str(x).strip() for x in data.get("sample_answers_outline", []) if str(x).strip()]
            outlines = _align_answer_outlines(outlines, tech)
            if len(tech) >= 3 and len(beh) >= 3:
                return InterviewPack(
                    technical_questions=tech,
                    behavioral_questions=beh,
                    sample_answers_outline=outlines,
                )
        return _interview_fallback(resume_text, job, analysis)


def _align_answer_outlines(outlines: list[str], technical: list[str]) -> list[str]:
    if not technical:
        return outlines
    default = (
        "Situation: one line of context from your work. | Task: what you were accountable for. | "
        "Action: tools, code, or process steps you took. | Result: measurable outcome (latency, users, revenue, defects)."
    )
    out = list(outlines)
    while len(out) < len(technical):
        out.append(default)
    return out[: len(technical)]


def _interview_fallback(resume_text: str, job: JobListing | None, analysis: ResumeAnalysis | None) -> InterviewPack:
    role = job.title if job else "this type of role"
    top = (analysis.skills if analysis and analysis.skills else llm_client.heuristic_skills(resume_text))[:8]
    stack = ", ".join(top) if top else "the technologies listed on your resume"
    first = top[0] if top else "your main language or framework"

    technical = [
        f"Walk through a feature or system you built that used {first} — APIs, data layer, and how you validated it before release.",
        f"How would you approach debugging a production issue in a stack like: {stack}?",
        f"Compare two tools from your experience (e.g. from {stack}) and when you would choose each.",
        "Describe how you structure components or modules for maintainability on a project you list on your resume.",
        "How do you test changes (unit, integration, manual) before shipping?",
        "Tell me about integrating an external API or third-party service — auth, errors, retries, and monitoring.",
    ]
    behavioral = [
        f"STAR: Tell me about a time you owned delivery end-to-end for work related to {role}; what was at risk and how you measured success?",
        "STAR: Describe disagreeing with a teammate or stakeholder on approach — how you resolved it while keeping quality.",
        "STAR: A tight deadline or shifting requirements — how you re-scoped, communicated, and what shipped as a result?",
        "STAR: A bug or incident you contributed to — how you mitigated, learned, and what process changed afterward?",
        "STAR: A time you improved performance, reliability, or cost — baseline, change, and quantified impact?",
    ]
    outlines = [
        f"Pick one shipped item from the resume using {first}. | Situation + Task in 2 sentences. | Action: stack steps ({stack}). | Result: numbers or user impact.",
        "Name the failing symptom → hypotheses → how you isolated (logs, metrics, repro). | Fix + verification. | Prevention: test or guard added.",
        "Give criteria: team skill, ecosystem, performance, hiring. | One example project each side. | Clear recommendation for this role.",
        "Layers you use (UI, state, API, DB). | One concrete module name or feature from your experience. | Tradeoff you accepted and why.",
        "List test types you run locally vs CI. | Example of a regression you caught. | Definition of 'done' for your team.",
        "Auth model, rate limits, error codes, idempotency. | How you mocked or stubbed in tests. | Rollout or feature flag if relevant.",
    ]
    return InterviewPack(
        technical_questions=technical,
        behavioral_questions=behavioral,
        sample_answers_outline=_align_answer_outlines(outlines, technical),
    )


def _json_payload(resume_text: str, analysis: ResumeAnalysis, jobs: list[dict]) -> str:
    return json.dumps(
        {
            "resume_excerpt": resume_text[:6000],
            "extracted_skills": analysis.skills,
            "jobs": jobs,
        },
        ensure_ascii=False,
    )
