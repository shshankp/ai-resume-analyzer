"""Resume Analyzer Agent — parses resume text and extracts structured signals."""

from __future__ import annotations

from models.schemas import ResumeAnalysis
from services import llm_client


class ResumeAnalyzerAgent:
    role = "Resume Analyzer"

    def analyze(self, resume_text: str) -> ResumeAnalysis:
        text = resume_text.strip()
        if not text:
            return ResumeAnalysis(summary="Empty resume text.", skills=[])

        heur_summary = llm_client.heuristic_resume_summary(text)
        heur_skills = llm_client.heuristic_skills(text)
        heur_improve = llm_client.heuristic_improvements(text, heur_skills)

        data = llm_client.chat_json(
            system=(
                "You extract structured resume data. Reply ONLY with valid JSON and these keys: "
                "summary (3-5 sentences: synthesize education, strongest skills, and work/internships — "
                "do NOT paste raw contact lines or long grade tables; write for a recruiter skimming in 10 seconds), "
                "skills (array: languages, frameworks, DB, tools only — max 25 short tokens), "
                "experience_highlights (2-5 bullets from internships/jobs), education_highlights (1-3 short lines), "
                "suggested_improvements (5-7 bullets that cite **specific** issues in THIS resume: section order, "
                "missing links, vague internship wording, skill grouping, lack of metrics — avoid one-size-fits-all clichés)."
            ),
            user=f"Resume:\n{text[:12000]}",
        )
        if data:
            llm_skills = [str(s).strip() for s in data.get("skills", []) if str(s).strip()]
            skills = _merge_skill_lists(llm_skills, heur_skills)[:30]
            summary = str(data.get("summary", "")).strip() or heur_summary
            llm_imp = [str(s).strip() for s in data.get("suggested_improvements", []) if str(s).strip()]
            improvements = _merge_improvements(llm_imp, heur_improve, 10)
            return ResumeAnalysis(
                summary=summary,
                skills=skills[:30],
                experience_highlights=[str(s) for s in data.get("experience_highlights", []) if s],
                education_highlights=[str(s) for s in data.get("education_highlights", []) if s],
                suggested_improvements=improvements,
            )

        return ResumeAnalysis(
            summary=heur_summary,
            skills=heur_skills,
            experience_highlights=[],
            education_highlights=[],
            suggested_improvements=heur_improve,
        )


def _merge_skill_lists(llm: list[str], heur: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in llm + heur:
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out[:30]


def _merge_improvements(llm: list[str], heur: list[str], limit: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in llm + heur:
        t = s.strip()
        if not t:
            continue
        k = t[:72].lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
        if len(out) >= limit:
            break
    return out
