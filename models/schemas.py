from __future__ import annotations

from pydantic import BaseModel, Field


class ResumeAnalysis(BaseModel):
    summary: str = ""
    skills: list[str] = Field(default_factory=list)
    experience_highlights: list[str] = Field(default_factory=list)
    education_highlights: list[str] = Field(default_factory=list)
    suggested_improvements: list[str] = Field(default_factory=list)


class JobListing(BaseModel):
    title: str
    company: str = ""
    location: str = ""
    description: str = ""
    url: str = ""
    salary_hint: str = ""
    raw: dict = Field(default_factory=dict)


class JobMatch(BaseModel):
    job: JobListing
    match_score: float = Field(ge=0, le=100)
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    rationale: str = ""


class InterviewPack(BaseModel):
    technical_questions: list[str] = Field(default_factory=list)
    behavioral_questions: list[str] = Field(default_factory=list)
    sample_answers_outline: list[str] = Field(default_factory=list)
