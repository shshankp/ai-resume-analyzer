"""Gemini (Google Gen AI) wrapper with graceful degradation when no API key."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from google import genai
from google.genai import types

from services.env_config import gemini_api_key

# Longer phrases first so substring matching prefers specific tools (e.g. "node.js" before "node").
_SKILL_CATALOG: tuple[str, ...] = (
    "machine learning",
    "deep learning",
    "computer vision",
    "natural language processing",
    "scikit-learn",
    "tensorflow",
    "pytorch",
    "langchain",
    "kubernetes",
    "postgresql",
    "tailwind css",
    "tailwind",
    "typescript",
    "javascript",
    "express.js",
    "express",
    "node.js",
    "nodejs",
    "react.js",
    "react native",
    "socket.io",
    "redux toolkit",
    "redux",
    "zustand",
    "next.js",
    "nextjs",
    "vue.js",
    "angular",
    "spring boot",
    "spring",
    "fastapi",
    "django",
    "flask",
    "mongodb",
    "mongoose",
    "redis",
    "kafka",
    "rabbitmq",
    "graphql",
    "rest api",
    "docker",
    "terraform",
    "ansible",
    "jenkins",
    "github actions",
    "ci/cd",
    "aws",
    "azure",
    "gcp",
    "google cloud",
    "firebase",
    "cloudinary",
    "stripe",
    "shopify",
    "liquid",
    "wordpress",
    "html",
    "css",
    "sass",
    "scss",
    "jquery",
    "bootstrap",
    "java",
    "kotlin",
    "swift",
    "swiftui",
    "c++",
    "c#",
    ".net",
    "rust",
    "go",
    "golang",
    "python",
    "ruby",
    "rails",
    "php",
    "laravel",
    "sql",
    "mysql",
    "sqlite",
    "pandas",
    "numpy",
    "spark",
    "hadoop",
    "airflow",
    "dbt",
    "snowflake",
    "tableau",
    "power bi",
    "excel",
    "git",
    "jira",
    "confluence",
    "figma",
    "postman",
    "insomnia",
    "nginx",
    "linux",
    "unix",
    "bash",
    "powershell",
    "agile",
    "scrum",
    "kanban",
    "jest",
    "mocha",
    "cypress",
    "playwright",
    "selenium",
    "pytest",
    "junit",
    "opencv",
    "nlp",
)


def client() -> genai.Client | None:
    key = gemini_api_key()
    if not key:
        return None
    return genai.Client(api_key=key)


def model_id() -> str:
    return os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip() or "gemini-2.0-flash"


def _response_text(response: Any) -> str:
    try:
        return (response.text or "").strip()
    except Exception:
        parts: list[str] = []
        for c in getattr(response, "candidates", None) or []:
            content = getattr(c, "content", None)
            for p in getattr(content, "parts", None) or []:
                t = getattr(p, "text", None)
                if t:
                    parts.append(t)
        return "".join(parts).strip()


def _parse_json_blob(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return None


def chat_json(system: str, user: str) -> dict[str, Any] | None:
    c = client()
    if not c:
        return None
    try:
        response = c.models.generate_content(
            model=model_id(),
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.2,
                response_mime_type="application/json",
                max_output_tokens=8192,
            ),
        )
        return _parse_json_blob(_response_text(response))
    except Exception:
        return None


def chat_text(system: str, user: str) -> str | None:
    c = client()
    if not c:
        return None
    try:
        response = c.models.generate_content(
            model=model_id(),
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.4,
                max_output_tokens=4096,
            ),
        )
        return _response_text(response) or None
    except Exception:
        return None


def _phrase_in_blob(blob: str, phrase: str) -> bool:
    pl = phrase.lower()
    if not pl:
        return False
    # Multi-word or symbols like .js / C++ / CI/CD → substring match
    if " " in pl or any(c in pl for c in (".", "+", "#", "/")):
        return pl in blob
    # Single token: avoid "java" in "javascript", "react" in "reactive"
    pat = re.escape(phrase)
    return bool(re.search(rf"(?<![a-z0-9#+./]){pat}(?![a-z0-9#+./])", blob, re.I))


def heuristic_skills(text: str) -> list[str]:
    """Keyword-only skill extraction when the LLM is unavailable (no random Title Case words)."""
    blob = re.sub(r"\s+", " ", text.lower())
    found: list[str] = []
    seen: set[str] = set()
    for phrase in sorted(set(_SKILL_CATALOG), key=len, reverse=True):
        pl = phrase.lower()
        if not _phrase_in_blob(blob, phrase):
            continue
        if pl in seen:
            continue
        seen.add(pl)
        found.append(phrase)
    for extra in _skills_from_skills_section(text):
        low = extra.lower()
        if low not in seen and 2 <= len(extra) <= 42:
            seen.add(low)
            found.append(extra)
    return sorted(found, key=str.lower)


def _skills_from_skills_section(text: str) -> list[str]:
    """Parse comma / pipe / bullet lists after a Skills-style heading."""
    out: list[str] = []
    m = re.search(
        r"(?im)^\s*(technical\s+skills|core\s+skills|key\s+skills|skills|technologies|tech\s+stack)\s*[:\-]?\s*(.+?)(?:\n\s*\n|\n(?=[A-Z][a-z]+\s*[:\-]?\s*$)|\Z)",
        text,
        re.DOTALL,
    )
    if not m:
        return out
    chunk = m.group(2)
    chunk = re.sub(r"<[^>]+>", " ", chunk)
    for part in re.split(r"[,|•·\n;]+", chunk):
        s = re.sub(r"\s+", " ", part).strip(" -•\t")
        if 2 <= len(s) <= 40 and not re.fullmatch(r"\d{1,4}", s):
            if not re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b", s, re.I):
                out.append(s)
    return out[:25]


def _truncate(s: str, max_len: int) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1].rsplit(" ", 1)[0] + "…"


def _guess_name_line(text: str) -> str:
    stop = {
        "faridabad", "delhi", "gurgaon", "noida", "ghaziabad", "haryana", "india", "bangalore",
        "mumbai", "hyderabad", "pune", "chennai", "kolkata", "jaipur", "lucknow",
    }
    for line in text.strip().splitlines():
        s = line.strip()
        if not s or "http" in s.lower():
            continue
        if "@" in s and len(s) < 50:
            continue
        if re.search(r"\d{5,}", s):
            s = re.split(r"\d{5,}", s, maxsplit=1)[0].strip(" ,;-\t")
        if not s:
            continue
        parts: list[str] = []
        for w in s.split():
            wl = w.lower().strip(".,;")
            if wl in stop or re.fullmatch(r"\d+", wl):
                break
            if re.match(r"^[A-Z][A-Za-z]{1,24}$", w) or re.match(r"^[A-Z]{2,15}$", w):
                parts.append(w)
            else:
                break
        if 2 <= len(parts) <= 5:
            return " ".join(parts)
        if 8 <= len(s) <= 55 and re.match(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4}$", s):
            return s
        break
    return ""


def _extract_skills_prose(text: str) -> str:
    """Flatten Technical Skills subsection; stop before Internships/Experience even on same line."""
    m = re.search(
        r"(?is)\b(technical\s+skills|core\s+skills|key\s+skills)\b\s*[:\-]?\s*(.+)",
        text,
    )
    if not m:
        m = re.search(r"(?is)^\s*skills\s*[:\-]\s*(.+)", text, re.MULTILINE)
        chunk = m.group(1).strip() if m else ""
    else:
        chunk = m.group(2).strip()
    if not chunk:
        return ""
    chunk = re.split(r"(?i)\b(internships?|experience|work\s+history|projects?)\b", chunk, maxsplit=1)[0]
    chunk = re.sub(r"<[^>]+>", " ", chunk)
    chunk = re.sub(r"\s+", " ", chunk).strip()
    chunk = re.sub(r"(?i)\b(developer\s+tools|tools|skills)\s*:\s*", "; ", chunk)
    chunk = re.sub(r"^[;\s,]+", "", chunk)
    chunk = re.sub(r"\s*;\s*", " | ", chunk)
    return _truncate(chunk, 320)


def _extract_internship_blurb(text: str) -> str:
    m = re.search(
        r"(?is)\b(internships?|experience|work\s+history)\b\s*[:\-]?\s*(.+?)(?=\n\s*\n|\n\s*(projects?|education|technical\s+skills|certifications|achievements)\b|\Z)",
        text,
    )
    if not m:
        return ""
    chunk = re.sub(r"<[^>]+>", " ", m.group(2))
    chunk = re.sub(r"\s+", " ", chunk).strip()
    return _truncate(chunk, 280)


def _extract_education_blurb(text: str) -> str:
    """One line: degree + college + CGPA when layout splits across lines."""
    blob = text[:5000]
    parts: list[str] = []
    deg = None
    if re.search(r"\bMCA\b", blob, re.I):
        deg = "MCA"
    elif re.search(r"\bM\.?Tech\b|\bMS\b", blob, re.I):
        deg = re.search(r"\b(M\.?Tech|MS)\b", blob, re.I)
        deg = deg.group(1).upper() if deg else "Postgraduate program"
    elif re.search(r"\bBCA\b", blob, re.I):
        deg = "BCA"
    elif re.search(r"\bB\.?Tech\b|\bB\.?Sc\b", blob, re.I):
        d = re.search(r"\b(B\.?Tech|B\.?Sc)\b", blob, re.I)
        deg = d.group(1) if d else "Undergraduate program"
    if deg:
        parts.append(deg)
    col = re.search(
        r"\b([A-Z][A-Za-z]+(?:\s+[A-Za-z&]+){1,14}\s+(?:College|University|Institute))\b",
        blob,
    )
    if col:
        parts.append(f"at {_truncate(col.group(1).strip(), 72)}")
    cgpas = re.findall(r"(?i)CGPA[:\s-]*(\d+(?:\.\d+)?)", blob)
    cgpa_str = ""
    if cgpas:
        cgpa_str = f"CGPA {cgpas[0]}" + (f"; earlier CGPA {cgpas[1]}" if len(cgpas) > 1 else "")
    if deg and col:
        line = f"{deg} at {col.group(1).strip()}"
        return f"{line}, {cgpa_str}" if cgpa_str else line
    if deg and cgpa_str:
        return f"{deg}, {cgpa_str}"
    if col and cgpa_str:
        return f"{col.group(1).strip()}, {cgpa_str}"
    return ", ".join(p for p in parts if p) if parts else ""


def heuristic_resume_summary(text: str, max_len: int = 560) -> str:
    """Summary without LLM: never dump raw header+education; synthesize profile + skills + work."""
    t = text.strip()
    if not t:
        return ""

    section = re.search(
        r"(?is)\b(summary|professional\s+summary|profile|career\s+objective|about\s*me)\b\s*[:\-]?\s*(.+?)(?=\n\s*(experience|education|projects|skills|work\s+history|internships?)\b|\Z)",
        t,
    )
    if section:
        body = re.sub(r"<[^>]+>", " ", section.group(2))
        body = re.sub(r"\s+", " ", body).strip()
        if len(body) >= 80:
            return _truncate(body, max_len)

    name = _guess_name_line(t)
    skills_line = _extract_skills_prose(t)
    intern_line = _extract_internship_blurb(t)
    edu_line = _extract_education_blurb(t)

    sentences: list[str] = []
    if name:
        role_hint = "MCA student"
        if re.search(r"\bBCA\b", t, re.I) and not re.search(r"\bMCA\b", t, re.I):
            role_hint = "BCA graduate"
        sentences.append(
            f"{name} is a {role_hint} focused on software engineering, with strengths from coursework and internships."
        )
    if edu_line:
        sentences.append(f"Education: {edu_line}.")
    if skills_line:
        sentences.append(f"Technical strengths include {skills_line}.")
    elif heuristic_skills(t):
        sentences.append(f"Stack signals include {', '.join(heuristic_skills(t)[:12])}.")
    if intern_line and skills_line and intern_line.lower() in skills_line.lower():
        intern_line = ""
    if intern_line:
        sentences.append(f"Internship highlight: {intern_line}.")

    out = " ".join(sentences).strip()
    if len(out) < 70:
        tail = re.sub(r"<[^>]+>", " ", t)
        tail = re.sub(r"\s+", " ", tail).strip()
        if re.search(r"(?i)\b(mca|bca|b\.?tech|internship)\b", tail[:800]):
            out = _truncate(
                "Candidate profile (auto-built): " + tail[:400],
                max_len,
            )
        else:
            out = _truncate(tail, max_len) if tail else "Resume text could not be summarized automatically."
    return _truncate(out, max_len)


def heuristic_improvements(text: str, skills: list[str]) -> list[str]:
    """Concrete, document-aware suggestions when the LLM is unavailable."""
    tips: list[str] = []
    low = text.lower()
    exp = ""
    for label in ("internships", "internship", "experience", "work history"):
        m = re.search(
            rf"(?is)\b{re.escape(label)}\b\s*[:\-]?\s*(.+?)(?=\n\s*\n|\n\s*(projects?|education|technical\s+skills|skills|certifications)\b|\Z)",
            text,
        )
        if m:
            exp += " " + m.group(1)
            break

    has_metric_story = bool(
        re.search(
            r"\d+\s*%|\d+\s*(users|hrs?|days|months|k\b|m\b)|\b(increased|reduced|decreased|improved|optimized|delivered|shipped|launched)\b",
            exp,
            re.I,
        )
    )
    if not has_metric_story and len(exp.strip()) > 15:
        tips.append(
            "Internships/experience read as **titles and dates only** - add 2-3 bullets each with **verbs + stack + outcome** "
            '(example: "Built internal HTML/CSS pages; cut manual steps for ~N users").'
        )

    if re.search(r"(?is)\b(cgpa|gpa)\b", text[:3500]) and not re.search(r"(?i)\b(project|portfolio|github|gitlab)\b", text[:5000]):
        tips.append(
            "Strong **CGPA** blocks dominate early pages; add a **3-line Profile** under your name (target role + stack + one internship win) "
            "so recruiters see fit before coursework detail."
        )

    if re.search(r"(?is)developer\s+tools\s*:", text) and re.search(r"(?is)(^|\n)\s*skills\s*:", text):
        tips.append(
            "Split **Developer tools** vs **Languages & stack** (e.g. C/C++/Java/Python | MySQL | Git/GitHub | VS Code) and mirror **keywords** from job ads you want."
        )

    if re.search(r"(?i)\boops?\b|object[-\s]?oriented|\bsdlc\b|\bdata\s+structures?\b", low):
        tips.append(
            "Keep **OOP / DSA / SDLC** under *Coursework* or *Foundations*; lead the scannable skills line with **job-facing tools** "
            "(languages, DB, Git) you can defend in interview."
        )

    if not re.search(r"linkedin\.com/in/|github\.com/", low):
        tips.append("Add **LinkedIn** + **GitHub** (or portfolio) hyperlinks beside contact - reviewers expect code or activity signals for dev roles.")

    if not re.search(r"(?i)\b(project|capstone|hackathon|portfolio)\b", text[:6000]):
        tips.append(
            "Add **2 small projects** (course or personal): name, stack, one metric (load time, test count, dataset size), and a link - "
            "this offsets short internship duration."
        )

    if re.search(r"(?i)web\s+development", exp) and not re.search(r"(?i)\b(react|node|django|flask|spring|api|rest)\b", exp):
        org = "your internship"
        if re.search(r"(?i)\bnhpc\b", exp):
            org = "NHPC"
        tips.append(
            f"**Web Development** ({org}) should name **deliverables** (pages, forms, dashboards), **stack** beyond HTML/CSS/JS if any, "
            f"and **deployment** (even static hosting) so it lines up with web/backend job descriptions."
        )

    if re.search(r"(?i)\b(nhpc|limited|pvt|ltd)\b", exp) and len(exp.strip()) < 140:
        tips.append(
            "Short org lines need **bullets**: problem -> your action -> tool -> result (even estimated), not only company + month range."
        )

    tips.append(
        "Run a **keyword pass** against 3 target job descriptions: paste missing verbs/tools into your bullets (without fabricating experience)."
    )

    seen: set[str] = set()
    out: list[str] = []
    for x in tips:
        k = x[:60]
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out[:8]
