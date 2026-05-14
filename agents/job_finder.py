"""Job Finder Agent — live listings (Remotive + optional Adzuna), then sample fallback."""

from __future__ import annotations

import html
import json
import os
import re
from pathlib import Path

import httpx

from models.schemas import JobListing

REMOTIVE_URL = "https://remotive.com/api/remote-jobs"


def _sample_jobs() -> list[JobListing]:
    path = Path(__file__).resolve().parent.parent / "data" / "sample_jobs.json"
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: list[JobListing] = []
    for row in raw:
        out.append(
            JobListing(
                title=row.get("title", "Role"),
                company=row.get("company", ""),
                location=row.get("location", ""),
                description=row.get("description", ""),
                url=row.get("url", ""),
                salary_hint=row.get("salary_hint", ""),
                raw={**row, "_source": "sample"},
            )
        )
    return out


def _strip_html(text: str, max_len: int = 5000) -> str:
    if not text:
        return ""
    t = html.unescape(text)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > max_len:
        t = t[:max_len] + "…"
    return t


def _merge_unique(preferred: list[JobListing], extra: list[JobListing], limit: int) -> list[JobListing]:
    seen: set[str] = set()
    out: list[JobListing] = []
    for j in preferred + extra:
        key = (j.url or "").strip() or f"{j.title}|{j.company}".lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(j)
        if len(out) >= limit:
            break
    return out


def _remotive_category(query: str) -> str | None:
    """Remotive category slug (hint only; API may still return mixed categories)."""
    q = query.lower()
    if any(k in q for k in ("data scientist", "data analyst", "machine learning", "ml ", "data engineer")):
        return "data"
    if any(k in q for k in ("devops", "sre", "kubernetes", "terraform", "site reliability")):
        return "devops"
    if any(k in q for k in ("product manager", "product owner", "pm ")):
        return "product"
    if any(
        k in q
        for k in (
            "developer",
            "engineer",
            "software",
            "programmer",
            "full stack",
            "fullstack",
            "backend",
            "frontend",
            "python",
            "java",
            "react",
            "node",
            "typescript",
            "javascript",
            "ios",
            "android",
            "web ",
        )
    ):
        return "software-dev"
    return None


def _remotive_relevance_score(row: dict, query: str) -> int:
    tokens = [t.lower() for t in re.findall(r"[#+a-zA-Z0-9.-]+", query) if len(t) > 2]
    if not tokens:
        return 1
    tags = [str(t).lower() for t in (row.get("tags") or [])]
    title = str(row.get("title") or "").lower()
    desc = str(row.get("description") or "").lower()[:8000]
    tag_blob = " ".join(tags)
    score = 0
    for t in tokens:
        if t in title:
            score += 5
        elif t in tag_blob:
            score += 3
        elif t in desc:
            score += 1
    return score


class JobFinderAgent:
    role = "Job Finder"

    def search(self, query: str, location: str = "", limit: int = 20) -> list[JobListing]:
        limit = max(5, min(limit, 50))
        q = (query or "").strip()
        loc = (location or "").strip()

        adzuna: list[JobListing] = []
        app_id = os.getenv("ADZUNA_APP_ID", "").strip()
        app_key = os.getenv("ADZUNA_APP_KEY", "").strip()
        country = os.getenv("ADZUNA_COUNTRY", "gb").strip() or "gb"
        if app_id and app_key:
            adzuna = self._adzuna_search(q, loc, country, app_id, app_key, limit)

        remotive = self._remotive_search(q, limit)

        for j in adzuna:
            j.raw["_source"] = "adzuna"
        for j in remotive:
            j.raw["_source"] = "remotive"

        merged = _merge_unique(adzuna + remotive, [], limit)

        if len(merged) < 3:
            samples = _sample_jobs()
            merged = _merge_unique(merged, samples, limit)

        if loc and merged:
            merged = self._prefer_location(merged, loc, limit)

        return merged[:limit]

    def _prefer_location(self, jobs: list[JobListing], location: str, limit: int) -> list[JobListing]:
        loc_l = location.lower()
        scored: list[tuple[int, JobListing]] = []
        for j in jobs:
            hay = f"{j.location} {j.description} {j.title}".lower()
            score = 2 if loc_l in hay else 0
            if "worldwide" in j.location.lower() or "anywhere" in hay:
                score += 1
            scored.append((score, j))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [j for _, j in scored][:limit]

    def _remotive_search(self, query: str, limit: int) -> list[JobListing]:
        fetch_n = min(100, max(limit * 6, 35))
        params: dict[str, str | int] = {"limit": fetch_n}
        if query:
            params["search"] = query
        cat = _remotive_category(query)
        if cat:
            params["category"] = cat
        try:
            r = httpx.get(REMOTIVE_URL, params=params, timeout=25.0, headers={"User-Agent": "JobSearchAgent/1.0"})
            r.raise_for_status()
            payload = r.json()
        except Exception:
            return []

        rows = list(payload.get("jobs") or [])
        q = (query or "").strip()
        if q:
            rows.sort(key=lambda row: _remotive_relevance_score(row if isinstance(row, dict) else {}, q), reverse=True)

        results: list[JobListing] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            desc = row.get("description") or ""
            if isinstance(desc, str) and ("<" in desc and ">" in desc):
                desc = _strip_html(desc)
            elif isinstance(desc, str) and len(desc) > 5000:
                desc = desc[:5000] + "…"
            loc = str(row.get("candidate_required_location") or "")
            if row.get("job_type"):
                loc = f"{loc} · {row['job_type']}".strip(" ·")
            results.append(
                JobListing(
                    title=str(row.get("title") or "Job"),
                    company=str(row.get("company_name") or "").strip(),
                    location=loc,
                    description=desc if isinstance(desc, str) else "",
                    url=str(row.get("url") or ""),
                    salary_hint=str(row.get("salary") or ""),
                    raw=dict(row),
                )
            )
            if len(results) >= limit:
                break
        return results

    def _adzuna_search(
        self,
        query: str,
        location: str,
        country: str,
        app_id: str,
        app_key: str,
        limit: int,
    ) -> list[JobListing]:
        base = f"https://api.adzuna.com/v1/api/jobs/{country}/search/1"
        params = {
            "app_id": app_id,
            "app_key": app_key,
            "results_per_page": min(limit, 20),
            "what": query or "software",
        }
        if location:
            params["where"] = location
        try:
            r = httpx.get(base, params=params, timeout=20.0)
            r.raise_for_status()
            payload = r.json()
        except Exception:
            return []

        results: list[JobListing] = []
        for row in payload.get("results", []):
            desc = row.get("description", "") or ""
            if isinstance(desc, str) and len(desc) > 4000:
                desc = desc[:4000] + "…"
            company = (row.get("company") or {}).get("display_name", "") if isinstance(row.get("company"), dict) else str(row.get("company") or "")
            loc = ""
            if isinstance(row.get("location"), dict):
                loc = row["location"].get("display_name", "") or ""
            results.append(
                JobListing(
                    title=row.get("title", "Job"),
                    company=company,
                    location=loc,
                    description=desc if isinstance(desc, str) else "",
                    url=row.get("redirect_url", "") or "",
                    salary_hint=str(row.get("salary_min", "") or row.get("salary_max", "") or ""),
                    raw=row,
                )
            )
        return results
