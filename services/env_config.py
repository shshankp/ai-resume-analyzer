"""Resolve API keys from OS env, .env (via load_dotenv in app), or Streamlit secrets."""

from __future__ import annotations

import os


def gemini_api_key() -> str:
    k = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if k:
        return k
    try:
        import streamlit as st

        sec = getattr(st, "secrets", None)
        if sec is None:
            return ""
        for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            try:
                if name in sec:
                    v = sec[name]
                    if v:
                        return str(v).strip()
            except Exception:
                continue
    except Exception:
        pass
    return ""


def adzuna_configured() -> bool:
    return bool(os.getenv("ADZUNA_APP_ID", "").strip() and os.getenv("ADZUNA_APP_KEY", "").strip())
