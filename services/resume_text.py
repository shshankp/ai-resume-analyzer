"""Extract plain text from resume files (.txt, .pdf, .docx)."""

from __future__ import annotations

import io
from pathlib import Path


def extract_text(uploaded_file) -> str:
    """Read text from a Streamlit UploadedFile."""
    name = getattr(uploaded_file, "name", "") or ""
    raw = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)
    return _decode_by_suffix(name, raw)


def extract_text_from_path(path: str | Path) -> str:
    p = Path(path)
    return _decode_by_suffix(p.name, p.read_bytes())


def _decode_by_suffix(name: str, raw: bytes) -> str:
    suffix = Path(name).suffix.lower()
    if suffix == ".pdf":
        return _pdf_text(raw)
    if suffix in {".docx"}:
        return _docx_text(raw)
    if suffix == ".doc":
        raise ValueError("Legacy .doc is not supported; save as .docx or PDF.")
    return raw.decode("utf-8", errors="replace")


def _pdf_text(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    parts: list[str] = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts).strip()


def _docx_text(data: bytes) -> str:
    from docx import Document

    doc = Document(io.BytesIO(data))
    return "\n".join(p.text for p in doc.paragraphs if p.text).strip()
