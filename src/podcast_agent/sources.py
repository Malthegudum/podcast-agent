from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader


MAX_SOURCE_CHARS = 80_000


def _is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _read_url(url: str) -> str:
    response = requests.get(
        url,
        timeout=20,
        headers={"User-Agent": "podcast-agent/0.1"},
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    return "\n".join(soup.stripped_strings)


def _read_pdf(path: Path) -> str:
    reader = PdfReader(path)
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def _read_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(path)
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")
    raise ValueError(f"Unsupported file type: {path}. Use .txt, .md or .pdf.")


def load_sources(sources: list[str]) -> str:
    """Load URLs or local source files into one bounded text context."""
    chunks: list[str] = []

    for source in sources:
        if _is_url(source):
            text = _read_url(source)
        else:
            path = Path(source).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"Source not found: {path}")
            text = _read_file(path)

        chunks.append(f"\n--- SOURCE: {source} ---\n{text.strip()}")

    combined = "\n".join(chunks)
    if len(combined) > MAX_SOURCE_CHARS:
        combined = combined[:MAX_SOURCE_CHARS]
        combined += "\n\n[Source context truncated in this MVP.]"
    return combined
