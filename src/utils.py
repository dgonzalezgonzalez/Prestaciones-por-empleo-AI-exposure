from __future__ import annotations

from pathlib import Path
import hashlib
import re
import unicodedata


CODE_PAREN_RE = re.compile(r"\([^)]*(?:c[oó]digos?|CNO|CNAE)[^)]*\)", re.IGNORECASE)
SPACE_RE = re.compile(r"\s+")


def normalize_space(value: str) -> str:
    return SPACE_RE.sub(" ", str(value).replace("\xa0", " ")).strip()


def clean_occupation_title(value: str) -> str:
    """Clean occupation title before embedding.

    Removes non-semantic code annotations such as "(códigos CNO-2011)".
    """
    text = normalize_space(value)
    text = CODE_PAREN_RE.sub("", text)
    text = re.sub(r"\s+\.", ".", text)
    text = re.sub(r"\.\s*", ". ", text)
    text = re.sub(r"(?:\s*\.\s*)+$", "", text)
    text = normalize_space(text)
    return text


def embedding_cache_key(model: str, text: str) -> str:
    payload = f"{model}\0{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ascii_slug(value: str) -> str:
    stripped = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-zA-Z0-9]+", "-", stripped).strip("-").lower()
