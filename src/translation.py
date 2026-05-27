from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
import hashlib
import sqlite3
import time

import requests

from .utils import clean_occupation_title, normalize_space


def translation_cache_key(model: str, source_lang: str, target_lang: str, text: str) -> str:
    payload = f"{model}\0{source_lang}\0{target_lang}\0{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class TranslationCache:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS translations (
                cache_key TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                source_lang TEXT NOT NULL,
                target_lang TEXT NOT NULL,
                source_text TEXT NOT NULL,
                translated_text TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.commit()

    def get(self, model: str, source_lang: str, target_lang: str, text: str) -> str | None:
        source = clean_occupation_title(text)
        key = translation_cache_key(model, source_lang, target_lang, source)
        row = self.conn.execute("SELECT translated_text FROM translations WHERE cache_key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set(self, model: str, source_lang: str, target_lang: str, text: str, translated: str) -> None:
        source = clean_occupation_title(text)
        target = normalize_space(translated).strip("\"'")
        key = translation_cache_key(model, source_lang, target_lang, source)
        self.conn.execute(
            """
            INSERT OR REPLACE INTO translations
            (cache_key, model, source_lang, target_lang, source_text, translated_text)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (key, model, source_lang, target_lang, source, target),
        )
        self.conn.commit()


class OllamaTranslationClient:
    def __init__(self, host: str, model: str, timeout: int = 180):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout

    def translate_to_english(self, text: str) -> str:
        source = clean_occupation_title(text)
        prompt = (
            "Translate the following Spanish job occupation title into concise English. "
            "Return only the English translation, with no quotation marks, explanation, or alternatives.\n\n"
            f"Spanish: {source}\nEnglish:"
        )
        response = requests.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        translated = normalize_space(response.json().get("response", ""))
        translated = translated.strip().strip("\"'")
        if not translated:
            raise RuntimeError(f"Empty translation from Ollama model '{self.model}' for: {source}")
        return translated


def translate_texts_to_english(
    texts: Iterable[str],
    cache: TranslationCache,
    client: OllamaTranslationClient,
    sleep_seconds: float = 0.0,
    progress: Callable[[int, str, str], None] | None = None,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for idx, text in enumerate(texts, start=1):
        source = clean_occupation_title(text)
        cached = cache.get(client.model, "es", "en", source)
        if cached is not None:
            result[source] = cached
            continue
        translated = client.translate_to_english(source)
        cache.set(client.model, "es", "en", source, translated)
        result[source] = translated
        if progress:
            progress(idx, source, translated)
        if sleep_seconds:
            time.sleep(sleep_seconds)
    return result
