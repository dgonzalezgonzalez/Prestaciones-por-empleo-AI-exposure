from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from urllib.parse import urlencode
import hashlib
import os
import sqlite3
import time

import requests

from .utils import clean_occupation_title, normalize_space


DEEPL_FREE_URL = "https://api-free.deepl.com/v2/translate"
DEEPL_PRO_URL = "https://api.deepl.com/v2/translate"
GOOGLE_CLOUD_URL = "https://translation.googleapis.com/language/translate/v2"
GOOGLE_UNOFFICIAL_URL = "https://translate.googleapis.com/translate_a/single"

DOMAIN_TRANSLATION_OVERRIDES = {
    "Trabajadores de servicios de restauración, personales, protección y vendedores de comercio": (
        "Food service, catering, hospitality, personal service, protection, and retail sales workers"
    ),
    "Directores y gerentes de empresas de alojamiento, restauración y comercio": (
        "Directors and managers of accommodation, food service, catering, hospitality, and retail businesses"
    ),
    "Trabajadores asalariados de los servicios de restauración": (
        "Salaried food service, catering, and hospitality workers"
    ),
}


def resolve_translation_provider(provider: str) -> str:
    if provider != "auto":
        return provider
    if os.environ.get("DEEPL_API_KEY"):
        return "deepl"
    if os.environ.get("GOOGLE_TRANSLATE_API_KEY"):
        return "google_cloud"
    return "google_unofficial"


def translation_cache_key(provider_id: str, source_lang: str, target_lang: str, text: str) -> str:
    payload = f"{provider_id}\0{source_lang}\0{target_lang}\0{text}".encode("utf-8")
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
                provider_id TEXT NOT NULL,
                source_lang TEXT NOT NULL,
                target_lang TEXT NOT NULL,
                source_text TEXT NOT NULL,
                translated_text TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.commit()
        columns = {row[1] for row in self.conn.execute("PRAGMA table_info(translations)").fetchall()}
        self.has_legacy_model_column = "model" in columns
        if "provider_id" not in columns and "model" in columns:
            self.conn.execute("ALTER TABLE translations ADD COLUMN provider_id TEXT")
            self.conn.execute("UPDATE translations SET provider_id = model WHERE provider_id IS NULL")
            self.conn.commit()

    def get(self, provider_id: str, source_lang: str, target_lang: str, text: str) -> str | None:
        source = clean_occupation_title(text)
        key = translation_cache_key(provider_id, source_lang, target_lang, source)
        row = self.conn.execute("SELECT translated_text FROM translations WHERE cache_key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set(self, provider_id: str, source_lang: str, target_lang: str, text: str, translated: str) -> None:
        source = clean_occupation_title(text)
        target = normalize_space(translated).strip("\"'")
        key = translation_cache_key(provider_id, source_lang, target_lang, source)
        if self.has_legacy_model_column:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO translations
                (cache_key, model, provider_id, source_lang, target_lang, source_text, translated_text)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (key, provider_id, provider_id, source_lang, target_lang, source, target),
            )
        else:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO translations
                (cache_key, provider_id, source_lang, target_lang, source_text, translated_text)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (key, provider_id, source_lang, target_lang, source, target),
            )
        self.conn.commit()


class TranslationClient:
    def __init__(
        self,
        provider: str,
        model: str = "",
        host: str = "http://127.0.0.1:11434",
        timeout: int = 180,
    ):
        self.provider = resolve_translation_provider(provider)
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.provider_id = self._provider_id()

    def _provider_id(self) -> str:
        if self.provider == "ollama":
            return f"ollama:{self.model}"
        if self.provider == "deepl":
            api_type = os.environ.get("DEEPL_API_TYPE", "free").lower()
            return f"deepl:{api_type}"
        return self.provider

    def translate_to_english(self, text: str) -> str:
        source = clean_occupation_title(text)
        if self.provider == "deepl":
            translated = self._deepl(source)
        elif self.provider == "google_cloud":
            translated = self._google_cloud(source)
        elif self.provider == "google_unofficial":
            translated = self._google_unofficial(source)
        elif self.provider == "ollama":
            translated = self._ollama(source)
        else:
            raise ValueError(
                "Unsupported translation provider. Use auto, deepl, google_cloud, google_unofficial, or ollama."
            )

        translated = normalize_space(translated).strip("\"'")
        if not translated:
            raise RuntimeError(f"Empty translation from provider '{self.provider_id}' for: {source}")
        return translated

    def _deepl(self, source: str) -> str:
        api_key = os.environ.get("DEEPL_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPL_API_KEY is required for --translation-provider deepl.")
        api_type = os.environ.get("DEEPL_API_TYPE", "free").lower()
        url = DEEPL_PRO_URL if api_type == "pro" else DEEPL_FREE_URL
        response = requests.post(
            url,
            headers={"Authorization": f"DeepL-Auth-Key {api_key}"},
            data={"text": source, "source_lang": "ES", "target_lang": "EN-US"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["translations"][0]["text"]

    def _google_cloud(self, source: str) -> str:
        api_key = os.environ.get("GOOGLE_TRANSLATE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_TRANSLATE_API_KEY is required for --translation-provider google_cloud.")
        response = requests.post(
            f"{GOOGLE_CLOUD_URL}?key={api_key}",
            json={"q": source, "source": "es", "target": "en", "format": "text"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["data"]["translations"][0]["translatedText"]

    def _google_unofficial(self, source: str) -> str:
        query = urlencode({"client": "gtx", "sl": "es", "tl": "en", "dt": "t", "q": source})
        response = requests.get(f"{GOOGLE_UNOFFICIAL_URL}?{query}", timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return "".join(part[0] for part in data[0] if part and part[0])

    def _ollama(self, source: str) -> str:
        if not self.model:
            raise RuntimeError("--translation-model is required for --translation-provider ollama.")
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
        return response.json().get("response", "")


def translate_texts_to_english(
    texts: Iterable[str],
    cache: TranslationCache,
    client: TranslationClient,
    sleep_seconds: float = 0.0,
    progress: Callable[[int, str, str], None] | None = None,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for idx, text in enumerate(texts, start=1):
        source = clean_occupation_title(text)
        override = DOMAIN_TRANSLATION_OVERRIDES.get(source)
        if override is not None:
            cache.set(client.provider_id, "es", "en", source, override)
            result[source] = override
            if progress:
                progress(idx, source, override)
            continue
        cached = cache.get(client.provider_id, "es", "en", source)
        if cached is not None:
            result[source] = cached
            continue
        translated = client.translate_to_english(source)
        cache.set(client.provider_id, "es", "en", source, translated)
        result[source] = translated
        if progress:
            progress(idx, source, translated)
        if sleep_seconds:
            time.sleep(sleep_seconds)
    return result
