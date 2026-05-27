from __future__ import annotations

from pathlib import Path
import json
import sqlite3
import time
from collections.abc import Iterable, Callable

import requests

from .utils import clean_occupation_title, embedding_cache_key


class EmbeddingCache:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                cache_key TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                normalized_text TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                vector_json TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.commit()

    def get(self, model: str, text: str) -> list[float] | None:
        normalized = clean_occupation_title(text)
        key = embedding_cache_key(model, normalized)
        row = self.conn.execute("SELECT vector_json FROM embeddings WHERE cache_key = ?", (key,)).fetchone()
        if not row:
            return None
        return json.loads(row[0])

    def set(self, model: str, text: str, vector: list[float]) -> None:
        normalized = clean_occupation_title(text)
        key = embedding_cache_key(model, normalized)
        self.conn.execute(
            """
            INSERT OR REPLACE INTO embeddings
            (cache_key, model, normalized_text, dimension, vector_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (key, model, normalized, len(vector), json.dumps(vector)),
        )
        self.conn.commit()


class OllamaEmbeddingClient:
    def __init__(self, host: str, model: str, timeout: int = 120):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout

    def embed(self, text: str) -> list[float]:
        normalized = clean_occupation_title(text)
        errors: list[str] = []
        for endpoint, payload in [
            ("/api/embed", {"model": self.model, "input": normalized}),
            ("/api/embeddings", {"model": self.model, "prompt": normalized}),
        ]:
            try:
                response = requests.post(f"{self.host}{endpoint}", json=payload, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                if "embedding" in data:
                    return [float(x) for x in data["embedding"]]
                if "embeddings" in data and data["embeddings"]:
                    return [float(x) for x in data["embeddings"][0]]
            except Exception as exc:
                errors.append(f"{endpoint}: {exc}")
        raise RuntimeError(
            f"Ollama embedding failed for model '{self.model}' at {self.host}. "
            f"Start Ollama and verify model is installed. Errors: {' | '.join(errors)}"
        )


def embed_texts(
    texts: Iterable[str],
    cache: EmbeddingCache,
    client: OllamaEmbeddingClient,
    sleep_seconds: float = 0.0,
    progress: Callable[[int, str], None] | None = None,
) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for idx, text in enumerate(texts, start=1):
        normalized = clean_occupation_title(text)
        cached = cache.get(client.model, normalized)
        if cached is not None:
            result[normalized] = cached
            continue
        vector = client.embed(normalized)
        cache.set(client.model, normalized, vector)
        result[normalized] = vector
        if progress:
            progress(idx, normalized)
        if sleep_seconds:
            time.sleep(sleep_seconds)
    return result
