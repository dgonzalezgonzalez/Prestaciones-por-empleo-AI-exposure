from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock

from src.embeddings import EmbeddingCache, embed_texts
from src.utils import clean_occupation_title


class EmbeddingTests(TestCase):
    def _tmp_path(self, name: str) -> Path:
        root = Path.cwd() / "test_artifacts"
        root.mkdir(parents=True, exist_ok=True)
        path = root / name
        if path.exists():
            path.unlink()
        return path

    def test_clean_occupation_title_removes_cno_code_parenthetical(self):
        self.assertEqual(
            clean_occupation_title("Directores comerciales (códigos CNO-2011)"),
            "Directores comerciales",
        )

    def test_cache_reuses_embedding_for_cleaned_text(self):
        cache = EmbeddingCache(self._tmp_path("embeddings.sqlite"))
        client = Mock()
        client.model = "test-model"
        client.embed.return_value = [0.1, 0.2]

        first = embed_texts(["Directores comerciales (códigos CNO-2011)"], cache, client)
        second = embed_texts(["Directores comerciales"], cache, client)

        self.assertEqual(first["Directores comerciales"], [0.1, 0.2])
        self.assertEqual(second["Directores comerciales"], [0.1, 0.2])
        client.embed.assert_called_once()
