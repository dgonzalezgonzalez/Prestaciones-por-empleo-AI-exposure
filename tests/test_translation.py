from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock

from src.translation import TranslationCache, TranslationClient, translate_texts_to_english


class TranslationTests(TestCase):
    def _tmp_path(self, name: str) -> Path:
        root = Path.cwd() / "test_artifacts"
        root.mkdir(parents=True, exist_ok=True)
        path = root / name
        if path.exists():
            path.unlink()
        return path

    def test_translation_cache_reuses_cleaned_source_text(self):
        cache = TranslationCache(self._tmp_path("translations.sqlite"))
        client = Mock(spec=TranslationClient)
        client.provider_id = "translation-provider"
        client.translate_to_english.return_value = "Managers and executives"

        first = translate_texts_to_english(["Directores y gerentes (códigos CNO-2011)"], cache, client)
        second = translate_texts_to_english(["Directores y gerentes"], cache, client)

        self.assertEqual(first["Directores y gerentes"], "Managers and executives")
        self.assertEqual(second["Directores y gerentes"], "Managers and executives")
        client.translate_to_english.assert_called_once()
