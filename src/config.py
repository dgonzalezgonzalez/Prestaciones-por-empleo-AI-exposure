from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PipelineConfig:
    root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    interim_dir: Path = PROJECT_ROOT / "data" / "interim"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    cache_dir: Path = PROJECT_ROOT / "data" / "cache"
    models_dir: Path = PROJECT_ROOT / "models"
    db_path: Path = PROJECT_ROOT / "data" / "processed" / "spanish_ai_exposure.sqlite"
    ollama_host: str = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    embedding_model: str = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    random_seed: int = int(os.environ.get("AI_EXPOSURE_RANDOM_SEED", "20260527"))
    n_estimators: int = int(os.environ.get("AI_EXPOSURE_RF_TREES", "500"))

    def ensure_dirs(self) -> None:
        for path in [
            self.raw_dir,
            self.interim_dir,
            self.processed_dir,
            self.cache_dir,
            self.models_dir,
            self.raw_dir / "anthropic",
            self.raw_dir / "ine",
        ]:
            path.mkdir(parents=True, exist_ok=True)


ANTHROPIC_JOB_EXPOSURE_URL = (
    "https://huggingface.co/datasets/Anthropic/EconomicIndex/raw/main/"
    "labor_market_impacts/job_exposure.csv"
)

INE_EPA_MICRODATA_PAGE = (
    "https://ine.es/dyngs/INEbase/es/operacion.htm?"
    "c=Estadistica_C&cid=1254736176918&menu=resultados&"
    "secc=1254736030639&idp=1254735976595"
)
