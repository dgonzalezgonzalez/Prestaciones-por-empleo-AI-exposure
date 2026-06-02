from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import time
from urllib.parse import urljoin

from bs4 import BeautifulSoup
import joblib
import pandas as pd
import requests

from .config import PipelineConfig
from .embeddings import EmbeddingCache, OllamaEmbeddingClient, embed_texts
from .model import EXPOSURE_COLUMNS, ExposureModelBundle, predict_occupation_exposure
from .taxonomy import load_cno4_records


SEPE_OCCUPATION_PAGE_URL = "https://www.sepe.es/HomeSepe/que-es-observatorio/informacion-mt-por-ocupacion.html"
SEPE_RESULTS_ENDPOINT = (
    "https://www.sepe.es/HomeSepe/que-es-observatorio/"
    "informacion-mt-por-ocupacion/main/04/content/resultados"
)

MONTH_NAME_TO_NUMBER = {
    "enero": "01",
    "febrero": "02",
    "marzo": "03",
    "abril": "04",
    "mayo": "05",
    "junio": "06",
    "julio": "07",
    "agosto": "08",
    "septiembre": "09",
    "octubre": "10",
    "noviembre": "11",
    "diciembre": "12",
}


@dataclass(frozen=True)
class SepeReportLink:
    cno4: str
    occupation_title: str
    period: str
    url: str


def scrape_sepe_monthly_dataset(
    config: PipelineConfig,
    exposure_path: Path | None = None,
    output_path: Path | None = None,
    merged_output_path: Path | None = None,
    model_path: Path | None = None,
    embedding_model: str | None = None,
    refresh: bool = False,
    delay_seconds: float = 0.25,
    max_occupations: int | None = None,
    max_reports: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config.ensure_dirs()
    raw = config.raw_dir / "sepe"
    raw.mkdir(parents=True, exist_ok=True)
    output_path = output_path or config.processed_dir / "sepe_cno4_monthly_long.csv"
    merged_output_path = merged_output_path or config.processed_dir / "sepe_cno4_monthly_ai_exposure_long.csv"

    cno4 = load_cno4_records(config, refresh=False)[["CNO4", "occupation_title"]].drop_duplicates()
    if max_occupations is not None:
        cno4 = cno4.head(max_occupations)

    session = requests.Session()
    links = discover_sepe_report_links(session, cno4, delay_seconds=delay_seconds, max_reports=max_reports)
    rows: list[dict[str, object]] = []
    for idx, link in enumerate(links, start=1):
        html = fetch_cached_report(session, raw, link, refresh=refresh)
        rows.extend(parse_sepe_report_html(html, link.url))
        if delay_seconds and idx < len(links):
            time.sleep(delay_seconds)

    dataset = pd.DataFrame(rows)
    if dataset.empty:
        raise ValueError("SEPE scrape produced no rows.")
    dataset = dataset.sort_values(["cno4", "period", "measure", "dimension", "category", "gender"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)

    exposure = load_cno4_exposure_measures(
        config,
        exposure_path=exposure_path,
        model_path=model_path,
        embedding_model=embedding_model or config.embedding_model,
    )
    exposure = exposure.rename(columns={"occupation_title": "exposure_occupation_title"})
    merged = dataset.merge(exposure, on="cno4", how="left")
    merged.to_csv(merged_output_path, index=False)
    return dataset, merged


def discover_sepe_report_links(
    session: requests.Session,
    occupations: pd.DataFrame,
    delay_seconds: float = 0.25,
    max_reports: int | None = None,
) -> list[SepeReportLink]:
    links: list[SepeReportLink] = []
    for _, occupation in occupations.iterrows():
        cno4 = str(occupation["CNO4"])
        page = 1
        while True:
            html = _fetch_detail_page(session, cno4, page)
            page_links = parse_report_links_from_listing(html, cno4)
            links.extend(page_links)
            if max_reports is not None and len(links) >= max_reports:
                return links[:max_reports]
            if not _listing_has_page(html, page + 1):
                break
            page += 1
            if delay_seconds:
                time.sleep(delay_seconds)
        if delay_seconds:
            time.sleep(delay_seconds)
    return links


def parse_report_links_from_listing(html: str, cno4: str) -> list[SepeReportLink]:
    soup = BeautifulSoup(html, "html.parser")
    links: list[SepeReportLink] = []
    for anchor in soup.find_all("a", href=True):
        href = str(anchor["href"])
        if "informacion-mercado-trabajo-por-ocupacion" not in href or "_mensuales_" not in href:
            continue
        row = anchor.find_parent("tr")
        if row is None:
            continue
        cells = [normalize_text(cell.get_text(" ", strip=True)) for cell in row.find_all("td")]
        if len(cells) < 2:
            continue
        month = MONTH_NAME_TO_NUMBER.get(cells[0].lower())
        year = re.search(r"\d{4}", cells[1])
        if not month or not year:
            continue
        title = _title_from_listing(soup, cno4)
        links.append(
            SepeReportLink(
                cno4=cno4,
                occupation_title=title,
                period=f"{year.group(0)}-{month}",
                url=urljoin(SEPE_OCCUPATION_PAGE_URL, href),
            )
        )
    return links


def _listing_has_page(html: str, page: int) -> bool:
    soup = BeautifulSoup(html, "html.parser")
    for anchor in soup.find_all("a"):
        if normalize_text(anchor.get_text(" ", strip=True)) == str(page):
            href = str(anchor.get("href", ""))
            if "page-pr=" in href:
                return True
    return False


def parse_sepe_report_html(html: str, source_url: str = "") -> list[dict[str, object]]:
    soup = BeautifulSoup(html, "html.parser")
    cno4, title, period = _report_identity(soup, source_url)
    rows: list[dict[str, object]] = []

    persons = _parse_banner_persons(soup)
    if persons is not None:
        rows.append(_row(cno4, title, period, "personas", "total", "Total", persons, source_url))

    for table in soup.find_all("table"):
        caption = normalize_text(table.find("caption").get_text(" ", strip=True) if table.find("caption") else "")
        caption_l = caption.lower()
        if caption_l == "parados según sexo y edad":
            rows.extend(_parse_gender_age_table(table, cno4, title, period, "parados", source_url))
        elif caption_l == "contratos según sexo y edad":
            rows.extend(_parse_gender_age_table(table, cno4, title, period, "contratos", source_url))
        elif caption_l == "distribución geográfica de parados":
            rows.extend(_parse_province_table(table, cno4, title, period, "parados", source_url))
        elif caption_l == "distribución geográfica de contratos":
            rows.extend(_parse_province_table(table, cno4, title, period, "contratos", source_url))
        elif caption_l == "movilidad geográfica de la contratación":
            rows.extend(_parse_mobility_total_table(table, cno4, title, period, source_url))

    rows.extend(_parse_mobility_gender_script(soup, cno4, title, period, source_url))
    return rows


def load_cno4_exposure_measures(
    config: PipelineConfig,
    exposure_path: Path | None = None,
    model_path: Path | None = None,
    embedding_model: str | None = None,
) -> pd.DataFrame:
    if exposure_path and exposure_path.exists():
        return _normalize_exposure_frame(pd.read_csv(exposure_path))

    model_path = model_path or _default_model_path(config, embedding_model or config.embedding_model)
    model: ExposureModelBundle = joblib.load(model_path)
    cno4 = load_cno4_records(config, refresh=False)
    cache = EmbeddingCache(config.cache_dir / "embeddings.sqlite")
    client = OllamaEmbeddingClient(config.ollama_host, embedding_model or config.embedding_model)
    embeddings = embed_texts(cno4["embedding_text"].dropna().unique(), cache, client)
    predictions = predict_occupation_exposure(cno4, embeddings, model, ("rf", "cosine_weighted", "cosine_nearest"))
    keep = ["CNO4", "occupation_title", *[column for column in EXPOSURE_COLUMNS if column in predictions.columns]]
    return _normalize_exposure_frame(predictions[keep])


def _fetch_detail_page(session: requests.Session, cno4: str, page: int) -> str:
    if page == 1:
        response = session.post(
            SEPE_RESULTS_ENDPOINT,
            data={"list-mode": "detail", "ocupacion-id": cno4, "year-busc": "", "month-busc": ""},
            timeout=90,
        )
    else:
        response = session.get(
            SEPE_RESULTS_ENDPOINT,
            params={"list-mode": "detail", "ocupacion-id": cno4, "page-pr": page},
            timeout=90,
        )
    response.raise_for_status()
    return response.text


def fetch_cached_report(session: requests.Session, raw_dir: Path, link: SepeReportLink, refresh: bool = False) -> str:
    path = raw_dir / "reports" / f"{link.cno4}_{link.period}.html"
    if path.exists() and not refresh:
        return path.read_text(encoding="utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    response = session.get(link.url, timeout=90)
    response.raise_for_status()
    path.write_text(response.text, encoding="utf-8")
    return response.text


def _parse_gender_age_table(
    table,
    cno4: str,
    title: str,
    period: str,
    measure: str,
    source_url: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    section = ""
    for tr in table.find_all("tr"):
        cells = [normalize_text(cell.get_text(" ", strip=True)) for cell in tr.find_all(["td", "th"])]
        if not cells:
            continue
        first = cells[0].lower()
        if first == "por sexo":
            section = "gender"
            continue
        if first == "por tramos de edad":
            section = "age"
            continue
        if first in {"", "total", "variación (1)", "mensual"} and len(cells) < 2:
            continue
        value = parse_number(cells[1] if len(cells) > 1 else "")
        if value is None:
            continue
        if section == "gender":
            rows.append(_row(cno4, title, period, measure, "gender", cells[0], value, source_url, gender=cells[0]))
        elif section == "age":
            rows.append(_row(cno4, title, period, measure, "age", cells[0], value, source_url))
    return rows


def _parse_province_table(
    table,
    cno4: str,
    title: str,
    period: str,
    measure: str,
    source_url: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for tr in table.find_all("tr"):
        cells = [normalize_text(cell.get_text(" ", strip=True)) for cell in tr.find_all(["td", "th"])]
        if len(cells) < 2 or cells[0].lower() in {"", "provincia"}:
            continue
        value = parse_number(cells[1])
        if value is not None:
            rows.append(_row(cno4, title, period, measure, "province", cells[0], value, source_url))
    total = sum(float(row["value"]) for row in rows)
    rows.append(_row(cno4, title, period, measure, "province", "Total", total, source_url))
    return rows


def _parse_mobility_total_table(table, cno4: str, title: str, period: str, source_url: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for tr in table.find_all("tr"):
        cells = [normalize_text(cell.get_text(" ", strip=True)) for cell in tr.find_all(["td", "th"])]
        if len(cells) < 2:
            continue
        label = cells[0]
        if label.lower().startswith("nº de contratos que permanecen"):
            category = "Permanecen"
        elif label.lower().startswith("nº de contratos que se mueven"):
            category = "Se mueven"
        else:
            continue
        value = parse_number(cells[1])
        if value is not None:
            rows.append(_row(cno4, title, period, "contratos", "geographic_mobility", category, value, source_url))
    if rows:
        rows.append(
            _row(
                cno4,
                title,
                period,
                "contratos",
                "geographic_mobility",
                "Total",
                sum(float(row["value"]) for row in rows),
                source_url,
            )
        )
    return rows


def _parse_mobility_gender_script(soup, cno4: str, title: str, period: str, source_url: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for script in soup.find_all("script"):
        text = script.get_text("\n")
        if "drawStuffMovilidad" not in text:
            continue
        genders = re.findall(r"addColumn\('number', '([^']+)'\)", text)
        for match in re.finditer(r"addRow\(\['([^']+)'\s*,\s*([^\]]+)\]\)", text):
            category = normalize_text(match.group(1))
            values = [parse_number(value) for value in match.group(2).split(",")]
            for gender, value in zip(genders, values):
                if value is not None:
                    rows.append(
                        _row(
                            cno4,
                            title,
                            period,
                            "contratos",
                            "geographic_mobility",
                            category,
                            value,
                            source_url,
                            gender=gender,
                        )
                    )
        break
    return rows


def _parse_banner_persons(soup) -> float | None:
    banner = None
    for title in soup.find_all("h4", class_="se-databanner--title"):
        if "contratos en esta ocupación" in normalize_text(title.get_text(" ", strip=True)).lower():
            banner = title.find_parent(class_="se-databanner")
            break
    if banner is None:
        return None
    figures = banner.find_all("p", class_="se-databanner--figure")
    for figure in figures:
        if "personas" in normalize_text(figure.get_text(" ", strip=True)).lower():
            digit = figure.find(class_="se-databanner--digit")
            return parse_number(digit.get_text(" ", strip=True) if digit else "")
    return None


def _report_identity(soup, source_url: str) -> tuple[str, str, str]:
    heading = ""
    for tag in soup.find_all(["h1", "h2", "h3"]):
        text = normalize_text(tag.get_text(" ", strip=True))
        if text.startswith("CNO-"):
            heading = text
            break
    match = re.search(r"CNO-(\d{4}):\s+(.+?)\s+([A-Za-zÁÉÍÓÚáéíóúñ]+)\s+(\d{4})$", heading)
    if match:
        cno4, title, month_name, year = match.groups()
        month = MONTH_NAME_TO_NUMBER[month_name.lower()]
        return cno4, title, f"{year}-{month}"
    url_match = re.search(r"_mensuales_(\d{4})_(\d{2})_(\d{4})-", source_url)
    if url_match:
        year, month, cno4 = url_match.groups()
        return cno4, "", f"{year}-{month}"
    raise ValueError("Could not parse SEPE report identity.")


def _title_from_listing(soup, cno4: str) -> str:
    heading = soup.find("h3")
    text = normalize_text(heading.get_text(" ", strip=True) if heading else "")
    match = re.search(rf"CNO-{re.escape(cno4)}:\s+(.+)$", text)
    return match.group(1) if match else ""


def _row(
    cno4: str,
    title: str,
    period: str,
    measure: str,
    dimension: str,
    category: str,
    value: float,
    source_url: str,
    gender: str = "Total",
) -> dict[str, object]:
    return {
        "period": period,
        "cno4": str(cno4).zfill(4),
        "occupation_title": title,
        "measure": measure,
        "dimension": dimension,
        "category": category,
        "gender": gender,
        "value": value,
        "source_url": source_url,
    }


def parse_number(text: str) -> float | None:
    cleaned = normalize_text(str(text)).replace("%", "").replace(".", "").replace(",", ".")
    cleaned = re.sub(r"[^\d.\-]", "", cleaned)
    if cleaned in {"", "-", "."}:
        return None
    return float(cleaned)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def _default_model_path(config: PipelineConfig, embedding_model: str) -> Path:
    slug = embedding_model.replace(":", "_")
    candidates = sorted(config.models_dir.glob(f"exposure_model_{slug}_*rf*cosine_weighted*cosine_nearest.joblib"))
    if not candidates:
        candidates = sorted(config.models_dir.glob(f"exposure_model_{slug}_*.joblib"))
    if not candidates:
        raise FileNotFoundError(f"No exposure model found for embedding model {embedding_model} in {config.models_dir}")
    return candidates[-1]


def _normalize_exposure_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename = {"CNO4": "cno4"}
    out = out.rename(columns=rename)
    out["cno4"] = out["cno4"].astype(str).str.zfill(4)
    if "occupation_title" not in out.columns and "spanish_title" in out.columns:
        out = out.rename(columns={"spanish_title": "occupation_title"})
    keep = ["cno4", "occupation_title", *[column for column in EXPOSURE_COLUMNS if column in out.columns]]
    return out[keep].drop_duplicates(subset=["cno4"])
