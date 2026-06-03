from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin

from bs4 import BeautifulSoup
import joblib
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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


def build_sepe_dataset_from_cached_reports(
    config: PipelineConfig,
    output_path: Path | None = None,
    exposure_path: Path | None = None,
    model_path: Path | None = None,
    embedding_model: str | None = None,
    workers: int = 8,
    progress: Callable[[str], None] | None = None,
    progress_every: int = 250,
    batch_size: int = 50,
) -> pd.DataFrame:
    config.ensure_dirs()
    raw_reports = config.raw_dir / "sepe" / "reports"
    output_path = output_path or config.processed_dir / "sepe_cno4_monthly_ai_exposure.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)

    _progress(progress, "Loading CNO4 AI exposure measures...")
    exposure = load_cno4_exposure_measures(
        config,
        exposure_path=exposure_path,
        model_path=model_path,
        embedding_model=embedding_model or config.embedding_model,
    ).rename(columns={"occupation_title": "exposure_occupation_title"})

    files = sorted(raw_reports.glob("*.html"))
    if not files:
        raise FileNotFoundError(f"No SEPE cached report HTML files found in {raw_reports}")
    _progress(progress, f"Parsing {len(files):,} cached SEPE report files with {workers} workers...")

    total_reports = 0
    total_rows = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for batch_start in range(0, len(files), batch_size):
            file_batch = files[batch_start : batch_start + batch_size]
            batch_chunks: list[pd.DataFrame] = []
            for path, rows in executor.map(_parse_cached_report_file, file_batch):
                chunk = pd.DataFrame(rows)
                if chunk.empty:
                    continue
                batch_chunks.append(sepe_long_to_compact_wide(chunk))
                total_reports += 1
                total_rows += len(batch_chunks[-1])
            if batch_chunks:
                _append_compact_batch(output_path, batch_chunks, exposure)
            _progress(progress, f"wrote batch at {total_reports:,}/{len(files):,} reports; rows {total_rows:,}; output {output_path}")
            if progress_every <= 1 or total_reports % progress_every == 0:
                last_path = file_batch[-1] if file_batch else ""
                _progress(progress, f"cached reports {total_reports:,}/{len(files):,}; rows {total_rows:,}; last {last_path}")
    _progress(progress, f"Wrote {total_rows:,} rows from {total_reports:,} cached reports to {output_path}.")
    return pd.DataFrame({"rows": [total_rows], "reports": [total_reports], "output": [str(output_path)]})


def _append_compact_batch(path: Path, chunks: list[pd.DataFrame], exposure: pd.DataFrame) -> None:
    batch = pd.concat(chunks, ignore_index=True)
    batch = batch.merge(exposure, on="cno4", how="left")
    _append_csv(path, batch)


def _parse_cached_report_file(path: Path) -> tuple[Path, list[dict[str, object]]]:
    match = re.match(r"^(\d{4})_(\d{4}-\d{2})\.html$", path.name)
    cno4 = match.group(1) if match else ""
    period = match.group(2) if match else ""
    source_url = f"cache://sepe/reports/{path.name}"
    html = path.read_text(encoding="utf-8")
    rows = parse_sepe_report_html(html, source_url)
    if cno4 or period:
        for row in rows:
            row["cno4"] = cno4 or row["cno4"]
            row["period"] = period or row["period"]
    return path, rows


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
    progress: Callable[[str], None] | None = None,
    resume: bool = False,
    workers: int = 1,
    progress_every: int = 25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config.ensure_dirs()
    raw = config.raw_dir / "sepe"
    raw.mkdir(parents=True, exist_ok=True)
    output_path = output_path or config.processed_dir / "sepe_cno4_monthly_ai_exposure.csv"
    if merged_output_path is not None:
        output_path = merged_output_path

    cno4 = load_cno4_records(config, refresh=False)[["CNO4", "occupation_title"]].drop_duplicates()
    if max_occupations is not None:
        cno4 = cno4.head(max_occupations)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not resume:
        output_path.unlink(missing_ok=True)
    exposure = load_cno4_exposure_measures(
        config,
        exposure_path=exposure_path,
        model_path=model_path,
        embedding_model=embedding_model or config.embedding_model,
    )
    exposure = exposure.rename(columns={"occupation_title": "exposure_occupation_title"})

    session = make_sepe_session()
    completed = _completed_report_keys(output_path) if resume else set()
    completed_counts = _completed_counts(completed)
    total_reports = 0
    total_rows = 0
    total_occupations = len(cno4)
    for occ_idx, (_, occupation) in enumerate(cno4.iterrows(), start=1):
        if max_reports is not None and total_reports >= max_reports:
            break
        occ_df = pd.DataFrame([occupation])
        remaining = None if max_reports is None else max_reports - total_reports
        cno4_code = str(occupation["CNO4"]).zfill(4)
        if resume and completed_counts.get(cno4_code, 0) >= 50:
            links = parse_report_links_from_listing(_fetch_detail_page(session, cno4_code, 1), cno4_code)
        else:
            links = discover_sepe_report_links(session, occ_df, delay_seconds=delay_seconds, max_reports=remaining)
        links = [link for link in links if (link.cno4, link.period) not in completed]
        _progress(progress, f"[{occ_idx}/{total_occupations}] CNO4 {occupation['CNO4']}: {len(links)} reports")
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_fetch_parse_report, raw, link, refresh) for link in links]
                for future in as_completed(futures):
                    total_reports, total_rows = _write_report_chunk(
                        future.result(),
                        exposure,
                        output_path,
                        total_reports,
                        total_rows,
                        progress,
                        progress_every,
                    )
                    if max_reports is not None and total_reports >= max_reports:
                        break
        else:
            for link in links:
                html = fetch_cached_report(session, raw, link, refresh=refresh)
                total_reports, total_rows = _write_report_chunk(
                    (link, parse_sepe_report_html(html, link.url)),
                    exposure,
                    output_path,
                    total_reports,
                    total_rows,
                    progress,
                    progress_every,
                )
                if delay_seconds:
                    time.sleep(delay_seconds)
                if max_reports is not None and total_reports >= max_reports:
                    break
    if total_rows == 0:
        raise ValueError("SEPE scrape produced no rows.")
    _progress(progress, f"Wrote {total_rows:,} rows from {total_reports:,} reports.")
    return pd.DataFrame({"rows": [total_rows], "reports": [total_reports]}), pd.DataFrame(
        {"rows": [total_rows], "reports": [total_reports]}
    )


def _fetch_parse_report(raw_dir: Path, link: SepeReportLink, refresh: bool) -> tuple[SepeReportLink, list[dict[str, object]]]:
    session = make_sepe_session()
    html = fetch_cached_report(session, raw_dir, link, refresh=refresh)
    return link, parse_sepe_report_html(html, link.url)


def _write_report_chunk(
    result: tuple[SepeReportLink, list[dict[str, object]]],
    exposure: pd.DataFrame,
    output_path: Path,
    total_reports: int,
    total_rows: int,
    progress: Callable[[str], None] | None,
    progress_every: int,
) -> tuple[int, int]:
    link, rows = result
    chunk = pd.DataFrame(rows)
    if chunk.empty:
        return total_reports, total_rows
    chunk = sepe_long_to_compact_wide(chunk)
    merged = chunk.merge(exposure, on="cno4", how="left")
    _append_csv(output_path, merged)
    total_reports += 1
    total_rows += len(chunk)
    if progress_every <= 1 or total_reports % progress_every == 0:
        _progress(progress, f"  report {total_reports}: {link.cno4} {link.period}, rows+{len(chunk)}, total rows {total_rows:,}")
    return total_reports, total_rows


def sepe_long_to_compact_wide(long: pd.DataFrame) -> pd.DataFrame:
    index_columns = ["period", "cno4", "occupation_title", "dimension", "category", "gender", "source_url"]
    df = long.copy()
    df["cno4"] = df["cno4"].astype(str).str.zfill(4)
    total_source = df[df["category"] == "Total"].copy()
    total_source = total_source.sort_values(["measure", "dimension"]).drop_duplicates(
        subset=["period", "cno4", "measure"], keep="first"
    )
    total_source["dimension"] = "total"
    total_source["category"] = "Total"
    total_source["gender"] = "Total"
    detail = df[(df["dimension"] != "total") & (df["category"] != "Total")].copy()
    compact_long = pd.concat([detail, total_source], ignore_index=True)
    wide = (
        compact_long.pivot_table(
            index=index_columns,
            columns="measure",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for column in ["contratos", "parados", "personas"]:
        if column not in wide.columns:
            wide[column] = pd.NA
    columns = [*index_columns, "contratos", "parados", "personas"]
    return wide[columns].sort_values(["cno4", "period", "dimension", "category", "gender"]).reset_index(drop=True)


def make_sepe_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST"),
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


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
            timeout=(10, 25),
        )
    else:
        response = session.get(
            SEPE_RESULTS_ENDPOINT,
            params={"list-mode": "detail", "ocupacion-id": cno4, "page-pr": page},
            timeout=(10, 25),
        )
    response.raise_for_status()
    return response.text


def fetch_cached_report(session: requests.Session, raw_dir: Path, link: SepeReportLink, refresh: bool = False) -> str:
    path = raw_dir / "reports" / f"{link.cno4}_{link.period}.html"
    if path.exists() and not refresh:
        return path.read_text(encoding="utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    response = session.get(link.url, timeout=(10, 25))
    response.raise_for_status()
    path.write_text(response.text, encoding="utf-8")
    return response.text


def _append_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, mode="a", header=not path.exists())


def _completed_report_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    keys: set[tuple[str, str]] = set()
    for chunk in pd.read_csv(path, usecols=["cno4", "period"], dtype={"cno4": "string", "period": "string"}, chunksize=250_000):
        chunk["cno4"] = chunk["cno4"].astype(str).str.zfill(4)
        keys.update((str(row.cno4), str(row.period)) for row in chunk.drop_duplicates().itertuples(index=False))
    return keys


def _completed_counts(keys: set[tuple[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for cno4, _period in keys:
        counts[cno4] = counts.get(cno4, 0) + 1
    return counts


def _progress(callback: Callable[[str], None] | None, message: str) -> None:
    if callback is not None:
        callback(message)


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
