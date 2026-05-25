"""Scraper météo — prevision-meteo.ch/climat/journalier/toulouse-blagnac.

Source  : https://prevision-meteo.ch/climat/journalier/toulouse-blagnac/YYYY-MM
Données : journalières → étendues à l'horaire (même valeur sur les 24h d'un jour)
Imputation : moyennes mensuelles pour les dates futures (> aujourd'hui)

Structure du tableau (11 colonnes, index 0-basé) :
  0  Date      1 T°min   2 T°max   3 T°moy
  4  Vent moy  5 Vent max 6 Vent min
  7  Ensoleillement  8 Précip.
  9  Pression min   10 Pression max

Cache : datas/weather_cache.csv  (format CSV, colonnes = _WEATHER_COLS)
"""

from __future__ import annotations

import re
import time
import urllib.request
import urllib.error
from datetime import date, datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd
from loguru import logger

_SITE_BASE = "https://prevision-meteo.ch/climat/journalier/toulouse-blagnac"
_CACHE_PATH = Path("datas/weather_cache.csv")
_WEATHER_COLS = ["observed_at", "temp_celsius", "humidity_pct", "pressure_hpa", "wind_speed_ms", "is_imputed"]

# Indices dans les lignes <td> du tableau (colonne 0 = Date, 10 colonnes au total)
# 0:Date  1:T°min  2:T°max  3:T°moy  4:Vent moy  5:Vent max
# 6:Ensoleillement  7:Précip.  8:Pression min  9:Pression max
_IDX_TEMP_MOY   = 3
_IDX_VENT_MOY   = 4
_IDX_PRESS_MIN  = 8
_IDX_PRESS_MAX  = 9


# ── Parser HTML ────────────────────────────────────────────────────────────────

class _TableRowExtractor(HTMLParser):
    """Collecte les lignes <td> du premier <table> dans <div class='toprint'.

    Le `</div>` ne réinitialise pas l'état car la table contient des div imbriqués
    dans sa <caption>. On s'arrête uniquement sur </table>.
    """

    def __init__(self) -> None:
        super().__init__()
        self._found_toprint = False
        self._in_table      = False
        self._in_row        = False
        self._in_td         = False
        self._cell_buf: list[str] = []
        self._row_buf:  list[str] = []
        self.rows: list[list[str]] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        ad = dict(attrs)
        if tag == "div" and ad.get("class", "").strip() == "toprint":
            self._found_toprint = True
        if self._found_toprint and tag == "table" and not self._in_table:
            self._in_table = True
        if self._in_table and tag == "tr":
            self._in_row  = True
            self._row_buf = []
        if self._in_row and tag == "td":
            self._in_td    = True
            self._cell_buf = []

    def handle_endtag(self, tag: str) -> None:
        # Pas de reset sur </div> : des divs imbriqués existent dans la <caption>
        if self._in_table and tag == "table":
            self._in_table      = False
            self._found_toprint = False   # ne traiter que le premier tableau
        if self._in_table and tag == "tr" and self._in_row:
            if self._row_buf:
                self.rows.append(self._row_buf)
            self._in_row = False
        if self._in_row and tag == "td":
            self._row_buf.append(" ".join(self._cell_buf).strip())
            self._in_td = False

    def handle_data(self, data: str) -> None:
        if self._in_td:
            stripped = data.strip()
            if stripped:
                self._cell_buf.append(stripped)


def _parse_table(html: str) -> list[list[str]]:
    parser = _TableRowExtractor()
    parser.feed(html)
    return parser.rows


# ── Scraping mensuel ───────────────────────────────────────────────────────────

def _to_float(value: str) -> float | None:
    """Convertit une cellule en float ; retourne None pour '--', '' ou non-numérique."""
    v = value.strip().replace(",", ".")
    if not v or v == "--":
        return None
    # Supprime les unités résiduelles (ex. "13.5 °C")
    m = re.match(r"^-?\d+(?:\.\d+)?", v)
    return float(m.group()) if m else None


def _scrape_month(year: int, month: int, delay: float = 0.5) -> list[dict]:
    """Scrape les données journalières d'un mois et retourne une liste de dicts."""
    url = f"{_SITE_BASE}/{year}-{month:02d}"
    req = urllib.request.Request(url, headers={"User-Agent": "InduSense-Pipeline/1.0 (educational)"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} sur {url}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Réseau impossible ({exc.reason}) pour {url}") from exc

    rows = _parse_table(html)
    records: list[dict] = []

    for row in rows:
        if len(row) < 10:
            continue
        # Extrait le numéro de jour depuis "Mar. 01", "Mer. 02", etc.
        day_match = re.search(r"\d+", row[0])
        if not day_match:
            continue
        day = int(day_match.group())
        try:
            day_date = date(year, month, day)
        except ValueError:
            continue

        temp     = _to_float(row[_IDX_TEMP_MOY])
        wind_kmh = _to_float(row[_IDX_VENT_MOY])
        p_min    = _to_float(row[_IDX_PRESS_MIN])
        p_max    = _to_float(row[_IDX_PRESS_MAX])

        wind_ms  = round(wind_kmh / 3.6, 3) if wind_kmh is not None else None
        pressure = round((p_min + p_max) / 2, 2) if (p_min is not None and p_max is not None) else (p_min or p_max)

        records.append({
            "date":         day_date,
            "temp_celsius": temp,
            "humidity_pct": None,   # non disponible sur cette source
            "pressure_hpa": pressure,
            "wind_speed_ms": wind_ms,
        })

    time.sleep(delay)
    logger.debug("Scraped {}/{}: {} jours", year, month, len(records))
    return records


def _expand_daily_to_hourly(records: list[dict]) -> list[dict]:
    """Duplique chaque enregistrement journalier sur les 24 heures UTC du jour."""
    hourly: list[dict] = []
    for r in records:
        for hour in range(24):
            ts = datetime(r["date"].year, r["date"].month, r["date"].day, hour, tzinfo=timezone.utc)
            hourly.append({
                "observed_at":  ts.isoformat(),
                "temp_celsius": r["temp_celsius"],
                "humidity_pct": r["humidity_pct"],
                "pressure_hpa": r["pressure_hpa"],
                "wind_speed_ms": r["wind_speed_ms"],
                "is_imputed":   False,
            })
    return hourly


# ── Cache ──────────────────────────────────────────────────────────────────────

def _load_cache() -> pd.DataFrame:
    if _CACHE_PATH.exists():
        df = pd.read_csv(_CACHE_PATH, parse_dates=["observed_at"])
        df["observed_at"] = pd.to_datetime(df["observed_at"], utc=True)
        return df
    return pd.DataFrame(columns=_WEATHER_COLS)


def _save_cache(df: pd.DataFrame) -> None:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(_CACHE_PATH, index=False)


def _cached_months(cache: pd.DataFrame) -> set[tuple[int, int]]:
    """Retourne l'ensemble des (year, month) présents dans le cache."""
    if cache.empty:
        return set()
    obs = pd.to_datetime(cache["observed_at"], utc=True)
    pairs = pd.DataFrame({"y": obs.dt.year, "m": obs.dt.month}).drop_duplicates()
    return set(zip(pairs["y"].astype(int), pairs["m"].astype(int)))


# ── Imputation moyennes mensuelles ─────────────────────────────────────────────

def _compute_monthly_means(df: pd.DataFrame) -> dict[int, dict[str, float | None]]:
    """Retourne {mois: {metric: mean}} calculé sur les données historiques (non imputées)."""
    hist = df[~df["is_imputed"].astype(bool)].copy()
    if hist.empty:
        return {}
    hist["month"] = pd.to_datetime(hist["observed_at"], utc=True).dt.month
    means: dict[int, dict[str, float | None]] = {}
    for month, grp in hist.groupby("month"):
        means[int(month)] = {
            "temp_celsius":  float(grp["temp_celsius"].mean())  if grp["temp_celsius"].notna().any()  else None,
            "humidity_pct":  None,
            "pressure_hpa":  float(grp["pressure_hpa"].mean())  if grp["pressure_hpa"].notna().any()  else None,
            "wind_speed_ms": float(grp["wind_speed_ms"].mean()) if grp["wind_speed_ms"].notna().any() else None,
        }
    return means


def _impute_date_range(
    start: date,
    end: date,
    monthly_means: dict[int, dict[str, float | None]],
) -> list[dict]:
    """Génère des lignes horaires imputées par moyenne mensuelle pour [start, end]."""
    rows: list[dict] = []
    current = datetime(start.year, start.month, start.day, tzinfo=timezone.utc)
    end_dt  = datetime(end.year,   end.month,   end.day,   23, tzinfo=timezone.utc)
    while current <= end_dt:
        m = monthly_means.get(current.month, {})
        rows.append({
            "observed_at":  current.isoformat(),
            "temp_celsius": m.get("temp_celsius"),
            "humidity_pct": None,
            "pressure_hpa": m.get("pressure_hpa"),
            "wind_speed_ms": m.get("wind_speed_ms"),
            "is_imputed":   True,
        })
        current += timedelta(hours=1)
    return rows


def _iter_months(start: date, end: date):
    """Génère les (year, month) de start à end inclus."""
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        m += 1
        if m > 12:
            m, y = 1, y + 1


# ── API publique ───────────────────────────────────────────────────────────────

def fetch_or_load_weather(
    start: date,
    end: date,
    api_key: str | None = None,   # ignoré — conservé pour compatibilité de signature
    request_delay: float = 0.5,
) -> pd.DataFrame:
    """Retourne un DataFrame horaire couvrant [start, end].

    Stratégie :
    - Mois déjà en cache             → servi depuis le cache
    - Mois historiques manquants     → scrapé depuis prevision-meteo.ch (mois par mois)
    - Dates futures (> aujourd'hui)  → imputées par moyenne mensuelle du même mois calendaire
    """
    today = date.today()
    cache = _load_cache()
    cached_months = _cached_months(cache)

    # ── Scraping des mois historiques manquants ────────────────────────────────
    hist_end = min(end, today)
    new_rows: list[dict] = []

    for y, m in _iter_months(start, hist_end):
        if (y, m) in cached_months:
            continue
        logger.info("Scraping prevision-meteo.ch : {}/{:02d}", y, m)
        try:
            daily = _scrape_month(y, m, delay=request_delay)
            new_rows.extend(_expand_daily_to_hourly(daily))
        except RuntimeError as exc:
            logger.warning("Impossible de scraper {}/{}: {}", y, m, exc)

    if new_rows:
        fetched_df = pd.DataFrame(new_rows)
        fetched_df["observed_at"] = pd.to_datetime(fetched_df["observed_at"], utc=True)
        cache = pd.concat([cache, fetched_df], ignore_index=True)
        cache = cache.drop_duplicates(subset=["observed_at"]).sort_values("observed_at")
        _save_cache(cache)
        cached_months = _cached_months(cache)
        logger.info("Cache mis à jour : {} lignes horaires au total", len(cache))

    # ── Imputation des dates futures ───────────────────────────────────────────
    future_start = max(start, today + timedelta(days=1))
    future_months_missing = [
        (y, m) for y, m in _iter_months(future_start, end)
        if (y, m) not in cached_months
    ] if future_start <= end else []

    if future_months_missing:
        monthly_means = _compute_monthly_means(cache)
        if not monthly_means:
            logger.warning("Aucune donnée historique en cache — imputation impossible.")
        else:
            fs = date(future_months_missing[0][0], future_months_missing[0][1], 1)
            fy, fm = future_months_missing[-1]
            # Dernier jour du dernier mois futur manquant
            fe = (date(fy, fm % 12 + 1, 1) - timedelta(days=1)) if fm < 12 else date(fy, 12, 31)
            fe = min(fe, end)
            imputed = _impute_date_range(fs, fe, monthly_means)
            imp_df = pd.DataFrame(imputed)
            imp_df["observed_at"] = pd.to_datetime(imp_df["observed_at"], utc=True)
            cache = pd.concat([cache, imp_df], ignore_index=True)
            cache = cache.drop_duplicates(subset=["observed_at"]).sort_values("observed_at")
            _save_cache(cache)
            logger.info("Imputation : {} lignes horaires (moyennes mensuelles)", len(imputed))

    # ── Filtre sur la plage demandée ───────────────────────────────────────────
    start_dt = pd.Timestamp(start, tz="UTC")
    end_dt   = pd.Timestamp(end,   tz="UTC") + pd.Timedelta(hours=23)
    result = cache[
        (cache["observed_at"] >= start_dt) & (cache["observed_at"] <= end_dt)
    ].reset_index(drop=True)

    logger.info("Météo prête : {} lignes pour {} → {}", len(result), start, end)
    return result
