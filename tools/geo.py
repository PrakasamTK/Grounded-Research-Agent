"""Country/geography facts tool — free, keyless public REST API.

NOTE: restcountries.com v3.1 (the historically common choice) was retired
mid-2026 and its successor (v5) now requires a paid API key, so it no longer
satisfies the "free public API" requirement. countries.dev is a free,
keyless mirror serving the same well-known country-data shape (name,
capital, population, region, area, etc.), verified working at build time.
"""
import re
import requests
from typing import Dict, Any

COUNTRIES_NAME_URL = "https://countries.dev/name/{name}"

STOPWORDS = {
    "what", "is", "the", "of", "population", "capital", "country",
    "area", "gdp", "border", "borders", "continent", "how", "big",
    "in", "a", "for", "does", "have", "and", "size", "?",
}


def _extract_country_name(question: str) -> str:
    words = re.findall(r"[A-Za-z]+", question)
    candidates = [w for w in words if w.lower() not in STOPWORDS]
    return " ".join(candidates) if candidates else question


def get_country_info(question: str) -> Dict[str, Any]:
    """Extract a likely country name from the question and fetch facts.

    Returns a structured dict with an exact source_url, or {"error": ...}.
    Never fabricates data — everything returned comes directly from the
    API response for the current run.
    """
    name = _extract_country_name(question)
    url = COUNTRIES_NAME_URL.format(name=requests.utils.quote(name))
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return {"error": f"Country lookup failed for '{name}' (HTTP {resp.status_code})."}
        data = resp.json()
        if not data:
            return {"error": f"No country data found for '{name}'."}
        candidates = data if isinstance(data, list) else [data]

        # Prefer an exact (case-insensitive) name match over the first fuzzy hit.
        top = next(
            (c for c in candidates if str(c.get("name", "")).lower() == name.lower()),
            candidates[0],
        )

        return {
            "country": top.get("name", name),
            "capital": top.get("capital"),
            "population": top.get("population"),
            "region": top.get("region"),
            "subregion": top.get("subregion"),
            "area_km2": top.get("area"),
            "source_url": resp.url,
        }
    except Exception as e:
        return {"error": f"Country data request failed: {e}"}
