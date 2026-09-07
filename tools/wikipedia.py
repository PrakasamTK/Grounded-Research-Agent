"""Wikipedia tool for general-knowledge/definitional questions.

Free, public REST API — no key required, but Wikimedia requires a
descriptive User-Agent on all requests (otherwise returns 403).
"""
import re
import requests
from typing import Dict, Any, Optional

API_URL = "https://en.wikipedia.org/w/api.php"
SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"

HEADERS = {
    "User-Agent": "GroundedResearchAgent/1.0 (educational screening-assignment project)"
}

_BOILERPLATE_PATTERNS = [
    r"^what('s| is| are)\s+",
    r"^define\s+",
    r"^explain\s+",
    r"^who (is|was)\s+",
    r"^tell me about\s+",
    r"^how does\s+",
]

# Trailing qualifier clause, e.g. "CI/CD in cloud computing" -> "CI/CD".
# Tried as a fallback when the full phrase doesn't match an article title.
_TRAILING_QUALIFIER = re.compile(r"\s+(in|for|of|on|within)\s+.+$", re.IGNORECASE)


def _extract_topic(question: str) -> str:
    q = question.strip().rstrip("?").strip()
    lowered = q.lower()
    for pattern in _BOILERPLATE_PATTERNS:
        new_lowered = re.sub(pattern, "", lowered, count=1)
        if new_lowered != lowered:
            q = q[len(lowered) - len(new_lowered):]
            lowered = new_lowered
            break
    q = re.sub(r"\s+work(s)?$", "", q.strip(), flags=re.IGNORECASE)
    return q.strip() or question.strip()


def _opensearch_title(query: str) -> Optional[str]:
    """Exact/prefix title match — high precision. Returns a title or None."""
    resp = requests.get(
        API_URL,
        params={"action": "opensearch", "search": query, "limit": 1, "format": "json"},
        headers=HEADERS,
        timeout=10,
    )
    resp.raise_for_status()
    results = resp.json()
    titles = results[1] if len(results) > 1 else []
    return titles[0] if titles else None


def _fulltext_search_title(query: str) -> Optional[str]:
    """Relevance-ranked full-text search — better recall, lower precision.
    Used only as a last resort; the LLM synthesis step independently checks
    whether the returned content actually supports the question before
    answering, so a loosely-related fallback article still can't produce a
    fabricated claim — worst case it triggers the honest refusal instead.
    """
    resp = requests.get(
        API_URL,
        params={"action": "query", "list": "search", "srsearch": query, "srlimit": 1, "format": "json"},
        headers=HEADERS,
        timeout=10,
    )
    resp.raise_for_status()
    hits = resp.json().get("query", {}).get("search", [])
    return hits[0]["title"] if hits else None


def _resolve_title(topic: str) -> Optional[str]:
    title = _opensearch_title(topic)
    if title:
        return title

    stripped = _TRAILING_QUALIFIER.sub("", topic).strip()
    if stripped and stripped.lower() != topic.lower():
        title = _opensearch_title(stripped)
        if title:
            return title

    return _fulltext_search_title(topic)


def get_wikipedia_summary(question: str) -> Dict[str, Any]:
    """Look up a Wikipedia article relevant to `question` and return its
    summary. Returns a structured dict with an exact source_url, or
    {"error": ...} on failure. Never fabricates content.
    """
    topic = _extract_topic(question)
    try:
        title = _resolve_title(topic)
        if not title:
            return {"error": f"No Wikipedia article found for '{topic}'."}
    except Exception as e:
        return {"error": f"Wikipedia search failed: {e}"}

    try:
        summary_resp = requests.get(
            SUMMARY_URL.format(title=requests.utils.quote(title)),
            headers=HEADERS,
            timeout=10,
        )
        if summary_resp.status_code != 200:
            return {"error": f"Wikipedia summary lookup failed for '{title}' (HTTP {summary_resp.status_code})."}
        data = summary_resp.json()
        extract = data.get("extract")
        if not extract:
            return {"error": f"Wikipedia article '{title}' has no usable summary."}
        return {
            "topic": data.get("title", title),
            "description": data.get("description", ""),
            "extract": extract,
            "source_url": data.get("content_urls", {}).get("desktop", {}).get("page")
            or f"https://en.wikipedia.org/wiki/{requests.utils.quote(title)}",
        }
    except Exception as e:
        return {"error": f"Wikipedia summary request failed: {e}"}
