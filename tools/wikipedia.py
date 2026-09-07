"""Wikipedia tool for general-knowledge/definitional questions.

Free, public REST API — no key required, but Wikimedia requires a
descriptive User-Agent on all requests (otherwise returns 403).
"""
import re
import requests
from typing import Dict, Any

OPENSEARCH_URL = "https://en.wikipedia.org/w/api.php"
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


def get_wikipedia_summary(question: str) -> Dict[str, Any]:
    """Look up a Wikipedia article relevant to `question` and return its
    summary. Returns a structured dict with an exact source_url, or
    {"error": ...} on failure. Never fabricates content.
    """
    topic = _extract_topic(question)
    try:
        search_resp = requests.get(
            OPENSEARCH_URL,
            params={"action": "opensearch", "search": topic, "limit": 1, "format": "json"},
            headers=HEADERS,
            timeout=10,
        )
        search_resp.raise_for_status()
        results = search_resp.json()
        titles = results[1] if len(results) > 1 else []
        if not titles:
            return {"error": f"No Wikipedia article found for '{topic}'."}
        title = titles[0]
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
