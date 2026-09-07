"""Hacker News search tool, via the official Algolia HN Search API.

Free, public, and keyless — no registration or credentials required.

Chosen as the social/opinion discussion source instead of Reddit: Reddit
closed self-serve API registration to new developers in late 2025 (the
"Responsible Builder Policy"), so `/prefs/apps` no longer issues working
PRAW credentials for new accounts. See README for the full rationale.
"""
import html
import re
import requests
from typing import List, Dict, Any

SEARCH_URL = "https://hn.algolia.com/api/v1/search"
ITEM_URL = "https://hn.algolia.com/api/v1/items/{id}"

# Question boilerplate that dilutes Algolia search relevance if sent as-is.
_BOILERPLATE_PATTERNS = [
    r"^what do people think (about|of)\s*",
    r"^what are (common |the )?(complaints|opinions|reviews) (about|on|of)\s*",
    r"^what('s| is) the opinion on\s*",
    r"^what are people saying about\s*",
    r"^(opinions|thoughts|discussion|reviews) (about|on)\s*",
    r"^(what|how|do|does|is|are)\s+",
]


def _clean(text: str) -> str:
    if not text:
        return ""
    return html.unescape(text)


def _extract_search_terms(question: str) -> str:
    """Strip generic question boilerplate so the remaining keywords carry
    the actual search relevance (e.g. 'What do people think about electric
    vehicles?' -> 'electric vehicles').
    """
    lowered = question.strip().rstrip("?").strip().lower()
    for pattern in _BOILERPLATE_PATTERNS:
        lowered = re.sub(pattern, "", lowered, count=1).strip()
    return lowered or question.strip()


def search_hackernews(query: str, limit: int = 5, comments_per_story: int = 3) -> List[Dict[str, Any]]:
    """Search Hacker News stories relevant to `query`.

    Returns a list of structured dicts. Never invents URLs or content —
    everything returned came directly from the Algolia HN API response.
    On failure, returns [] (caller records the error separately).
    """
    search_terms = _extract_search_terms(query)
    try:
        resp = requests.get(SEARCH_URL, params={"query": search_terms, "tags": "story"}, timeout=10)
        resp.raise_for_status()
        hits = resp.json().get("hits", [])[:limit]
    except Exception:
        return []

    results: List[Dict[str, Any]] = []
    for hit in hits:
        object_id = hit.get("objectID")
        title = _clean(hit.get("title") or hit.get("story_title") or "")
        story_text = _clean(hit.get("story_text") or "")
        hn_url = f"https://news.ycombinator.com/item?id={object_id}"

        top_comments: List[str] = []
        try:
            item = requests.get(ITEM_URL.format(id=object_id), timeout=10).json()
            for child in (item.get("children") or [])[:comments_per_story]:
                text = _clean(child.get("text") or "")
                if text:
                    top_comments.append(text)
        except Exception:
            pass

        results.append({
            "title": title,
            "source": "Hacker News",
            "body": story_text,
            "comments": top_comments,
            "url": hn_url,
            "score": hit.get("points"),
        })

    return results
