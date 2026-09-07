"""Builds citation objects strictly from actual retrieved tool outputs.

NEVER invents URLs. Every citation here is derived from data that was
actually returned by a tool call during the current run.
"""
from typing import List, Dict, Any


def _api_citation_label(api_results: Dict[str, Any]) -> str:
    if "city" in api_results:
        return f"Open-Meteo ({api_results.get('city', 'weather data')})"
    if "country" in api_results:
        return f"countries.dev ({api_results.get('country', 'country data')})"
    if "topic" in api_results:
        return f"Wikipedia ({api_results.get('topic', 'article')})"
    return "Public API"


def build_citations(social_results: List[Dict[str, Any]], api_results: Dict[str, Any]) -> List[Dict[str, str]]:
    citations: List[Dict[str, str]] = []

    if api_results and api_results.get("source_url"):
        citations.append({
            "label": _api_citation_label(api_results),
            "url": api_results["source_url"],
        })

    for r in social_results or []:
        if r.get("url"):
            citations.append({
                "label": f"{r.get('source', 'Hacker News')} — {r.get('title', '')[:70]}",
                "url": r["url"],
            })

    return citations
