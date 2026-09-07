"""Basic content-safety filtering for retrieved (untrusted) text."""
import re

UNSAFE_PATTERNS = [
    r"\bkill yourself\b",
    r"\bnigger\b", r"\bfaggot\b", r"\bretard(ed)?\b",
    r"\brape\b",
    r"\bchild porn\b", r"\bcp\b",
    r"\bsuicide method\b",
    r"\bhow to make a bomb\b",
]

_compiled = [re.compile(p, re.IGNORECASE) for p in UNSAFE_PATTERNS]


def is_unsafe(text: str) -> bool:
    if not text:
        return False
    return any(p.search(text) for p in _compiled)


def filter_unsafe_results(results: list) -> list:
    """Drop social-discussion results whose title/body/comments trip the unsafe filter."""
    safe = []
    for r in results:
        blob = " ".join([
            str(r.get("title", "")),
            str(r.get("body", "")),
            " ".join(str(c) for c in r.get("comments", [])),
        ])
        if not is_unsafe(blob):
            safe.append(r)
    return safe
