"""Guardrail checks applied before and after retrieval/synthesis."""
from agent.router import detect_injection
from utils.safety import is_unsafe, filter_unsafe_results


def check_input_injection(question: str) -> bool:
    """True if the user question itself looks like a prompt-injection/jailbreak attempt."""
    return detect_injection(question)


def sanitize_social_results(results: list) -> tuple:
    """Filter unsafe content and flag any retrieved post/comment containing
    injection-like instruction text (so it's neutralized as DATA, not followed).
    Returns (clean_results, flagged_count).
    """
    safe_results = filter_unsafe_results(results)

    flagged = 0
    for r in safe_results:
        blob = " ".join([str(r.get("title", "")), str(r.get("body", ""))] +
                         [str(c) for c in r.get("comments", [])])
        if detect_injection(blob):
            flagged += 1
            r["_injection_flagged"] = True

    return safe_results, flagged


def compute_grounding_status(social_results: list, api_results: dict) -> str:
    has_social = bool(social_results)
    has_api = bool(api_results) and not api_results.get("error")

    if has_social or has_api:
        if has_social and len(social_results) < 2 and not has_api:
            return "PARTIAL"
        return "GROUNDED"
    return "INSUFFICIENT"
