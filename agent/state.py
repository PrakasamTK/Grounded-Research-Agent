"""Shared state object passed between LangGraph nodes."""
from typing import TypedDict, List, Dict, Any, Optional


class AgentState(TypedDict, total=False):
    question: str
    route: str                     # WEATHER | GEO | SOCIAL | BOTH | OFF_TOPIC | UNKNOWN
    tools_used: List[str]
    social_results: List[Dict[str, Any]]
    api_results: Dict[str, Any]
    sources: List[Dict[str, str]]  # [{"label": ..., "url": ...}]
    grounding_status: str          # GROUNDED | PARTIAL | INSUFFICIENT
    guardrail_status: str          # OK | BLOCKED_INJECTION | BLOCKED_UNSAFE | BLOCKED_OFF_TOPIC
    answer: str
    errors: List[str]
    trace: List[str]               # human-readable execution steps for UI
