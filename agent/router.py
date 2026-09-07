"""Question classification / routing logic.

Uses fast keyword heuristics first (cheap, deterministic, no token cost),
falling back to the LLM only when heuristics are ambiguous.
"""
import re

WEATHER_KEYWORDS = [
    "weather", "temperature", "temp", "rain", "raining", "wind", "windy",
    "humidity", "forecast", "sunny", "cloudy", "snow", "climate today",
    "hot", "cold outside", "precipitation",
]

GEO_KEYWORDS = [
    "population", "capital of", "country", "geography", "area of",
    "how big is", "gdp of", "border", "continent",
]

SOCIAL_KEYWORDS = [
    "people think", "opinions", "opinion", "reddit", "discussion",
    "complaints", "complain", "review", "reviews", "sentiment",
    "what do people", "community think", "public opinion", "feel about",
    "thoughts on", "experience with", "pros and cons",
]

OFF_TOPIC_SIGNS = [
    "write me a poem", "write code", "solve this equation", "translate",
    "who won", "history of", "when did", "who is the president",
    "stock price", "medical advice", "diagnose", "legal advice",
]

INJECTION_PATTERNS = [
    r"ignore (all )?previous instructions",
    r"system prompt",
    r"developer message",
    r"reveal your instructions",
    r"act as",
    r"jailbreak",
    r"disregard (all )?(the )?(above|prior)",
    r"you are now",
]


def detect_injection(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(p, lowered) for p in INJECTION_PATTERNS)


def classify_question(question: str) -> str:
    """Return one of WEATHER, GEO, SOCIAL, BOTH, OFF_TOPIC, UNKNOWN."""
    q = question.lower()

    # Guardrail: direct prompt-injection / jailbreak attempts on the agent itself
    if detect_injection(q):
        return "OFF_TOPIC"

    has_weather = any(k in q for k in WEATHER_KEYWORDS)
    has_geo = any(k in q for k in GEO_KEYWORDS)
    has_social = any(k in q for k in SOCIAL_KEYWORDS)
    has_off_topic_sign = any(k in q for k in OFF_TOPIC_SIGNS)

    if has_weather and has_social:
        return "BOTH"
    if has_weather:
        return "WEATHER"
    if has_geo:
        return "GEO"
    if has_social:
        return "SOCIAL"
    if has_off_topic_sign:
        return "OFF_TOPIC"

    # Nothing matched confidently
    return "UNKNOWN"
