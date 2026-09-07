"""LangGraph orchestration for the Grounded Research Agent.

Graph:

START -> classify -> (route)
  OFF_TOPIC -> decline
  SOCIAL    -> hackernews_search -> validate -> synthesize
  WEATHER   -> weather_lookup -> validate -> synthesize
  GEO       -> geo_lookup -> validate -> synthesize
  GENERAL   -> wikipedia_lookup -> validate -> synthesize
  BOTH      -> hackernews_search -> weather_lookup -> validate -> synthesize
  UNKNOWN   -> validate (no sources) -> synthesize (refusal)
"""
import os
from typing import Dict, Any

from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.router import classify_question
from agent.guardrails import (
    check_input_injection,
    sanitize_social_results,
    compute_grounding_status,
)
from agent.prompts import SYSTEM_PROMPT, build_user_prompt, DECLINE_OFF_TOPIC, DECLINE_INSUFFICIENT
from tools.hackernews import search_hackernews
from tools.weather import get_weather
from tools.geo import get_country_info
from tools.wikipedia import get_wikipedia_summary
from utils.citations import build_citations


def _init_state(state: AgentState) -> AgentState:
    state.setdefault("tools_used", [])
    state.setdefault("social_results", [])
    state.setdefault("api_results", {})
    state.setdefault("sources", [])
    state.setdefault("errors", [])
    state.setdefault("trace", [])
    return state


def classify_node(state: AgentState) -> AgentState:
    state = _init_state(state)
    question = state["question"]

    if check_input_injection(question):
        state["route"] = "OFF_TOPIC"
        state["guardrail_status"] = "BLOCKED_INJECTION"
        state["trace"].append("Guardrail: prompt-injection / jailbreak pattern detected in question -> blocked")
        return state

    route = classify_question(question)
    state["route"] = route
    state["guardrail_status"] = "OK"
    state["trace"].append(f"Question classified -> {route}")
    return state


def hackernews_node(state: AgentState) -> AgentState:
    question = state["question"]
    state["trace"].append("Calling Hacker News search tool (Algolia HN API)")
    try:
        raw = search_hackernews(question, limit=5, comments_per_story=3)
        clean, flagged = sanitize_social_results(raw)
        state["social_results"] = clean
        state["tools_used"].append("Hacker News (Algolia API)")
        state["trace"].append(f"Hacker News returned {len(raw)} result(s); {len(clean)} passed safety filter"
                               + (f"; {flagged} flagged for embedded instruction-like text (neutralized)" if flagged else ""))
        if not raw:
            state["errors"].append("Hacker News search returned no results for this query.")
    except Exception as e:
        state["errors"].append(f"Hacker News search failed: {e}")
        state["social_results"] = []
        state["trace"].append(f"Hacker News search failed: {e}")
    return state


def weather_node(state: AgentState) -> AgentState:
    question = state["question"]
    city = _extract_city(question)
    state["trace"].append(f"Calling Open-Meteo weather tool for city='{city}'")
    result = get_weather(city)
    if result.get("error"):
        state["errors"].append(result["error"])
        state["trace"].append(f"Open-Meteo error: {result['error']}")
    else:
        state["api_results"] = result
        state["tools_used"].append("Open-Meteo")
        state["trace"].append(f"Open-Meteo returned current weather for {result.get('city')}")
    return state


def geo_node(state: AgentState) -> AgentState:
    question = state["question"]
    state["trace"].append("Calling Country Data API tool (countries.dev)")
    result = get_country_info(question)
    if result.get("error"):
        state["errors"].append(result["error"])
        state["trace"].append(f"Country data error: {result['error']}")
    else:
        state["api_results"] = result
        state["tools_used"].append("Country Data API (countries.dev)")
        state["trace"].append(f"Country data API returned data for {result.get('country')}")
    return state


def wikipedia_node(state: AgentState) -> AgentState:
    question = state["question"]
    state["trace"].append("Calling Wikipedia summary tool")
    result = get_wikipedia_summary(question)
    if result.get("error"):
        state["errors"].append(result["error"])
        state["trace"].append(f"Wikipedia error: {result['error']}")
    else:
        state["api_results"] = result
        state["tools_used"].append("Wikipedia (REST API)")
        state["trace"].append(f"Wikipedia returned a summary for '{result.get('topic')}'")
    return state


def _extract_city(question: str) -> str:
    import re
    match = re.search(r"in ([A-Za-z\s]+?)(?:\?|$| right now| today| now)", question, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return question


def decline_node(state: AgentState) -> AgentState:
    state["answer"] = DECLINE_OFF_TOPIC
    state["grounding_status"] = "INSUFFICIENT"
    if state.get("guardrail_status") == "OK":
        state["guardrail_status"] = "BLOCKED_OFF_TOPIC"
    state["sources"] = []
    state["trace"].append("Off-topic / guardrail decline -> refusal returned")
    return state


def validate_node(state: AgentState) -> AgentState:
    grounding = compute_grounding_status(state.get("social_results", []), state.get("api_results", {}))
    state["grounding_status"] = grounding
    state["sources"] = build_citations(state.get("social_results", []), state.get("api_results", {}))
    state["trace"].append(f"Grounding validation -> {grounding} ({len(state['sources'])} source(s))")
    return state


def synthesize_node(state: AgentState) -> AgentState:
    if state["grounding_status"] == "INSUFFICIENT":
        state["answer"] = DECLINE_INSUFFICIENT
        state["trace"].append("No sufficient sources -> refusal returned, LLM not used to avoid hallucination")
        return state

    from groq import Groq

    api_key = os.environ.get("GROQ_API_KEY")
    model = os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")
    if not api_key:
        state["answer"] = "Configuration error: GROQ_API_KEY is not set. Cannot synthesize an answer."
        state["errors"].append("Missing GROQ_API_KEY")
        return state

    client = Groq(api_key=api_key)
    user_prompt = build_user_prompt(state["question"], state.get("social_results", []), state.get("api_results", {}))

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=600,
        )
        answer = resp.choices[0].message.content
        state["answer"] = answer
        state["trace"].append(f"LLM ({model} via Groq) synthesized grounded answer")
    except Exception as e:
        state["answer"] = f"LLM synthesis failed: {e}"
        state["errors"].append(str(e))
        state["trace"].append(f"LLM call failed: {e}")

    return state


def route_after_classify(state: AgentState) -> str:
    route = state["route"]
    if route == "OFF_TOPIC":
        return "decline"
    if route == "SOCIAL":
        return "hackernews"
    if route == "WEATHER":
        return "weather"
    if route == "GEO":
        return "geo"
    if route == "GENERAL":
        return "wikipedia"
    if route == "BOTH":
        return "hackernews_then_weather"
    return "validate"  # UNKNOWN -> straight to validation -> will be INSUFFICIENT


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("classify", classify_node)
    graph.add_node("hackernews", hackernews_node)
    graph.add_node("weather", weather_node)
    graph.add_node("geo", geo_node)
    graph.add_node("wikipedia", wikipedia_node)
    graph.add_node("decline", decline_node)
    graph.add_node("validate", validate_node)
    graph.add_node("synthesize", synthesize_node)

    graph.set_entry_point("classify")

    graph.add_conditional_edges(
        "classify",
        route_after_classify,
        {
            "decline": "decline",
            "hackernews": "hackernews",
            "weather": "weather",
            "geo": "geo",
            "wikipedia": "wikipedia",
            "hackernews_then_weather": "hackernews",
            "validate": "validate",
        },
    )

    # BOTH route: hackernews -> weather -> validate. Plain SOCIAL: hackernews -> validate.
    def after_hackernews(state: AgentState) -> str:
        return "weather" if state["route"] == "BOTH" else "validate"

    graph.add_conditional_edges("hackernews", after_hackernews, {"weather": "weather", "validate": "validate"})
    graph.add_edge("weather", "validate")
    graph.add_edge("geo", "validate")
    graph.add_edge("wikipedia", "validate")
    graph.add_edge("validate", "synthesize")
    graph.add_edge("synthesize", END)
    graph.add_edge("decline", END)

    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_agent(question: str) -> Dict[str, Any]:
    graph = get_graph()
    initial_state: AgentState = {"question": question}
    final_state = graph.invoke(initial_state)
    return final_state
