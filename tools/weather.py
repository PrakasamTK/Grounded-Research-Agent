"""Open-Meteo REST API tool. Real HTTP calls, no mocked data."""
import requests
from typing import Dict, Any

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    80: "Rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Thunderstorm with heavy hail",
}


def geocode_city(city: str) -> Dict[str, Any]:
    resp = requests.get(GEOCODE_URL, params={"name": city, "count": 1}, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results")
    if not results:
        raise ValueError(f"Could not geocode city: {city}")
    top = results[0]
    return {
        "name": top["name"],
        "country": top.get("country", ""),
        "latitude": top["latitude"],
        "longitude": top["longitude"],
    }


def get_weather(city: str) -> Dict[str, Any]:
    """Geocode a city then fetch current weather from Open-Meteo.

    Returns a structured dict with an exact source_url, or {"error": ...}
    on failure. Never fabricates data.
    """
    try:
        loc = geocode_city(city)
    except Exception as e:
        return {"error": f"Geocoding failed for '{city}': {e}"}

    params = {
        "latitude": loc["latitude"],
        "longitude": loc["longitude"],
        "current_weather": "true",
    }
    try:
        resp = requests.get(FORECAST_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return {"error": f"Open-Meteo request failed: {e}"}

    current = data.get("current_weather")
    if not current:
        return {"error": "Open-Meteo returned no current weather data."}

    prepped = requests.Request("GET", FORECAST_URL, params=params).prepare()

    return {
        "city": loc["name"],
        "country": loc["country"],
        "latitude": loc["latitude"],
        "longitude": loc["longitude"],
        "temperature_c": current.get("temperature"),
        "windspeed_kmh": current.get("windspeed"),
        "winddirection_deg": current.get("winddirection"),
        "weather_code": current.get("weathercode"),
        "condition": WEATHER_CODES.get(current.get("weathercode"), "Unknown"),
        "observed_time": current.get("time"),
        "source_url": prepped.url,
    }
