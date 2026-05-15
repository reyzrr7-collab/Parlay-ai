"""
data/odds.py
────────────
Ambil odds dari:
- Pinnacle  → sharp market (paling akurat di dunia)
- OddsAPI   → aggregator multi-bookmaker
"""

import os
import time
import logging
import requests
from typing import Optional, Dict
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("odds")

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
ODDS_API_URL = "https://api.the-odds-api.com/v4"

_cache: Dict = {}
CACHE_TTL = 180  # 3 menit untuk odds (lebih sering update)


def _cache_get(key):
    e = _cache.get(key)
    return e["data"] if e and (time.time() - e["ts"]) < CACHE_TTL else None

def _cache_set(key, data):
    _cache[key] = {"data": data, "ts": time.time()}


# ── OddsAPI ───────────────────────────────────────────────────────────────────

SPORT_KEYS = {
    "EPL":        "soccer_epl",
    "La Liga":    "soccer_spain_la_liga",
    "Bundesliga": "soccer_germany_bundesliga",
    "Serie A":    "soccer_italy_serie_a",
    "Ligue 1":    "soccer_france_ligue_one",
    "Champions":  "soccer_uefa_champs_league",
}


def get_odds_api(sport: str = "EPL", markets: str = "h2h") -> list:
    """Ambil odds 1X2 dari OddsAPI."""
    cached = _cache_get(f"oddsapi:{sport}")
    if cached:
        return cached

    sport_key = SPORT_KEYS.get(sport, "soccer_epl")
    try:
        resp = requests.get(
            f"{ODDS_API_URL}/sports/{sport_key}/odds",
            params={
                "apiKey":  ODDS_API_KEY,
                "regions": "eu",
                "markets": markets,
                "oddsFormat": "decimal",
            },
            timeout=15
        )
        resp.raise_for_status()
        data = resp.json()
        _cache_set(f"oddsapi:{sport}", data)
        return data
    except Exception as e:
        log.error("OddsAPI error: %s", e)
        return []


def parse_odds(match_data: dict) -> dict:
    """Parse odds dari satu event OddsAPI."""
    result = {
        "home_team": match_data.get("home_team", ""),
        "away_team": match_data.get("away_team", ""),
        "commence":  match_data.get("commence_time", ""),
        "bookmakers": {}
    }
    for bm in match_data.get("bookmakers", []):
        name = bm["key"]
        for market in bm.get("markets", []):
            if market["key"] == "h2h":
                odds = {o["name"]: o["price"] for o in market["outcomes"]}
                result["bookmakers"][name] = odds

    # Ambil rata-rata odds semua bookmaker
    if result["bookmakers"]:
        home_odds = [v.get(result["home_team"], 0) for v in result["bookmakers"].values() if v.get(result["home_team"])]
        draw_odds = [v.get("Draw", 0) for v in result["bookmakers"].values() if v.get("Draw")]
        away_odds = [v.get(result["away_team"], 0) for v in result["bookmakers"].values() if v.get(result["away_team"])]

        result["avg_odds"] = {
            "home": round(sum(home_odds) / len(home_odds), 3) if home_odds else 0,
            "draw": round(sum(draw_odds) / len(draw_odds), 3) if draw_odds else 0,
            "away": round(sum(away_odds) / len(away_odds), 3) if away_odds else 0,
        }
    return result


# ── Pinnacle ──────────────────────────────────────────────────────────────────

def get_pinnacle_odds(match_id: str) -> dict:
    """
    Ambil odds Pinnacle — sharp market terbaik di dunia.
    Pinnacle membuka API publik untuk semua user.
    """
    cached = _cache_get(f"pinnacle:{match_id}")
    if cached:
        return cached

    try:
        resp = requests.get(
            f"https://guest.api.arcadia.pinnacle.com/0.1/matchups/{match_id}/markets/straight",
            headers={
                "X-API-Key": "CmX2KcMrXuFmNg6YFbmTxE0y9CIMqOiq",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            timeout=15
        )
        resp.raise_for_status()
        data = resp.json()
        _cache_set(f"pinnacle:{match_id}", data)
        return data
    except Exception as e:
        log.warning("Pinnacle API error: %s", e)
        return {}


# ── Konversi Odds → Probabilitas ─────────────────────────────────────────────

def odds_to_prob(odds_home: float, odds_draw: float, odds_away: float) -> dict:
    """
    Konversi odds decimal ke implied probability.
    Hilangkan margin bookmaker (overround).
    """
    if not all([odds_home, odds_draw, odds_away]):
        return {"home": 0.0, "draw": 0.0, "away": 0.0}

    raw_home = 1 / odds_home
    raw_draw = 1 / odds_draw
    raw_away = 1 / odds_away
    total = raw_home + raw_draw + raw_away  # overround

    return {
        "home": round(raw_home / total, 4),
        "draw": round(raw_draw / total, 4),
        "away": round(raw_away / total, 4),
        "overround": round((total - 1) * 100, 2)
    }


def get_best_odds_for_match(home_team: str, away_team: str,
                             sport: str = "EPL") -> dict:
    """Cari odds terbaik untuk pertandingan tertentu dari OddsAPI."""
    all_odds = get_odds_api(sport)
    for event in all_odds:
        h = event.get("home_team", "").lower()
        a = event.get("away_team", "").lower()
        if home_team.lower() in h or away_team.lower() in a:
            return parse_odds(event)
    return {}
