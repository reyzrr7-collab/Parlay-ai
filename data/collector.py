"""
data/collector.py
─────────────────
Ambil data dari API-Football (RapidAPI):
- Fixtures / jadwal
- Statistik tim
- Head-to-head
- Lineup & injury
- Form terkini
- Home/Away split (terpisah)
"""

import os
import time
import logging
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("collector")

API_KEY = os.getenv("FOOTBALL_API_KEY", "")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS  = {"x-apisports-key": API_KEY}

# Cache sederhana
_cache: Dict[str, Any] = {}
CACHE_TTL = 300  # 5 menit


def _cache_get(key: str) -> Optional[Any]:
    entry = _cache.get(key)
    if entry and (time.time() - entry["ts"]) < CACHE_TTL:
        return entry["data"]
    return None


def _cache_set(key: str, data: Any) -> None:
    _cache[key] = {"data": data, "ts": time.time()}


def _get(endpoint: str, params: dict) -> Optional[dict]:
    cache_key = f"{endpoint}:{str(params)}"
    cached = _cache_get(cache_key)
    if cached:
        return cached

    try:
        resp = requests.get(f"{BASE_URL}/{endpoint}", headers=HEADERS,
                            params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        _cache_set(cache_key, data)
        return data
    except requests.RequestException as e:
        log.error("API-Football error [%s]: %s", endpoint, e)
        return None


def get_fixtures_today(league_id: int = None) -> list:
    """Ambil semua pertandingan hari ini."""
    today = datetime.now().strftime("%Y-%m-%d")
    params = {"date": today, "timezone": "Asia/Jakarta"}
    if league_id:
        params["league"] = league_id
    data = _get("fixtures", params)
    return data.get("response", []) if data else []


def get_fixtures_by_date(date_str: str) -> list:
    """Ambil pertandingan di tanggal tertentu (YYYY-MM-DD)."""
    data = _get("fixtures", {"date": date_str, "timezone": "Asia/Jakarta"})
    return data.get("response", []) if data else []


def get_team_stats(team_id: int, league_id: int, season: int = 2024) -> dict:
    """Statistik lengkap tim dalam satu musim."""
    data = _get("teams/statistics", {
        "team": team_id, "league": league_id, "season": season
    })
    if not data or not data.get("response"):
        return {}
    r = data["response"]

    # Home/Away split terpisah
    fixtures = r.get("fixtures", {})
    goals    = r.get("goals", {})

    home_played = fixtures.get("played", {}).get("home", 0) or 1
    away_played = fixtures.get("played", {}).get("away", 0) or 1

    return {
        "team_id":        team_id,
        "team_name":      r.get("team", {}).get("name", ""),
        "league":         r.get("league", {}).get("name", ""),
        # Keseluruhan
        "wins":           fixtures.get("wins", {}).get("total", 0),
        "draws":          fixtures.get("draws", {}).get("total", 0),
        "losses":         fixtures.get("loses", {}).get("total", 0),
        "goals_scored":   goals.get("for", {}).get("total", {}).get("total", 0),
        "goals_conceded": goals.get("against", {}).get("total", {}).get("total", 0),
        # Home split
        "home_wins":      fixtures.get("wins", {}).get("home", 0),
        "home_draws":     fixtures.get("draws", {}).get("home", 0),
        "home_losses":    fixtures.get("loses", {}).get("home", 0),
        "home_goals_avg": (goals.get("for", {}).get("total", {}).get("home", 0) or 0) / home_played,
        "home_conc_avg":  (goals.get("against", {}).get("total", {}).get("home", 0) or 0) / home_played,
        # Away split
        "away_wins":      fixtures.get("wins", {}).get("away", 0),
        "away_draws":     fixtures.get("draws", {}).get("away", 0),
        "away_losses":    fixtures.get("loses", {}).get("away", 0),
        "away_goals_avg": (goals.get("for", {}).get("total", {}).get("away", 0) or 0) / away_played,
        "away_conc_avg":  (goals.get("against", {}).get("total", {}).get("away", 0) or 0) / away_played,
        # Form
        "form_string":    r.get("form", ""),
        "clean_sheets":   r.get("clean_sheet", {}).get("total", 0),
    }


def get_team_form(team_id: int, last_n: int = 10) -> list:
    """N pertandingan terakhir tim — dipakai untuk time decay."""
    data = _get("fixtures", {
        "team": team_id, "last": last_n, "status": "FT"
    })
    if not data:
        return []
    fixtures = data.get("response", [])
    results = []
    for f in fixtures:
        home = f["teams"]["home"]
        away = f["teams"]["away"]
        goals = f["goals"]
        is_home = home["id"] == team_id
        team_goals = goals["home"] if is_home else goals["away"]
        opp_goals  = goals["away"] if is_home else goals["home"]
        result = "W" if team_goals > opp_goals else ("D" if team_goals == opp_goals else "L")
        results.append({
            "fixture_id": f["fixture"]["id"],
            "date":       f["fixture"]["date"][:10],
            "is_home":    is_home,
            "opponent":   away["name"] if is_home else home["name"],
            "scored":     team_goals,
            "conceded":   opp_goals,
            "result":     result,
        })
    return results


def get_head_to_head(team1_id: int, team2_id: int, last: int = 10) -> dict:
    """Histori head-to-head dua tim."""
    data = _get("fixtures/headtohead", {
        "h2h": f"{team1_id}-{team2_id}", "last": last, "status": "FT"
    })
    if not data:
        return {}
    fixtures = data.get("response", [])
    t1_wins = t2_wins = draws = total_goals = 0
    for f in fixtures:
        home_id    = f["teams"]["home"]["id"]
        home_goals = f["goals"]["home"] or 0
        away_goals = f["goals"]["away"] or 0
        total_goals += home_goals + away_goals
        if home_goals > away_goals:
            if home_id == team1_id: t1_wins += 1
            else: t2_wins += 1
        elif home_goals == away_goals:
            draws += 1
        else:
            if home_id == team1_id: t2_wins += 1
            else: t1_wins += 1

    n = len(fixtures) or 1
    return {
        "total_matches": len(fixtures),
        "team1_wins":    t1_wins,
        "team2_wins":    t2_wins,
        "draws":         draws,
        "avg_goals":     round(total_goals / n, 2),
        "fixtures":      fixtures[:5],  # 5 pertemuan terakhir
    }


def get_injuries(team_id: int) -> list:
    """Daftar pemain cedera & suspensi."""
    data = _get("injuries", {"team": team_id})
    if not data:
        return []
    return [
        {
            "player": p["player"]["name"],
            "type":   p["player"]["type"],
            "reason": p["player"]["reason"],
        }
        for p in data.get("response", [])
    ]


def get_fixture_odds(fixture_id: int) -> dict:
    """Odds 1X2 dari API-Football."""
    data = _get("odds", {"fixture": fixture_id, "bookmaker": 8})
    if not data or not data.get("response"):
        return {}
    try:
        bets = data["response"][0]["bookmakers"][0]["bets"]
        for bet in bets:
            if bet["name"] == "Match Winner":
                values = {v["value"]: float(v["odd"]) for v in bet["values"]}
                return {
                    "home": values.get("Home", 0),
                    "draw": values.get("Draw", 0),
                    "away": values.get("Away", 0),
                }
    except (IndexError, KeyError):
        pass
    return {}


def get_fatigue_score(team_id: int, days_back: int = 21) -> float:
    """
    Hitung fatigue score berdasarkan kepadatan jadwal.
    Semakin banyak main dalam 3 minggu terakhir → semakin tinggi fatigue.
    """
    data = _get("fixtures", {"team": team_id, "last": 8, "status": "FT"})
    if not data:
        return 0.0
    fixtures = data.get("response", [])
    cutoff = datetime.now() - timedelta(days=days_back)
    recent = []
    for f in fixtures:
        try:
            match_date = datetime.strptime(f["fixture"]["date"][:10], "%Y-%m-%d")
            if match_date >= cutoff:
                recent.append(match_date)
        except:
            pass

    count = len(recent)
    # 0-1 match = 0.0, 2 = 0.2, 3 = 0.4, 4 = 0.6, 5+ = 0.8-1.0
    return min(1.0, count * 0.2)
