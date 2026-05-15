import requests
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
HEADERS = {
    "X-RapidAPI-Key": RAPIDAPI_KEY,
    "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
}
BASE_URL = "https://api-football-v1.p.rapidapi.com/v3"


def get_today_fixtures(league_id=39, season=2024):
    """Ambil semua pertandingan hari ini."""
    today = datetime.utcnow().strftime("%Y-%m-%d")
    url = f"{BASE_URL}/fixtures"
    params = {"league": league_id, "season": season, "date": today}
    res = requests.get(url, headers=HEADERS, params=params)
    return res.json().get("response", [])


def get_team_form(team_id, last=10):
    """Ambil 10 match terakhir tim."""
    url = f"{BASE_URL}/fixtures"
    params = {"team": team_id, "last": last, "status": "FT"}
    res = requests.get(url, headers=HEADERS, params=params)
    matches = res.json().get("response", [])
    form = []
    for m in matches:
        goals = m["goals"]
        teams = m["teams"]
        is_home = teams["home"]["id"] == team_id
        scored = goals["home"] if is_home else goals["away"]
        conceded = goals["away"] if is_home else goals["home"]
        won = teams["home"]["winner"] if is_home else teams["away"]["winner"]
        form.append({
            "date": m["fixture"]["date"],
            "is_home": is_home,
            "scored": scored,
            "conceded": conceded,
            "result": "W" if won else ("D" if won is None else "L")
        })
    return form


def get_h2h(team1_id, team2_id, last=10):
    """Ambil head-to-head 10 pertemuan terakhir."""
    url = f"{BASE_URL}/fixtures/headtohead"
    params = {"h2h": f"{team1_id}-{team2_id}", "last": last}
    res = requests.get(url, headers=HEADERS, params=params)
    return res.json().get("response", [])


def get_lineup_and_injuries(fixture_id):
    """Ambil lineup dan injury terbaru."""
    lineup_url = f"{BASE_URL}/fixtures/lineups"
    injury_url = f"{BASE_URL}/injuries"

    lineup_res = requests.get(lineup_url, headers=HEADERS, params={"fixture": fixture_id})
    injury_res = requests.get(injury_url, headers=HEADERS, params={"fixture": fixture_id})

    lineups = lineup_res.json().get("response", [])
    injuries = injury_res.json().get("response", [])
    return lineups, injuries


def get_team_statistics(team_id, league_id, season=2024):
    """Ambil statistik tim (home/away record)."""
    url = f"{BASE_URL}/teams/statistics"
    params = {"team": team_id, "league": league_id, "season": season}
    res = requests.get(url, headers=HEADERS, params=params)
    return res.json().get("response", {})


def collect_match_data(fixture):
    """Kumpulkan semua data untuk satu pertandingan."""
    fixture_id = fixture["fixture"]["id"]
    home_id = fixture["teams"]["home"]["id"]
    away_id = fixture["teams"]["away"]["id"]
    league_id = fixture["league"]["id"]

    print(f"Collecting data for fixture {fixture_id}...")

    home_form = get_team_form(home_id)
    away_form = get_team_form(away_id)
    h2h = get_h2h(home_id, away_id)
    lineups, injuries = get_lineup_and_injuries(fixture_id)
    home_stats = get_team_statistics(home_id, league_id)
    away_stats = get_team_statistics(away_id, league_id)

    return {
        "fixture_id": fixture_id,
        "home_team": fixture["teams"]["home"]["name"],
        "away_team": fixture["teams"]["away"]["name"],
        "league": fixture["league"]["name"],
        "match_date": fixture["fixture"]["date"],
        "home_form": home_form,
        "away_form": away_form,
        "h2h": h2h,
        "lineups": lineups,
        "injuries": injuries,
        "home_stats": home_stats,
        "away_stats": away_stats
    }
