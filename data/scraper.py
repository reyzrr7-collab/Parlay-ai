"""
data/scraper.py
───────────────
Scraping data gratis:
- Understat  → xG per shot level
- FBref      → advanced stats (PPDA, progressive passes, xA)
"""

import re
import json
import logging
import asyncio
import requests
import pandas as pd
from typing import Optional, Dict
from bs4 import BeautifulSoup

log = logging.getLogger("scraper")
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


# ── Understat — xG data ──────────────────────────────────────────────────────

def get_understat_team_stats(team_name: str, league: str = "EPL",
                              season: str = "2024") -> dict:
    """
    Ambil statistik xG tim dari Understat.
    league: EPL | La_liga | Bundesliga | Serie_A | Ligue_1 | RFPL
    """
    url = f"https://understat.com/league/{league}/{season}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()

        # Extract JSON embedded di halaman
        pattern = r"teamsData\s*=\s*JSON\.parse\('(.+?)'\)"
        match = re.search(pattern, resp.text)
        if not match:
            return {}

        raw = match.group(1).encode("utf-8").decode("unicode_escape")
        teams_data = json.loads(raw)

        # Cari tim yang cocok
        for team_id, data in teams_data.items():
            if team_name.lower() in data["title"].lower():
                history = data.get("history", [])
                if not history:
                    return {}

                xg_total    = sum(h.get("xG", 0) for h in history)
                xga_total   = sum(h.get("xGA", 0) for h in history)
                n = len(history) or 1

                return {
                    "team":     data["title"],
                    "xg_avg":   round(xg_total / n, 3),
                    "xga_avg":  round(xga_total / n, 3),
                    "npxg_avg": round(sum(h.get("npxG", 0) for h in history) / n, 3),
                    "deep_avg": round(sum(h.get("deep", 0) for h in history) / n, 2),
                    "ppda_avg": round(sum(h.get("ppda_coef", 1) for h in history) / n, 3),
                    "matches":  n,
                }
        return {}
    except Exception as e:
        log.error("Understat scraping error: %s", e)
        return {}


def get_understat_match_xg(fixture_url: str) -> dict:
    """Ambil xG per shot untuk satu pertandingan spesifik."""
    try:
        resp = requests.get(fixture_url, headers=HEADERS, timeout=15)
        resp.raise_for_status()

        pattern = r"shotsData\s*=\s*JSON\.parse\('(.+?)'\)"
        match = re.search(pattern, resp.text)
        if not match:
            return {}

        raw = match.group(1).encode("utf-8").decode("unicode_escape")
        shots = json.loads(raw)

        home_xg = sum(float(s.get("xG", 0)) for s in shots.get("h", []))
        away_xg = sum(float(s.get("xG", 0)) for s in shots.get("a", []))

        return {
            "home_xg":    round(home_xg, 3),
            "away_xg":    round(away_xg, 3),
            "home_shots": len(shots.get("h", [])),
            "away_shots": len(shots.get("a", [])),
        }
    except Exception as e:
        log.error("Understat match xG error: %s", e)
        return {}


# ── FBref — Advanced stats ────────────────────────────────────────────────────

FBREF_LEAGUE_URLS = {
    "EPL":        "https://fbref.com/en/comps/9/stats/Premier-League-Stats",
    "La Liga":    "https://fbref.com/en/comps/12/stats/La-Liga-Stats",
    "Bundesliga": "https://fbref.com/en/comps/20/stats/Bundesliga-Stats",
    "Serie A":    "https://fbref.com/en/comps/11/stats/Serie-A-Stats",
    "Ligue 1":    "https://fbref.com/en/comps/13/stats/Ligue-1-Stats",
}


def get_fbref_team_stats(team_name: str, league: str = "EPL") -> dict:
    """
    Scraping advanced stats dari FBref.
    Return: PPDA, progressive carries, xG, xA, dll.
    """
    url = FBREF_LEAGUE_URLS.get(league, FBREF_LEAGUE_URLS["EPL"])
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()

        # pandas bisa langsung baca tabel HTML
        tables = pd.read_html(resp.text)
        if not tables:
            return {}

        # Cari tabel yang mengandung nama tim
        for table in tables:
            if "Squad" in table.columns:
                row = table[table["Squad"].str.lower().str.contains(
                    team_name.lower(), na=False
                )]
                if not row.empty:
                    r = row.iloc[0]
                    return {
                        "team":       str(r.get("Squad", "")),
                        "matches":    int(r.get("MP", 0) or 0),
                        "xg":         float(r.get("xG", 0) or 0),
                        "xga":        float(r.get("xGA", 0) or 0),
                        "xg_diff":    float(r.get("xGD", 0) or 0),
                        "possession": float(r.get("Poss", 0) or 0),
                    }
        return {}
    except Exception as e:
        log.error("FBref scraping error: %s", e)
        return {}


# ── Google News — Berita terkini ──────────────────────────────────────────────

def get_team_news(team_name: str, max_results: int = 5) -> list:
    """Cari berita terkini tentang tim (injury, lineup, kondisi)."""
    try:
        url = f"https://news.google.com/rss/search?q={team_name}+football&hl=en"
        resp = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(resp.text, "xml")
        items = soup.find_all("item")[:max_results]
        return [
            {
                "title": item.find("title").text if item.find("title") else "",
                "date":  item.find("pubDate").text[:16] if item.find("pubDate") else "",
            }
            for item in items
        ]
    except Exception as e:
        log.warning("News scraping error: %s", e)
        return []
