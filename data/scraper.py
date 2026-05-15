import requests
from bs4 import BeautifulSoup
import json
import re


UNDERSTAT_URL = "https://understat.com"
FBREF_URL = "https://fbref.com"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}


def get_understat_team_xg(team_name: str, league: str = "EPL", season: str = "2024"):
    """Ambil xG rata-rata tim dari Understat."""
    league_map = {
        "EPL": "EPL", "La Liga": "La_liga", "Bundesliga": "Bundesliga",
        "Serie A": "Serie_A", "Ligue 1": "Ligue_1"
    }
    league_key = league_map.get(league, "EPL")
    url = f"{UNDERSTAT_URL}/league/{league_key}/{season}"

    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(res.text, "lxml")
        scripts = soup.find_all("script")
        for script in scripts:
            if "teamsData" in script.text:
                data_str = re.search(r"JSON\.parse\('(.+?)'\)", script.text)
                if data_str:
                    raw = data_str.group(1).encode().decode("unicode_escape")
                    data = json.loads(raw)
                    for tid, tdata in data.items():
                        if team_name.lower() in tdata["title"].lower():
                            history = tdata.get("history", [])
                            if history:
                                xg_avg = sum(h["xG"] for h in history[-10:]) / min(10, len(history))
                                xga_avg = sum(h["xGA"] for h in history[-10:]) / min(10, len(history))
                                return {"xg_avg": round(xg_avg, 2), "xga_avg": round(xga_avg, 2)}
    except Exception as e:
        print(f"Understat scraping error: {e}")
    return {"xg_avg": 0.0, "xga_avg": 0.0}


def get_fbref_team_stats(team_name: str, league_url: str = None):
    """Ambil statistik lanjutan dari FBref."""
    try:
        if not league_url:
            league_url = f"{FBREF_URL}/en/comps/9/Premier-League-Stats"
        res = requests.get(league_url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(res.text, "lxml")
        table = soup.find("table", {"id": "results2024-202591_home_away"})
        if not table:
            return {}
        rows = table.find("tbody").find_all("tr")
        for row in rows:
            name_cell = row.find("td", {"data-stat": "team"})
            if name_cell and team_name.lower() in name_cell.text.lower():
                stats = {}
                for td in row.find_all("td"):
                    stat = td.get("data-stat")
                    if stat:
                        stats[stat] = td.text.strip()
                return stats
    except Exception as e:
        print(f"FBref scraping error: {e}")
    return {}


def get_google_news(team1: str, team2: str):
    """Ambil berita terkini tentang pertandingan."""
    query = f"{team1} vs {team2} preview".replace(" ", "+")
    url = f"https://news.google.com/rss/search?q={query}&hl=en"
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(res.text, "xml")
        items = soup.find_all("item")[:5]
        news = []
        for item in items:
            news.append({
                "title": item.find("title").text if item.find("title") else "",
                "pub_date": item.find("pubDate").text if item.find("pubDate") else ""
            })
        return news
    except Exception as e:
        print(f"News scraping error: {e}")
    return []
