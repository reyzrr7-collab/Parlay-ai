import requests
import os
from dotenv import load_dotenv

load_dotenv()

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_BASE_URL = "https://api.the-odds-api.com/v4"


def get_match_odds(sport="soccer_epl", regions="eu", markets="h2h"):
    """Ambil odds real-time dari OddsAPI."""
    url = f"{ODDS_BASE_URL}/sports/{sport}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "decimal",
        "bookmakers": "pinnacle"
    }
    try:
        res = requests.get(url, params=params, timeout=10)
        data = res.json()
        results = []
        for event in data:
            home_team = event["home_team"]
            away_team = event["away_team"]
            odds = {"home": None, "draw": None, "away": None}
            for bookmaker in event.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    if market["key"] == "h2h":
                        for outcome in market["outcomes"]:
                            if outcome["name"] == home_team:
                                odds["home"] = outcome["price"]
                            elif outcome["name"] == away_team:
                                odds["away"] = outcome["price"]
                            else:
                                odds["draw"] = outcome["price"]
            results.append({
                "home_team": home_team,
                "away_team": away_team,
                "commence_time": event["commence_time"],
                "odds": odds
            })
        return results
    except Exception as e:
        print(f"OddsAPI error: {e}")
        return []


def find_odds_for_match(home_team: str, away_team: str, sport="soccer_epl"):
    """Cari odds untuk pertandingan spesifik."""
    all_odds = get_match_odds(sport=sport)
    for match in all_odds:
        if home_team.lower() in match["home_team"].lower() and \
           away_team.lower() in match["away_team"].lower():
            return match["odds"]
    return {"home": None, "draw": None, "away": None}


def odds_to_implied_prob(odds: float) -> float:
    """Konversi odds decimal ke implied probability."""
    if not odds or odds <= 0:
        return 0.0
    return round(1 / odds, 4)


def remove_vig(home_odds, draw_odds, away_odds):
    """Hapus vig dari odds Pinnacle untuk dapat true probability."""
    if not all([home_odds, draw_odds, away_odds]):
        return None, None, None
    h = odds_to_implied_prob(home_odds)
    d = odds_to_implied_prob(draw_odds)
    a = odds_to_implied_prob(away_odds)
    total = h + d + a
    return round(h / total, 4), round(d / total, 4), round(a / total, 4)
