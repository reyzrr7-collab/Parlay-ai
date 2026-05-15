from langchain.tools import tool
from data.collector import collect_match_data, get_today_fixtures
from data.scraper import get_understat_team_xg, get_google_news
from data.odds import find_odds_for_match, remove_vig
from models.ensemble import ensemble_predict
from data.preprocessor import build_feature_vector
import requests
import os


WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")


@tool
def get_fixtures_today(league_id: int = 39) -> str:
    """Ambil semua pertandingan hari ini dari API-Football."""
    fixtures = get_today_fixtures(league_id=league_id)
    if not fixtures:
        return "Tidak ada pertandingan hari ini."
    result = []
    for f in fixtures:
        result.append(
            f"{f['teams']['home']['name']} vs {f['teams']['away']['name']} "
            f"- {f['fixture']['date']} - League: {f['league']['name']}"
        )
    return "\n".join(result)


@tool
def analyze_match(home_team: str, away_team: str) -> str:
    """Analisis pertandingan lengkap: form, H2H, xG, odds."""
    fixtures = get_today_fixtures()
    fixture = None
    for f in fixtures:
        if home_team.lower() in f["teams"]["home"]["name"].lower() and \
           away_team.lower() in f["teams"]["away"]["name"].lower():
            fixture = f
            break

    if not fixture:
        return f"Pertandingan {home_team} vs {away_team} tidak ditemukan."

    match_data = collect_match_data(fixture)

    # Tambah xG dari Understat
    home_xg = get_understat_team_xg(home_team)
    away_xg = get_understat_team_xg(away_team)
    match_data["home_xg_avg"] = home_xg["xg_avg"]
    match_data["away_xg_avg"] = away_xg["xg_avg"]
    match_data["home_xga_avg"] = home_xg["xga_avg"]
    match_data["away_xga_avg"] = away_xg["xga_avg"]

    features = build_feature_vector(match_data)
    match_data.update(features)
    prediction = ensemble_predict(match_data)

    return (
        f"=== {home_team} vs {away_team} ===\n"
        f"Home Win: {prediction['home_win']:.1%}\n"
        f"Draw: {prediction['draw']:.1%}\n"
        f"Away Win: {prediction['away_win']:.1%}\n"
        f"Prediksi: {prediction['predicted_outcome']}\n"
        f"Confidence: {prediction['confidence']:.1%}"
    )


@tool
def check_value_bet(home_team: str, away_team: str, sport: str = "soccer_epl") -> str:
    """Cek apakah ada value bet vs odds Pinnacle."""
    fixtures = get_today_fixtures()
    fixture = None
    for f in fixtures:
        if home_team.lower() in f["teams"]["home"]["name"].lower():
            fixture = f
            break

    if not fixture:
        return "Fixture tidak ditemukan."

    match_data = collect_match_data(fixture)
    home_xg = get_understat_team_xg(home_team)
    away_xg = get_understat_team_xg(away_team)
    match_data["home_xg_avg"] = home_xg["xg_avg"]
    match_data["away_xg_avg"] = away_xg["xg_avg"]
    match_data.update(build_feature_vector(match_data))

    prediction = ensemble_predict(match_data)
    odds = find_odds_for_match(home_team, away_team, sport)
    true_h, true_d, true_a = remove_vig(odds["home"], odds["draw"], odds["away"])

    if not true_h:
        return "Odds tidak tersedia dari Pinnacle."

    edge_home = prediction["home_win"] - true_h
    edge_draw = prediction["draw"] - true_d
    edge_away = prediction["away_win"] - true_a

    MIN_EDGE = float(os.getenv("MIN_EDGE", 0.05))
    value_bets = []
    if edge_home > MIN_EDGE:
        value_bets.append(f"Home Win edge: {edge_home:.1%} (odds: {odds['home']})")
    if edge_draw > MIN_EDGE:
        value_bets.append(f"Draw edge: {edge_draw:.1%} (odds: {odds['draw']})")
    if edge_away > MIN_EDGE:
        value_bets.append(f"Away Win edge: {edge_away:.1%} (odds: {odds['away']})")

    if value_bets:
        return "VALUE BET DITEMUKAN:\n" + "\n".join(value_bets)
    return f"Tidak ada value bet. Edge: Home={edge_home:.1%}, Draw={edge_draw:.1%}, Away={edge_away:.1%}"


@tool
def get_weather(city: str) -> str:
    """Ambil cuaca untuk kota lokasi pertandingan."""
    url = f"https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": WEATHER_API_KEY, "units": "metric"}
    try:
        res = requests.get(url, params=params, timeout=5)
        data = res.json()
        weather = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        wind = data["wind"]["speed"]
        return f"Cuaca {city}: {weather}, {temp}°C, angin {wind} m/s"
    except:
        return f"Data cuaca tidak tersedia untuk {city}."


@tool
def get_latest_news(home_team: str, away_team: str) -> str:
    """Ambil berita terkini tentang pertandingan."""
    news = get_google_news(home_team, away_team)
    if not news:
        return "Tidak ada berita terkini."
    return "\n".join([f"- {n['title']}" for n in news])


ALL_TOOLS = [get_fixtures_today, analyze_match, check_value_bet, get_weather, get_latest_news]
