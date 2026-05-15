import numpy as np


def form_to_points(form: list) -> float:
    """Hitung rata-rata poin dari form terakhir."""
    points = {"W": 3, "D": 1, "L": 0}
    if not form:
        return 0.0
    return round(sum(points.get(m["result"], 0) for m in form) / len(form), 3)


def form_goal_stats(form: list) -> dict:
    """Hitung rata-rata gol dari form."""
    if not form:
        return {"scored_avg": 0, "conceded_avg": 0}
    scored = [m.get("scored", 0) or 0 for m in form]
    conceded = [m.get("conceded", 0) or 0 for m in form]
    return {
        "scored_avg": round(np.mean(scored), 3),
        "conceded_avg": round(np.mean(conceded), 3)
    }


def h2h_stats(h2h: list, home_team: str) -> dict:
    """Analisis H2H terakhir."""
    if not h2h:
        return {"h2h_win_rate": 0.33, "h2h_draw_rate": 0.33}
    wins, draws, losses = 0, 0, 0
    for m in h2h[:10]:
        teams = m.get("teams", {})
        home_winner = teams.get("home", {}).get("winner")
        away_winner = teams.get("away", {}).get("winner")
        is_home = teams.get("home", {}).get("name", "").lower() == home_team.lower()
        if is_home:
            if home_winner:
                wins += 1
            elif away_winner:
                losses += 1
            else:
                draws += 1
        else:
            if away_winner:
                wins += 1
            elif home_winner:
                losses += 1
            else:
                draws += 1
    total = wins + draws + losses or 1
    return {
        "h2h_win_rate": round(wins / total, 3),
        "h2h_draw_rate": round(draws / total, 3)
    }


def build_feature_vector(match_data: dict) -> dict:
    """Build feature vector lengkap untuk model ML."""
    home_form = match_data.get("home_form", [])
    away_form = match_data.get("away_form", [])
    h2h = match_data.get("h2h", [])
    home_team = match_data.get("home_team", "")

    home_pts = form_to_points(home_form)
    away_pts = form_to_points(away_form)
    home_goals = form_goal_stats(home_form)
    away_goals = form_goal_stats(away_form)
    h2h_data = h2h_stats(h2h, home_team)

    features = {
        "home_form_pts": home_pts,
        "away_form_pts": away_pts,
        "home_scored_avg": home_goals["scored_avg"],
        "home_conceded_avg": home_goals["conceded_avg"],
        "away_scored_avg": away_goals["scored_avg"],
        "away_conceded_avg": away_goals["conceded_avg"],
        "h2h_home_win_rate": h2h_data["h2h_win_rate"],
        "h2h_draw_rate": h2h_data["h2h_draw_rate"],
        "home_xg_avg": match_data.get("home_xg_avg", 1.2),
        "away_xg_avg": match_data.get("away_xg_avg", 1.0),
        "home_xga_avg": match_data.get("home_xga_avg", 1.2),
        "away_xga_avg": match_data.get("away_xga_avg", 1.0),
    }
    return features


def features_to_array(features: dict) -> np.ndarray:
    """Konversi feature dict ke numpy array."""
    keys = [
        "home_form_pts", "away_form_pts",
        "home_scored_avg", "home_conceded_avg",
        "away_scored_avg", "away_conceded_avg",
        "h2h_home_win_rate", "h2h_draw_rate",
        "home_xg_avg", "away_xg_avg",
        "home_xga_avg", "away_xga_avg"
    ]
    return np.array([features.get(k, 0.0) for k in keys])
