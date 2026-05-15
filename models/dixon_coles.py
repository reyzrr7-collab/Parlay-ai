import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson


def dixon_coles_tau(lambda_x, mu_y, rho):
    """Koreksi Dixon-Coles untuk skor rendah."""
    if lambda_x == 0 and mu_y == 0:
        return 1 - lambda_x * mu_y * rho
    elif lambda_x == 0 and mu_y == 1:
        return 1 + lambda_x * rho
    elif lambda_x == 1 and mu_y == 0:
        return 1 + mu_y * rho
    elif lambda_x == 1 and mu_y == 1:
        return 1 - rho
    return 1.0


def match_score_probability(home_goals, away_goals, lambda_home, lambda_away, rho=0.0):
    """Probabilitas skor spesifik."""
    tau = dixon_coles_tau(home_goals, away_goals, rho)
    return tau * poisson.pmf(home_goals, lambda_home) * poisson.pmf(away_goals, lambda_away)


def calculate_match_outcome_probs(lambda_home: float, lambda_away: float, rho: float = -0.1, max_goals: int = 10):
    """Hitung probabilitas home win, draw, away win."""
    home_win = draw = away_win = 0.0
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            prob = match_score_probability(h, a, lambda_home, lambda_away, rho)
            if h > a:
                home_win += prob
            elif h == a:
                draw += prob
            else:
                away_win += prob
    total = home_win + draw + away_win
    return {
        "home_win": round(home_win / total, 4),
        "draw": round(draw / total, 4),
        "away_win": round(away_win / total, 4)
    }


def estimate_attack_defense(home_xg: float, away_xg: float, home_xga: float, away_xga: float,
                              league_avg_goals: float = 1.35):
    """Estimasi attack/defense strength dari xG data."""
    home_attack = home_xg / league_avg_goals
    home_defense = home_xga / league_avg_goals
    away_attack = away_xg / league_avg_goals
    away_defense = away_xga / league_avg_goals

    home_advantage = 1.2
    lambda_home = home_attack * away_defense * home_advantage * league_avg_goals
    lambda_away = away_attack * home_defense * league_avg_goals

    return max(lambda_home, 0.1), max(lambda_away, 0.1)


def dixon_coles_predict(match_data: dict) -> dict:
    """Prediksi utama dengan model Dixon-Coles."""
    home_xg = match_data.get("home_xg_avg", 1.2)
    away_xg = match_data.get("away_xg_avg", 1.0)
    home_xga = match_data.get("home_xga_avg", 1.2)
    away_xga = match_data.get("away_xga_avg", 1.0)

    lambda_home, lambda_away = estimate_attack_defense(home_xg, away_xg, home_xga, away_xga)
    probs = calculate_match_outcome_probs(lambda_home, lambda_away)
    probs["lambda_home"] = round(lambda_home, 3)
    probs["lambda_away"] = round(lambda_away, 3)
    probs["model"] = "dixon_coles"
    return probs
