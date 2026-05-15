import numpy as np
from scipy.stats import poisson, gamma


def bayesian_predict(match_data: dict, n_samples: int = 5000) -> dict:
    """
    Hierarchical Bayesian model sederhana menggunakan sampling.
    Menggunakan prior Gamma untuk rate gol.
    """
    home_xg = match_data.get("home_xg_avg", 1.2)
    away_xg = match_data.get("away_xg_avg", 1.0)
    home_xga = match_data.get("home_xga_avg", 1.2)
    away_xga = match_data.get("away_xga_avg", 1.0)
    home_form_pts = match_data.get("home_form_pts", 1.5)
    away_form_pts = match_data.get("away_form_pts", 1.5)

    # Prior Gamma berdasarkan xG + form
    home_alpha = max(home_xg * 2 + home_form_pts * 0.1, 0.5)
    home_beta = 2.0
    away_alpha = max(away_xg * 2 + away_form_pts * 0.1, 0.5)
    away_beta = 2.0

    # Home advantage
    home_advantage = 1.15

    home_wins = 0
    draws = 0
    away_wins = 0

    for _ in range(n_samples):
        # Sample lambda dari posterior
        lambda_home = gamma.rvs(home_alpha, scale=1 / home_beta) * home_advantage
        lambda_away = gamma.rvs(away_alpha, scale=1 / away_beta)

        # Sample skor
        home_goals = poisson.rvs(max(lambda_home, 0.1))
        away_goals = poisson.rvs(max(lambda_away, 0.1))

        if home_goals > away_goals:
            home_wins += 1
        elif home_goals == away_goals:
            draws += 1
        else:
            away_wins += 1

    total = n_samples
    return {
        "home_win": round(home_wins / total, 4),
        "draw": round(draws / total, 4),
        "away_win": round(away_wins / total, 4),
        "model": "bayesian"
    }
