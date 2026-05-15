"""
models/dixon_coles.py
─────────────────────
Model Dixon-Coles (1997) dengan time decay.
Lebih akurat dari Poisson biasa untuk skor rendah (0-0, 1-0, 0-1, 1-1).
"""

import math
import logging
import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize
from typing import Dict, List, Optional

log = logging.getLogger("dixon_coles")


# ── Time Decay ────────────────────────────────────────────────────────────────

def time_weight(date_str: str, xi: float = 0.0065) -> float:
    """
    Bobot exponential decay berdasarkan umur data.
    xi=0.0065 adalah nilai standar dari paper Dixon-Coles asli.
    Match 30 hari lalu  → ~0.82
    Match 90 hari lalu  → ~0.56
    Match 180 hari lalu → ~0.31
    """
    from datetime import datetime
    try:
        days = (datetime.now() - datetime.strptime(date_str[:10], "%Y-%m-%d")).days
        return math.exp(-xi * days)
    except:
        return 1.0


# ── Koreksi Tau (korelasi skor rendah) ───────────────────────────────────────

def tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    """
    Faktor koreksi Dixon-Coles untuk skor rendah.
    Mengoreksi ketidakakuratan Poisson di skor 0-0, 1-0, 0-1, 1-1.
    """
    if x == 0 and y == 0:
        return 1.0 - lam * mu * rho
    elif x == 1 and y == 0:
        return 1.0 + mu * rho
    elif x == 0 and y == 1:
        return 1.0 + lam * rho
    elif x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


# ── Log-Likelihood untuk Optimasi ────────────────────────────────────────────

def log_likelihood(params: np.ndarray, matches: list, teams: list,
                   n_teams: int) -> float:
    """
    Fungsi log-likelihood Dixon-Coles dengan time decay.
    Diminimasi oleh scipy.optimize.minimize.
    """
    attack     = params[:n_teams]
    defense    = params[n_teams:2*n_teams]
    home_adv   = params[-2]
    rho        = params[-1]

    total_ll = 0.0
    team_idx = {t: i for i, t in enumerate(teams)}

    for match in matches:
        hi = team_idx.get(match["home_team"])
        ai = team_idx.get(match["away_team"])
        if hi is None or ai is None:
            continue

        lam = math.exp(attack[hi] - defense[ai] + home_adv)
        mu  = math.exp(attack[ai] - defense[hi])

        hg = match["home_goals"]
        ag = match["away_goals"]
        w  = match.get("weight", 1.0)

        tau_val = tau(hg, ag, lam, mu, rho)
        if tau_val <= 0:
            continue

        ll = (math.log(tau_val)
              + hg * math.log(lam) - lam - math.lgamma(hg + 1)
              + ag * math.log(mu)  - mu  - math.lgamma(ag + 1))
        total_ll += w * ll

    return -total_ll  # minimize → negatif


# ── Model Utama ────────────────────────────────────────────────────────────────

class DixonColes:
    """
    Model Dixon-Coles lengkap:
    - Fit dari data historis pertandingan
    - Time decay otomatis
    - Prediksi probabilitas semua skor
    """

    def __init__(self):
        self.params = None
        self.teams  = []
        self.team_idx = {}
        self.fitted = False

    def fit(self, matches: list) -> None:
        """
        Fit model dari list pertandingan.
        matches: [{"home_team", "away_team", "home_goals", "away_goals", "date"}, ...]
        """
        # Tambah time decay weight
        for m in matches:
            m["weight"] = time_weight(m.get("date", "2024-01-01"))

        # Kumpulkan semua tim
        self.teams = sorted(set(
            [m["home_team"] for m in matches] + [m["away_team"] for m in matches]
        ))
        self.team_idx = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)

        # Initial params: attack=0.1, defense=0.1, home_adv=0.3, rho=0.08
        x0 = np.concatenate([
            np.full(n, 0.1),   # attack
            np.full(n, 0.1),   # defense
            [0.3, 0.08]        # home_adv, rho
        ])

        # Constraint: sum attack = 0 (identifiability)
        constraints = [{"type": "eq", "fun": lambda p: sum(p[:n])}]
        bounds = (
            [(-3, 3)] * n +  # attack
            [(-3, 3)] * n +  # defense
            [(-0.5, 1.0)] +  # home_adv
            [(-0.5, 0.5)]    # rho
        )

        result = minimize(
            log_likelihood,
            x0,
            args=(matches, self.teams, n),
            method="L-BFGS-B",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-8}
        )

        self.params = result.x
        self.fitted = True
        log.info("✅ Dixon-Coles fitted — %d tim, %d matches", n, len(matches))

    def predict(self, home_team: str, away_team: str,
                max_goals: int = 7) -> dict:
        """
        Prediksi probabilitas untuk satu pertandingan.
        Return: home_win, draw, away_win, over_25, btts, best_score, top_scores
        """
        if not self.fitted:
            return self._fallback_predict(home_team, away_team)

        n = len(self.teams)
        hi = self.team_idx.get(home_team)
        ai = self.team_idx.get(away_team)

        if hi is None or ai is None:
            log.warning("Tim tidak ditemukan dalam model: %s / %s", home_team, away_team)
            return self._fallback_predict(home_team, away_team)

        attack   = self.params[:n]
        defense  = self.params[n:2*n]
        home_adv = self.params[-2]
        rho      = self.params[-1]

        lam = math.exp(attack[hi] - defense[ai] + home_adv)
        mu  = math.exp(attack[ai] - defense[hi])

        return self._compute_probs(lam, mu, rho, max_goals)

    def predict_with_xg(self, home_xg: float, away_xg: float,
                        rho: float = 0.08, max_goals: int = 7) -> dict:
        """Prediksi langsung dari xG — tanpa perlu fit dari historis."""
        return self._compute_probs(home_xg, away_xg, rho, max_goals)

    def _compute_probs(self, lam: float, mu: float,
                       rho: float, max_goals: int) -> dict:
        """Hitung matrix probabilitas semua skor."""
        prob_matrix = np.zeros((max_goals, max_goals))

        for i in range(max_goals):
            for j in range(max_goals):
                t = tau(i, j, lam, mu, rho)
                prob_matrix[i][j] = t * poisson.pmf(i, lam) * poisson.pmf(j, mu)

        # Normalisasi
        total = prob_matrix.sum()
        if total > 0:
            prob_matrix /= total

        home_win = float(np.sum(np.tril(prob_matrix, -1)))
        draw     = float(np.sum(np.diag(prob_matrix)))
        away_win = float(np.sum(np.triu(prob_matrix, 1)))

        best_idx = np.unravel_index(prob_matrix.argmax(), prob_matrix.shape)

        over_25 = max(0.0, 1.0 - float(prob_matrix[:3, :3].sum()))
        btts    = max(0.0, 1.0 - float(prob_matrix[:, 0].sum())
                              - float(prob_matrix[0, :].sum())
                              + float(prob_matrix[0, 0]))

        top_scores = {
            f"{i}-{j}": round(float(prob_matrix[i][j]) * 100, 1)
            for i in range(5) for j in range(5)
            if prob_matrix[i][j] > 0.02
        }
        top_scores = dict(sorted(top_scores.items(), key=lambda x: x[1], reverse=True)[:6])

        return {
            "home_win":   round(home_win * 100, 1),
            "draw":       round(draw * 100, 1),
            "away_win":   round(away_win * 100, 1),
            "over_25":    round(over_25 * 100, 1),
            "btts":       round(btts * 100, 1),
            "best_score": f"{best_idx[0]}-{best_idx[1]}",
            "xg":         {"home": round(lam, 2), "away": round(mu, 2)},
            "top_scores": top_scores,
        }

    def _fallback_predict(self, home_team: str, away_team: str) -> dict:
        """Fallback jika tim tidak ada di model — pakai Poisson sederhana."""
        lam, mu = 1.4, 1.1  # rata-rata liga umum
        return self._compute_probs(lam, mu, 0.08, 7)

    def get_team_strength(self, team: str) -> dict:
        """Tampilkan kekuatan attack/defense tim."""
        if not self.fitted or team not in self.team_idx:
            return {}
        n  = len(self.teams)
        i  = self.team_idx[team]
        return {
            "team":    team,
            "attack":  round(float(self.params[i]), 4),
            "defense": round(float(self.params[n + i]), 4),
        }


# ── Instance global ───────────────────────────────────────────────────────────
dixon_model = DixonColes()
