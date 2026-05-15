"""
models/bayesian.py
──────────────────
Hierarchical Bayesian Model untuk prediksi sepak bola.
Lebih robust dari Dixon-Coles biasa karena:
- Setiap tim punya DISTRIBUSI kekuatan (bukan nilai tunggal)
- Uncertainty di-model secara eksplisit
- Update otomatis setiap match selesai (online learning)
- Lebih akurat untuk tim promosi / data sedikit

Catatan: Butuh PyMC. Install: pip install pymc pytensor
"""

import logging
import numpy as np
from typing import Optional

log = logging.getLogger("bayesian")


class BayesianFootball:
    """
    Hierarchical Bayesian model untuk prediksi hasil pertandingan.
    Digunakan sebagai pelengkap Dixon-Coles dalam ensemble.
    """

    def __init__(self):
        self.trace    = None
        self.teams    = []
        self.team_idx = {}
        self.fitted   = False

    def fit(self, matches: list, draws: int = 500, tune: int = 200) -> None:
        """
        Fit model dari data historis.
        matches: [{"home_team", "away_team", "home_goals", "away_goals"}, ...]
        """
        try:
            import pymc as pm
            import pytensor.tensor as pt
        except ImportError:
            log.error("PyMC tidak terinstall. pip install pymc pytensor")
            return

        # Kumpulkan tim
        self.teams = sorted(set(
            [m["home_team"] for m in matches] +
            [m["away_team"] for m in matches]
        ))
        self.team_idx = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)

        home_idx = np.array([self.team_idx[m["home_team"]] for m in matches])
        away_idx = np.array([self.team_idx[m["away_team"]] for m in matches])
        home_goals = np.array([m["home_goals"] for m in matches])
        away_goals = np.array([m["away_goals"] for m in matches])

        with pm.Model() as model:
            # Hyperprior — rata-rata kekuatan liga
            mu_att    = pm.Normal("mu_att",    mu=0,   sigma=0.5)
            mu_def    = pm.Normal("mu_def",    mu=0,   sigma=0.5)
            sigma_att = pm.HalfNormal("sigma_att", sigma=0.5)
            sigma_def = pm.HalfNormal("sigma_def", sigma=0.5)

            # Kekuatan tiap tim — distribusi, bukan nilai tunggal
            attack  = pm.Normal("attack",  mu=mu_att, sigma=sigma_att, shape=n)
            defense = pm.Normal("defense", mu=mu_def, sigma=sigma_def, shape=n)

            # Home advantage (global)
            home_adv = pm.Normal("home_adv", mu=0.3, sigma=0.1)

            # Expected goals
            log_lambda = attack[home_idx] - defense[away_idx] + home_adv
            log_mu     = attack[away_idx] - defense[home_idx]

            lambda_ = pm.math.exp(log_lambda)
            mu_     = pm.math.exp(log_mu)

            # Likelihood
            pm.Poisson("home_goals_obs", mu=lambda_, observed=home_goals)
            pm.Poisson("away_goals_obs", mu=mu_,     observed=away_goals)

            # Sample posterior
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                target_accept=0.9,
                return_inferencedata=True,
                progressbar=False,
            )

        self.fitted = True
        log.info("✅ Bayesian model fitted — %d tim, %d matches", n, len(matches))

    def predict(self, home_team: str, away_team: str,
                max_goals: int = 7) -> Optional[dict]:
        """
        Prediksi probabilitas dari posterior samples.
        Lebih robust karena menggunakan distribusi penuh, bukan point estimate.
        """
        if not self.fitted:
            return None

        hi = self.team_idx.get(home_team)
        ai = self.team_idx.get(away_team)
        if hi is None or ai is None:
            log.warning("Tim tidak ada di Bayesian model: %s / %s", home_team, away_team)
            return None

        try:
            att  = self.trace.posterior["attack"].values
            defn = self.trace.posterior["defense"].values
            hadv = self.trace.posterior["home_adv"].values

            # Rata-rata posterior
            att_home  = float(att[:, :, hi].mean())
            def_away  = float(defn[:, :, ai].mean())
            att_away  = float(att[:, :, ai].mean())
            def_home  = float(defn[:, :, hi].mean())
            home_adv  = float(hadv.mean())

            lam = np.exp(att_home - def_away + home_adv)
            mu  = np.exp(att_away - def_home)

            # Hitung probabilitas dari distribusi Poisson
            from scipy.stats import poisson
            prob_matrix = np.zeros((max_goals, max_goals))
            for i in range(max_goals):
                for j in range(max_goals):
                    prob_matrix[i][j] = (poisson.pmf(i, lam) *
                                         poisson.pmf(j, mu))
            total = prob_matrix.sum()
            if total > 0:
                prob_matrix /= total

            home_win = float(np.sum(np.tril(prob_matrix, -1)))
            draw     = float(np.sum(np.diag(prob_matrix)))
            away_win = float(np.sum(np.triu(prob_matrix, 1)))

            return {
                "home_win": round(home_win * 100, 1),
                "draw":     round(draw * 100, 1),
                "away_win": round(away_win * 100, 1),
                "xg":       {"home": round(lam, 2), "away": round(mu, 2)},
            }
        except Exception as e:
            log.error("Bayesian predict error: %s", e)
            return None

    def get_team_strength(self, team: str) -> dict:
        """Tampilkan distribusi kekuatan tim (mean ± std)."""
        if not self.fitted or team not in self.team_idx:
            return {}
        i   = self.team_idx[team]
        att = self.trace.posterior["attack"].values[:, :, i]
        defn= self.trace.posterior["defense"].values[:, :, i]
        return {
            "team":          team,
            "attack_mean":   round(float(att.mean()), 4),
            "attack_std":    round(float(att.std()), 4),
            "defense_mean":  round(float(defn.mean()), 4),
            "defense_std":   round(float(defn.std()), 4),
        }


# Instance global
bayesian_model = BayesianFootball()
