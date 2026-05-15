"""
data/preprocessor.py
────────────────────
Feature engineering untuk model ML (XGBoost):
- Gabungkan data dari collector + scraper + odds
- Hitung form points, fatigue, home advantage, dll
- Output: feature vector siap pakai model
"""

import math
import logging
import numpy as np
from typing import Optional
from datetime import datetime

log = logging.getLogger("preprocessor")


def form_to_points(form_list: list, is_home: bool = True) -> float:
    """
    Hitung form points dari list pertandingan terakhir.
    W=3, D=1, L=0 — dengan time decay.
    is_home: ambil hanya hasil home atau away saja.
    """
    pts = 0.0
    weight_total = 0.0
    decay = 0.85  # setiap match lebih lama bobotnya 0.85x

    relevant = [m for m in form_list if m.get("is_home") == is_home] if is_home is not None else form_list

    for i, match in enumerate(relevant[:10]):
        w = decay ** i
        result = match.get("result", "L")
        pts += {"W": 3, "D": 1, "L": 0}.get(result, 0) * w
        weight_total += w * 3

    return round(pts / weight_total, 4) if weight_total else 0.0


def goals_avg_from_form(form_list: list, scored: bool = True) -> float:
    """Rata-rata gol dari form list."""
    key = "scored" if scored else "conceded"
    values = [m.get(key, 0) for m in form_list[:10] if m.get(key) is not None]
    return round(sum(values) / len(values), 3) if values else 0.0


def clean_sheet_pct(form_list: list) -> float:
    """Persentase clean sheet dari form list."""
    if not form_list:
        return 0.0
    cs = sum(1 for m in form_list[:10] if m.get("conceded", 1) == 0)
    return round(cs / min(len(form_list), 10), 3)


def h2h_to_features(h2h: dict, is_home_team: bool) -> dict:
    """Konversi H2H data ke features."""
    total = h2h.get("total_matches", 1) or 1
    wins  = h2h.get("team1_wins", 0) if is_home_team else h2h.get("team2_wins", 0)
    return {
        "h2h_win_rate":  round(wins / total, 3),
        "h2h_avg_goals": h2h.get("avg_goals", 2.5),
    }


def build_match_features(
    home_form:    list,
    away_form:    list,
    home_stats:   dict,
    away_stats:   dict,
    h2h:          dict,
    home_xg:      dict,
    away_xg:      dict,
    home_fatigue: float,
    away_fatigue: float,
    home_injuries: int,
    away_injuries: int,
) -> np.ndarray:
    """
    Bangun feature vector lengkap untuk XGBoost.
    20 fitur total.
    """
    features = [
        # Form keseluruhan
        form_to_points(home_form, is_home=None),
        form_to_points(away_form, is_home=None),
        # Form home/away split
        form_to_points(home_form, is_home=True),
        form_to_points(away_form, is_home=False),
        # xG
        home_xg.get("xg_avg", 0) or home_stats.get("home_goals_avg", 1.2),
        away_xg.get("xg_avg", 0) or away_stats.get("away_goals_avg", 1.0),
        home_xg.get("xga_avg", 0) or home_stats.get("home_conc_avg", 1.0),
        away_xg.get("xga_avg", 0) or away_stats.get("away_conc_avg", 1.2),
        # Goals avg dari form
        goals_avg_from_form(home_form, scored=True),
        goals_avg_from_form(away_form, scored=True),
        # Clean sheet
        clean_sheet_pct(home_form),
        clean_sheet_pct(away_form),
        # H2H
        h2h_to_features(h2h, True)["h2h_win_rate"],
        h2h_to_features(h2h, False)["h2h_win_rate"],
        h2h.get("avg_goals", 2.5),
        # Kontekstual
        home_fatigue,
        away_fatigue,
        float(home_injuries),
        float(away_injuries),
        # Home advantage baseline
        1.0,
    ]
    return np.array(features, dtype=np.float32).reshape(1, -1)


FEATURE_NAMES = [
    "home_form_overall", "away_form_overall",
    "home_form_split", "away_form_split",
    "home_xg_avg", "away_xg_avg",
    "home_xga_avg", "away_xga_avg",
    "home_goals_avg", "away_goals_avg",
    "home_cs_pct", "away_cs_pct",
    "h2h_home_win_rate", "h2h_away_win_rate",
    "h2h_avg_goals",
    "home_fatigue", "away_fatigue",
    "home_injuries", "away_injuries",
    "home_advantage",
]
