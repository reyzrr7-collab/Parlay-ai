"""
parlay/value_bet.py
───────────────────
Deteksi value bet & manajemen bankroll:
- Hitung edge antara model vs pasar
- Kelly Criterion untuk ukuran bet optimal
- Filter berdasarkan expected value (EV)
"""

import logging
from typing import Optional
from data.odds import odds_to_prob

log = logging.getLogger("value_bet")

EDGE_THRESHOLD = 0.04   # min 4% edge untuk dianggap value bet
MIN_KELLY      = 0.01   # min kelly fraction (1% bankroll)
MAX_KELLY      = 0.10   # max kelly fraction (10% bankroll — safety cap)


def calculate_edge(model_prob: float, market_odds: float) -> float:
    """
    Hitung edge = selisih probabilitas model vs pasar.
    Positif = model lebih optimis → value bet
    """
    if market_odds <= 1.0:
        return 0.0
    market_prob = 1 / market_odds
    return round(model_prob - market_prob, 4)


def kelly_criterion(model_prob: float, odds: float,
                    fraction: float = 0.5) -> float:
    """
    Kelly Criterion — hitung ukuran bet optimal.
    fraction=0.5 → Half-Kelly (lebih konservatif, direkomendasikan).

    Rumus: f = (p * b - q) / b
    p = probabilitas menang, q = probabilitas kalah, b = net odds
    """
    if odds <= 1.0 or model_prob <= 0:
        return 0.0

    b = odds - 1  # net odds
    q = 1 - model_prob
    kelly = (model_prob * b - q) / b

    if kelly <= 0:
        return 0.0

    # Terapkan fraction & cap
    kelly_adjusted = kelly * fraction
    return round(min(kelly_adjusted, MAX_KELLY), 4)


def expected_value(model_prob: float, odds: float) -> float:
    """
    Expected Value = (prob × odds) - 1
    EV > 0 → menguntungkan jangka panjang
    """
    return round((model_prob * odds) - 1, 4)


def analyze_value(
    prediction:  str,
    model_prob:  float,
    market_odds: dict,  # {"home": x, "draw": x, "away": x}
) -> dict:
    """
    Analisis value bet lengkap untuk satu prediksi.
    Return: edge, kelly, EV, is_value
    """
    if not market_odds:
        return {"is_value": False, "edge": 0, "kelly": 0, "ev": 0}

    # Ambil odds sesuai prediksi
    pred_lower = prediction.lower()
    if "home" in pred_lower:
        odds = market_odds.get("home", 0)
    elif "away" in pred_lower:
        odds = market_odds.get("away", 0)
    else:
        odds = market_odds.get("draw", 0)

    if not odds or odds <= 1.0:
        return {"is_value": False, "edge": 0, "kelly": 0, "ev": 0}

    edge = calculate_edge(model_prob, odds)
    ev   = expected_value(model_prob, odds)
    kel  = kelly_criterion(model_prob, odds) if edge > EDGE_THRESHOLD else 0.0

    return {
        "is_value":   edge > EDGE_THRESHOLD,
        "edge":       round(edge * 100, 2),   # dalam %
        "edge_raw":   edge,
        "ev":         round(ev * 100, 2),     # dalam %
        "kelly":      kel,
        "kelly_pct":  round(kel * 100, 2),
        "odds":       odds,
        "model_prob": round(model_prob * 100, 1),
        "market_prob": round((1/odds) * 100, 1) if odds > 0 else 0,
    }


def overround(odds_home: float, odds_draw: float, odds_away: float) -> float:
    """
    Hitung overround (margin) bookmaker.
    Semakin rendah = pasar lebih efisien.
    Pinnacle biasanya ~2-3%, bookie biasa ~8-12%.
    """
    if not all([odds_home, odds_draw, odds_away]):
        return 0.0
    return round(((1/odds_home + 1/odds_draw + 1/odds_away) - 1) * 100, 2)
