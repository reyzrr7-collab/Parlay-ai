"""
models/ensemble.py
──────────────────
Gabungkan output semua model:
- Dixon-Coles (model statistik utama)
- XGBoost (machine learning)
- Odds calibration (wisdom of market)

Output: probabilitas final + confidence score
"""

import logging
from typing import Optional
from models.dixon_coles import dixon_model
from models.xgboost_model import xgb_model
from data.odds import odds_to_prob

log = logging.getLogger("ensemble")


# Bobot tiap model (total harus = 1.0)
WEIGHTS = {
    "dixon_coles": 0.45,
    "xgboost":     0.35,
    "market":      0.20,
}


def ensemble_predict(
    home_team:     str,
    away_team:     str,
    features:      "np.ndarray" = None,
    home_xg:       float = 1.4,
    away_xg:       float = 1.1,
    market_odds:   dict = None,
) -> dict:
    """
    Prediksi ensemble dari semua model.
    Return: probabilitas final + confidence + breakdown per model
    """
    results = {}
    weights_used = {}

    # ── 1. Dixon-Coles ──────────────────────────────────────────────────────
    try:
        if dixon_model.fitted:
            dc = dixon_model.predict(home_team, away_team)
        else:
            dc = dixon_model.predict_with_xg(home_xg, away_xg)
        results["dixon_coles"] = {
            "home": dc["home_win"] / 100,
            "draw": dc["draw"] / 100,
            "away": dc["away_win"] / 100,
        }
        weights_used["dixon_coles"] = WEIGHTS["dixon_coles"]
    except Exception as e:
        log.warning("Dixon-Coles error: %s", e)

    # ── 2. XGBoost ──────────────────────────────────────────────────────────
    if features is not None and xgb_model.fitted:
        try:
            xgb = xgb_model.predict(features)
            if xgb:
                results["xgboost"] = {
                    "home": xgb["home_win"] / 100,
                    "draw": xgb["draw"] / 100,
                    "away": xgb["away_win"] / 100,
                }
                weights_used["xgboost"] = WEIGHTS["xgboost"]
        except Exception as e:
            log.warning("XGBoost error: %s", e)

    # ── 3. Market (Odds) ────────────────────────────────────────────────────
    if market_odds and all(market_odds.get(k, 0) > 1 for k in ["home", "draw", "away"]):
        try:
            mkt = odds_to_prob(
                market_odds["home"],
                market_odds["draw"],
                market_odds["away"]
            )
            results["market"] = {
                "home": mkt["home"],
                "draw": mkt["draw"],
                "away": mkt["away"],
            }
            weights_used["market"] = WEIGHTS["market"]
        except Exception as e:
            log.warning("Market odds error: %s", e)

    if not results:
        # Fallback jika semua model gagal
        return {
            "home_win": 40.0, "draw": 25.0, "away_win": 35.0,
            "confidence": 30, "breakdown": {}, "best_score": "1-1"
        }

    # ── Normalisasi bobot ────────────────────────────────────────────────────
    total_w = sum(weights_used.values())
    norm_weights = {k: v / total_w for k, v in weights_used.items()}

    # ── Weighted average ─────────────────────────────────────────────────────
    final_home = sum(results[m]["home"] * norm_weights[m] for m in results)
    final_draw = sum(results[m]["draw"] * norm_weights[m] for m in results)
    final_away = sum(results[m]["away"] * norm_weights[m] for m in results)

    # Normalisasi agar total = 1
    total = final_home + final_draw + final_away
    final_home /= total
    final_draw /= total
    final_away /= total

    # ── Confidence score ─────────────────────────────────────────────────────
    # Tinggi jika model-model sepakat, rendah jika saling bertentangan
    max_prob = max(final_home, final_draw, final_away)
    agreement = _model_agreement(results)
    confidence = int(max_prob * 70 + agreement * 30)

    # ── Prediction label ─────────────────────────────────────────────────────
    if final_home >= final_draw and final_home >= final_away:
        prediction = "Home Win"
    elif final_away >= final_home and final_away >= final_draw:
        prediction = "Away Win"
    else:
        prediction = "Draw"

    # Dixon-Coles best score
    best_score = "1-1"
    if "dixon_coles" in results and dixon_model.fitted:
        dc_full = dixon_model.predict(home_team, away_team)
        best_score = dc_full.get("best_score", "1-1")

    return {
        "prediction":  prediction,
        "home_win":    round(final_home * 100, 1),
        "draw":        round(final_draw * 100, 1),
        "away_win":    round(final_away * 100, 1),
        "confidence":  confidence,
        "best_score":  best_score,
        "models_used": list(results.keys()),
        "breakdown": {
            m: {
                "home": round(results[m]["home"] * 100, 1),
                "draw": round(results[m]["draw"] * 100, 1),
                "away": round(results[m]["away"] * 100, 1),
                "weight": round(norm_weights[m] * 100, 1)
            }
            for m in results
        }
    }


def _model_agreement(results: dict) -> float:
    """
    Ukur seberapa sepakat model-model satu sama lain.
    1.0 = semua sepakat, 0.0 = semua berbeda.
    """
    if len(results) < 2:
        return 1.0

    predictions = []
    for m in results:
        r = results[m]
        if r["home"] >= r["draw"] and r["home"] >= r["away"]:
            predictions.append("home")
        elif r["away"] >= r["home"] and r["away"] >= r["draw"]:
            predictions.append("away")
        else:
            predictions.append("draw")

    most_common = max(set(predictions), key=predictions.count)
    return predictions.count(most_common) / len(predictions)
