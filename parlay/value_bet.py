from data.odds import find_odds_for_match, remove_vig
import os

MIN_EDGE = float(os.getenv("MIN_EDGE", 0.05))


def detect_edge(model_prob: float, market_prob: float) -> float:
    """Hitung edge: selisih probabilitas model vs pasar."""
    return round(model_prob - market_prob, 4)


def find_value_bet(prediction: dict, home_team: str, away_team: str, sport: str = "soccer_epl") -> dict:
    """
    Bandingkan prediksi model vs true probability Pinnacle.
    Return dict dengan info value bet.
    """
    odds = find_odds_for_match(home_team, away_team, sport)
    true_h, true_d, true_a = remove_vig(odds.get("home"), odds.get("draw"), odds.get("away"))

    if true_h is None:
        return {"value_bet": False, "edge": 0, "reason": "Odds tidak tersedia"}

    edges = {
        "home_win": detect_edge(prediction["home_win"], true_h),
        "draw": detect_edge(prediction["draw"], true_d),
        "away_win": detect_edge(prediction["away_win"], true_a),
    }

    best_outcome = max(edges, key=edges.get)
    best_edge = edges[best_outcome]

    if best_edge > MIN_EDGE:
        return {
            "value_bet": True,
            "best_outcome": best_outcome,
            "edge": best_edge,
            "odds": odds.get(best_outcome.split("_")[0], odds.get("home")),
            "model_prob": prediction.get(best_outcome, 0),
            "market_prob": {"home_win": true_h, "draw": true_d, "away_win": true_a}.get(best_outcome, 0),
            "all_edges": edges
        }

    return {
        "value_bet": False,
        "edge": best_edge,
        "best_outcome": best_outcome,
        "all_edges": edges
    }


def expected_value(model_prob: float, odds: float) -> float:
    """Hitung expected value dari bet."""
    return round((model_prob * odds) - 1, 4)
