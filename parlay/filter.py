import os

MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", 0.65))
MIN_EDGE = float(os.getenv("MIN_EDGE", 0.05))


def filter_candidates(predictions: list) -> list:
    """
    Filter pertandingan yang layak masuk parlay.
    
    predictions: list of dict dengan keys:
    - match_id, home_team, away_team
    - home_win, draw, away_win, confidence
    - edge, value_bet, red_flag
    - predicted_outcome, odds
    """
    candidates = []
    for pred in predictions:
        # Skip jika ada red flag
        if pred.get("red_flag", False):
            print(f"SKIP {pred['home_team']} vs {pred['away_team']}: Red flag detected")
            continue

        # Skip jika confidence kurang
        if pred.get("confidence", 0) < MIN_CONFIDENCE:
            print(f"SKIP {pred['home_team']} vs {pred['away_team']}: Low confidence {pred['confidence']:.1%}")
            continue

        # Skip jika tidak ada value bet
        if pred.get("edge", 0) < MIN_EDGE:
            print(f"SKIP {pred['home_team']} vs {pred['away_team']}: No edge {pred.get('edge', 0):.1%}")
            continue

        candidates.append(pred)
        print(f"MASUK {pred['home_team']} vs {pred['away_team']}: Conf={pred['confidence']:.1%}, Edge={pred.get('edge', 0):.1%}")

    return candidates


def rank_candidates(candidates: list) -> list:
    """Urutkan kandidat berdasarkan edge x confidence."""
    return sorted(
        candidates,
        key=lambda x: x.get("edge", 0) * x.get("confidence", 0),
        reverse=True
    )
