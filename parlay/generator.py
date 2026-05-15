from itertools import combinations
from parlay.filter import filter_candidates, rank_candidates
import os

MAX_LEGS = int(os.getenv("MAX_PARLAY_LEGS", 4))
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", 0.65))


def calculate_cumulative_prob(selections: list) -> float:
    """Hitung probabilitas kumulatif parlay."""
    prob = 1.0
    for s in selections:
        prob *= s["confidence"]
    return round(prob, 4)


def calculate_combined_odds(selections: list) -> float:
    """Hitung combined odds parlay."""
    odds = 1.0
    for s in selections:
        odds *= s.get("odds", 1.0)
    return round(odds, 3)


def calculate_ev(cumulative_prob: float, combined_odds: float) -> float:
    """Expected value parlay."""
    return round((cumulative_prob * combined_odds) - 1, 4)


def generate_optimal_parlay(predictions: list) -> dict:
    """
    Generate kombinasi parlay optimal dari semua kandidat.
    
    1. Filter kandidat layak
    2. Rank berdasarkan edge x confidence
    3. Cari kombinasi dengan EV tertinggi
    4. Return parlay terbaik
    """
    candidates = filter_candidates(predictions)
    candidates = rank_candidates(candidates)

    if not candidates:
        return {"status": "no_parlay", "reason": "Tidak ada kandidat yang memenuhi kriteria"}

    best_parlay = None
    best_ev = -999

    # Coba semua kombinasi dari 2 sampai MAX_LEGS
    for n_legs in range(2, min(MAX_LEGS + 1, len(candidates) + 1)):
        for combo in combinations(candidates, n_legs):
            selections = []
            for c in combo:
                selections.append({
                    "match": f"{c['home_team']} vs {c['away_team']}",
                    "pick": c["predicted_outcome"],
                    "confidence": c["confidence"],
                    "odds": c.get("odds", 1.9),
                    "edge": c.get("edge", 0)
                })

            cum_prob = calculate_cumulative_prob(selections)
            if cum_prob < MIN_CONFIDENCE:
                continue

            combined_odds = calculate_combined_odds(selections)
            ev = calculate_ev(cum_prob, combined_odds)

            if ev > best_ev:
                best_ev = ev
                best_parlay = {
                    "status": "parlay_found",
                    "legs": n_legs,
                    "selections": selections,
                    "cumulative_prob": cum_prob,
                    "combined_odds": combined_odds,
                    "expected_value": ev
                }

    if not best_parlay:
        # Ambil top 2 jika tidak ada kombinasi yang memenuhi threshold prob
        top2 = candidates[:2]
        if len(top2) >= 2:
            selections = [{
                "match": f"{c['home_team']} vs {c['away_team']}",
                "pick": c["predicted_outcome"],
                "confidence": c["confidence"],
                "odds": c.get("odds", 1.9),
                "edge": c.get("edge", 0)
            } for c in top2]
            best_parlay = {
                "status": "parlay_low_prob",
                "legs": 2,
                "selections": selections,
                "cumulative_prob": calculate_cumulative_prob(selections),
                "combined_odds": calculate_combined_odds(selections),
                "expected_value": calculate_ev(calculate_cumulative_prob(selections), calculate_combined_odds(selections))
            }
        else:
            return {"status": "no_parlay", "reason": "Kandidat tidak cukup"}

    return best_parlay


def print_parlay(parlay: dict):
    """Print parlay dengan format yang mudah dibaca."""
    if parlay["status"] == "no_parlay":
        print(f"\n❌ TIDAK ADA PARLAY: {parlay['reason']}")
        return

    print("\n" + "="*50)
    print("🎯 PARLAY REKOMENDASI")
    print("="*50)
    for i, sel in enumerate(parlay["selections"], 1):
        print(f"{i}. {sel['match']}")
        print(f"   Pick: {sel['pick']} | Conf: {sel['confidence']:.1%} | Odds: {sel['odds']} | Edge: {sel['edge']:.1%}")
    print("-"*50)
    print(f"Legs          : {parlay['legs']}")
    print(f"Combined Odds : {parlay['combined_odds']}")
    print(f"Cumul. Prob   : {parlay['cumulative_prob']:.1%}")
    print(f"Expected Value: {parlay['expected_value']:+.1%}")
    print("="*50)
