"""
parlay/generator.py
───────────────────
Build kombinasi parlay optimal dari kandidat terpilih:
- Hitung kumulatif probabilitas
- Hitung expected value parlay
- GO/NO-GO recommendation
"""

import math
import logging
from typing import List, Dict
from parlay.filter import filter_parlay_candidates
from parlay.value_bet import analyze_value

log = logging.getLogger("generator")

MIN_CUM_PROB  = 20.0  # % minimum kumulatif untuk GO
MIN_LEGS      = 2     # minimum kaki parlay


def calculate_parlay_odds(picks: List[dict]) -> float:
    """
    Hitung total odds parlay (perkalian semua odds).
    """
    total = 1.0
    for pick in picks:
        total *= pick.get("odds", 1.0)
    return round(total, 2)


def calculate_cumulative_prob(picks: List[dict]) -> float:
    """
    Hitung probabilitas kumulatif parlay.
    Asumsi: setiap pertandingan independen.
    """
    cum = 1.0
    for pick in picks:
        cum *= pick.get("confidence", 50) / 100
    return round(cum * 100, 2)


def parlay_expected_value(cum_prob: float, total_odds: float) -> float:
    """
    EV parlay = (cum_prob × total_odds) - 1
    """
    return round((cum_prob / 100) * total_odds - 1, 4)


def generate_parlay(matches: List[dict]) -> dict:
    """
    Generate rekomendasi parlay dari list pertandingan.

    Alur:
    1. Filter candidates
    2. Ambil top picks
    3. Hitung probabilitas kumulatif
    4. GO/NO-GO decision
    5. Return lengkap dengan reasoning
    """
    # Filter
    filtered = filter_parlay_candidates(matches)
    picks = filtered["accepted"]

    if len(picks) < MIN_LEGS:
        return {
            "status":     "NO-GO",
            "reason":     f"Hanya {len(picks)} match lolos filter (min {MIN_LEGS})",
            "picks":      [],
            "rejected":   filtered["rejected"],
        }

    # Hitung odds & probabilitas
    total_odds  = calculate_parlay_odds(picks)
    cum_prob    = calculate_cumulative_prob(picks)
    ev          = parlay_expected_value(cum_prob, total_odds)

    # GO/NO-GO
    go = cum_prob >= MIN_CUM_PROB and ev > 0

    return {
        "status":          "✅ GO" if go else "❌ NO-GO",
        "go":              go,
        "total_legs":      len(picks),
        "total_odds":      total_odds,
        "cum_probability": cum_prob,
        "expected_value":  round(ev * 100, 2),   # dalam %
        "picks":           picks,
        "rejected":        filtered["rejected"],
        "summary":         _generate_summary(picks, cum_prob, total_odds, ev),
    }


def _generate_summary(picks: list, cum_prob: float,
                       total_odds: float, ev: float) -> str:
    lines = ["📊 PARLAY SUMMARY", "─" * 40]
    for i, p in enumerate(picks, 1):
        lines.append(
            f"{i}. {p.get('match_name','?')} → {p.get('prediction','?')} "
            f"({p.get('confidence',0)}%) @ {p.get('odds',0):.2f}"
        )
    lines += [
        "─" * 40,
        f"Total Odds      : {total_odds}x",
        f"Kumulatif Prob  : {cum_prob}%",
        f"Expected Value  : {round(ev*100, 1)}%",
    ]
    return "\n".join(lines)
