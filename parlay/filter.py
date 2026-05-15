"""
parlay/filter.py
────────────────
Filter pertandingan yang layak masuk parlay:
- Minimum confidence threshold
- Minimum value edge
- Hindari red flags (derby, key injuries, odds movement)
"""

import logging
from typing import List, Dict

log = logging.getLogger("filter")

# Threshold filter
MIN_CONFIDENCE  = 65    # % minimum confidence model
MIN_EDGE        = 3.0   # % minimum edge vs pasar
MAX_INJURIES    = 2     # max pemain kunci absen
MAX_ODDS_MOVE   = 15.0  # % max pergerakan odds (tanda insider info)
MAX_LEGS        = 4     # max kaki parlay


RED_FLAG_KEYWORDS = [
    "derby", "el clasico", "north london", "merseyside",
    "manchester", "milan", "rome", "glasgow",
    "cup final", "final", "playoff"
]


def is_derby(match_name: str) -> bool:
    """Deteksi derby/final — unpredictable, hindari."""
    name_lower = match_name.lower()
    return any(kw in name_lower for kw in RED_FLAG_KEYWORDS)


def check_red_flags(match: dict) -> List[str]:
    """
    Periksa semua red flag yang harus dihindari.
    Return: list alasan penolakan (kosong = aman)
    """
    flags = []

    if is_derby(match.get("match_name", "")):
        flags.append("❌ Derby/Final match — sangat unpredictable")

    if match.get("home_injuries", 0) + match.get("away_injuries", 0) > MAX_INJURIES * 2:
        flags.append(f"❌ Terlalu banyak pemain absen ({match.get('home_injuries',0) + match.get('away_injuries',0)})")

    odds_move = abs(match.get("odds_movement", 0))
    if odds_move > MAX_ODDS_MOVE:
        flags.append(f"❌ Odds bergerak {odds_move:.1f}% — kemungkinan info insider")

    if match.get("confidence", 0) < MIN_CONFIDENCE:
        flags.append(f"❌ Confidence terlalu rendah ({match.get('confidence',0)}%)")

    if match.get("edge", 0) < MIN_EDGE:
        flags.append(f"❌ Edge terlalu kecil ({match.get('edge',0):.1f}%)")

    return flags


def filter_parlay_candidates(matches: List[dict]) -> Dict:
    """
    Filter semua pertandingan — pilih yang layak parlay.
    Return: {"accepted": [...], "rejected": [...]}
    """
    accepted = []
    rejected = []

    for match in matches:
        flags = check_red_flags(match)
        if flags:
            rejected.append({**match, "reasons": flags})
            log.info("REJECTED: %s — %s", match.get("match_name"), flags[0])
        else:
            accepted.append(match)
            log.info("ACCEPTED: %s (confidence: %d%%, edge: %.1f%%)",
                     match.get("match_name"),
                     match.get("confidence", 0),
                     match.get("edge", 0))

    # Sort by confidence × edge (skor gabungan)
    accepted.sort(
        key=lambda m: m.get("confidence", 0) * 0.7 + m.get("edge", 0) * 0.3,
        reverse=True
    )

    return {
        "accepted": accepted[:MAX_LEGS],   # ambil top N saja
        "rejected": rejected,
        "total_candidates": len(matches),
        "total_accepted":   min(len(accepted), MAX_LEGS),
    }
