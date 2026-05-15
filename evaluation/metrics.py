"""
evaluation/metrics.py
─────────────────────
Metrik evaluasi prediksi:
- Brier Score     : akurasi probabilitas (< 0.20 = bagus)
- RPS             : Ranked Probability Score (lebih robust dari accuracy)
- ROI             : Return on Investment jika dipakai betting
- Calibration     : seberapa terkalibrasi probabilitas model
"""

import sqlite3
import logging
import numpy as np
from typing import List, Dict
from database.models import DB_PATH

log = logging.getLogger("metrics")


def brier_score(probs: List[float], outcome_idx: int) -> float:
    """
    Brier Score untuk satu prediksi.
    probs: [p_home, p_draw, p_away]
    outcome_idx: 0=home, 1=draw, 2=away
    """
    outcome = [0, 0, 0]
    outcome[outcome_idx] = 1
    return sum((p - o) ** 2 for p, o in zip(probs, outcome)) / 3


def ranked_probability_score(probs: List[float], outcome_idx: int) -> float:
    """
    RPS — lebih baik dari accuracy untuk evaluasi probabilistik.
    Menghukum prediksi yang 'jauh' dari hasil sebenarnya.
    """
    cum_probs   = np.cumsum(probs)
    cum_outcome = np.cumsum([1 if i == outcome_idx else 0 for i in range(len(probs))])
    rps = sum((cp - co) ** 2 for cp, co in zip(cum_probs[:-1], cum_outcome[:-1]))
    return rps / (len(probs) - 1)


def calculate_roi(predictions: List[dict], bankroll: float = 1000.0) -> dict:
    """
    Hitung ROI simulasi jika semua prediksi dipasang dengan Kelly.
    """
    total_staked = 0.0
    total_return = 0.0

    for p in predictions:
        if p.get("actual_result") is None:
            continue
        kelly  = p.get("kelly_fraction", 0) or 0
        odds   = p.get("odds", 1.5) or 1.5
        stake  = bankroll * kelly
        won    = p.get("was_correct", 0)

        total_staked += stake
        total_return += stake * odds if won else 0.0

    if total_staked == 0:
        return {"roi": 0, "profit": 0, "total_staked": 0}

    profit = total_return - total_staked
    roi    = (profit / total_staked) * 100
    return {
        "roi":           round(roi, 2),
        "profit":        round(profit, 2),
        "total_staked":  round(total_staked, 2),
        "total_return":  round(total_return, 2),
    }


def full_report() -> None:
    """Generate laporan lengkap dari semua data di database."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        """SELECT prediction, confidence, model_prob_home, model_prob_draw,
                  model_prob_away, actual_result, was_correct, brier_score,
                  kelly_fraction, edge
           FROM prediction_log WHERE actual_result IS NOT NULL"""
    ).fetchall()
    conn.close()

    if not rows:
        print("Belum ada data evaluasi.")
        return

    total     = len(rows)
    correct   = sum(1 for r in rows if r[6] == 1)
    avg_brier = np.mean([r[7] for r in rows if r[7] is not None])
    avg_conf  = np.mean([r[1] for r in rows if r[1] is not None])

    # RPS
    rps_scores = []
    for r in rows:
        pred, conf, ph, pd_v, pa, actual = r[0], r[1], r[2], r[3], r[4], r[5]
        probs = [ph or 0, pd_v or 0, pa or 0]
        outcome_map = {"home": 0, "draw": 1, "away": 2}
        idx = outcome_map.get(actual, 0)
        rps_scores.append(ranked_probability_score(probs, idx))

    avg_rps = np.mean(rps_scores)

    # Per prediksi type
    by_type = {}
    for r in rows:
        pred = r[0]
        by_type.setdefault(pred, {"total": 0, "correct": 0})
        by_type[pred]["total"] += 1
        if r[6] == 1:
            by_type[pred]["correct"] += 1

    print("\n" + "═" * 50)
    print("  📊 LAPORAN METRIK LENGKAP")
    print("═" * 50)
    print(f"  Total sampel         : {total}")
    print(f"  Akurasi              : {round(correct/total*100, 1)}%")
    print(f"  Avg Confidence       : {round(avg_conf, 1)}%")
    print(f"  Avg Brier Score      : {round(avg_brier, 4)} (target < 0.20)")
    print(f"  Avg RPS              : {round(avg_rps, 4)}  (target < 0.22)")
    print()
    print("  Per Tipe Prediksi:")
    for ptype, v in by_type.items():
        acc = round(v["correct"] / v["total"] * 100, 1)
        print(f"    {ptype:<12} : {acc}% ({v['correct']}/{v['total']})")
    print("═" * 50)
