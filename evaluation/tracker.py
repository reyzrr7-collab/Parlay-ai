"""
evaluation/tracker.py
─────────────────────
Simpan semua prediksi dan evaluasi akurasi setelah match selesai.
Feedback loop: prediksi → match selesai → evaluasi → improve model.
"""

import logging
from datetime import datetime
from database.queries import (
    save_prediction, evaluate_prediction,
    get_accuracy_stats, save_parlay, evaluate_parlay
)

log = logging.getLogger("tracker")


def log_prediction(
    match_name:    str,
    prediction:    str,
    confidence:    float,
    ensemble_result: dict,
    value_analysis:  dict,
    match_id:      str = "",
    league:        str = "",
    pick_type:     str = "1X2",
) -> int:
    """
    Simpan prediksi ke database.
    Return pred_id untuk evaluasi nanti.
    """
    probs = {
        "home": ensemble_result.get("home_win", 0) / 100,
        "draw": ensemble_result.get("draw", 0) / 100,
        "away": ensemble_result.get("away_win", 0) / 100,
    }
    dixon = ensemble_result.get("breakdown", {}).get("dixon_coles", {})
    xgb   = ensemble_result.get("breakdown", {}).get("xgboost", {})

    pred_id = save_prediction(
        match_name   = match_name,
        prediction   = prediction,
        confidence   = confidence,
        probs        = probs,
        dixon        = {
            "home": dixon.get("home", 0) / 100,
            "draw": dixon.get("draw", 0) / 100,
            "away": dixon.get("away", 0) / 100,
        },
        xgb          = {
            "home": xgb.get("home", 0) / 100,
            "draw": xgb.get("draw", 0) / 100,
            "away": xgb.get("away", 0) / 100,
        },
        pinnacle_prob = value_analysis.get("market_prob", 0) / 100,
        edge          = value_analysis.get("edge_raw", 0),
        kelly         = value_analysis.get("kelly", 0),
        match_id      = match_id,
        league        = league,
        pick_type     = pick_type,
    )
    log.info("✅ Prediksi tersimpan — ID: %d | %s → %s (%.0f%%)",
             pred_id, match_name, prediction, confidence)
    return pred_id


def record_result(pred_id: int, actual_result: str) -> dict:
    """
    Isi hasil aktual setelah match selesai.
    actual_result: "home" | "draw" | "away"
    """
    result = evaluate_prediction(pred_id, actual_result)
    status = "✅ BENAR" if result.get("was_correct") else "❌ SALAH"
    log.info("Evaluasi ID %d: %s | Brier: %.4f", pred_id, status, result.get("brier_score", 0))
    return result


def print_accuracy_report() -> None:
    """Cetak laporan akurasi keseluruhan."""
    stats = get_accuracy_stats()
    print("\n" + "═" * 45)
    print("  📊 LAPORAN AKURASI PREDIKSI")
    print("═" * 45)
    print(f"  Total prediksi dievaluasi : {stats['total']}")
    print(f"  Akurasi                   : {stats['accuracy']}%")
    print(f"  Avg Brier Score           : {stats['avg_brier']} (< 0.20 = bagus)")
    print(f"  Avg Edge vs pasar         : {stats['avg_edge']}%")
    print("═" * 45)
    if stats["total"] > 0:
        if stats["accuracy"] >= 60:
            print("  🟢 Performa: BAGUS")
        elif stats["accuracy"] >= 52:
            print("  🟡 Performa: RATA-RATA")
        else:
            print("  🔴 Performa: PERLU PERBAIKAN")
    print()
