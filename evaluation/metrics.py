import numpy as np
from database.queries import get_all_predictions
from database.models import Session, EvaluationLog, Prediction


def get_accuracy() -> float:
    """Hitung akurasi prediksi keseluruhan."""
    session = Session()
    try:
        logs = session.query(EvaluationLog).all()
        if not logs:
            return 0.0
        correct = sum(1 for l in logs if l.correct)
        return round(correct / len(logs), 4)
    finally:
        session.close()


def get_average_brier_score() -> float:
    """Hitung rata-rata Brier Score (lebih rendah = lebih baik)."""
    session = Session()
    try:
        logs = session.query(EvaluationLog).all()
        if not logs:
            return 1.0
        scores = [l.brier_score for l in logs if l.brier_score is not None]
        return round(np.mean(scores), 4) if scores else 1.0
    finally:
        session.close()


def get_rps(predictions_with_actuals: list) -> float:
    """
    Ranked Probability Score - metric terbaik untuk probabilistic forecast.
    Lebih rendah = lebih baik.
    """
    rps_scores = []
    outcome_map = {"home_win": 0, "draw": 1, "away_win": 2}

    for item in predictions_with_actuals:
        pred_probs = [item["home_win"], item["draw"], item["away_win"]]
        actual_idx = outcome_map.get(item["actual"], 0)
        actual_probs = [1 if i == actual_idx else 0 for i in range(3)]

        # Kumulatif
        cum_pred = np.cumsum(pred_probs)
        cum_actual = np.cumsum(actual_probs)
        rps = np.mean((cum_pred - cum_actual) ** 2)
        rps_scores.append(rps)

    return round(float(np.mean(rps_scores)), 4) if rps_scores else 1.0


def get_roi(parlays_with_results: list, stake: float = 1.0) -> float:
    """
    Hitung ROI dari semua parlay.
    parlays_with_results: list of {combined_odds, result: 'win'/'loss'}
    """
    total_staked = len(parlays_with_results) * stake
    total_return = sum(
        p["combined_odds"] * stake if p["result"] == "win" else 0
        for p in parlays_with_results
    )
    if total_staked == 0:
        return 0.0
    return round((total_return - total_staked) / total_staked * 100, 2)


def print_summary():
    """Print ringkasan performa sistem."""
    accuracy = get_accuracy()
    brier = get_average_brier_score()

    print("\n" + "="*40)
    print("📊 RINGKASAN PERFORMA SISTEM")
    print("="*40)
    print(f"Akurasi          : {accuracy:.1%}")
    print(f"Brier Score      : {brier:.4f} (target < 0.22)")
    print("="*40)

    if accuracy >= 0.55:
        print("✅ Model performing well")
    else:
        print("⚠️ Model needs retraining")
