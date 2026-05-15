import numpy as np
from database.models import Session, Prediction, EvaluationLog
from models.xgboost_model import train_xgboost
from data.preprocessor import features_to_array
from evaluation.metrics import get_accuracy, get_average_brier_score

RETRAIN_THRESHOLD_ACCURACY = 0.50
RETRAIN_MIN_SAMPLES = 50


def collect_training_data() -> tuple:
    """Kumpulkan data training dari evaluasi yang sudah ada."""
    session = Session()
    try:
        logs = session.query(EvaluationLog).all()
        if len(logs) < RETRAIN_MIN_SAMPLES:
            print(f"Data tidak cukup untuk retrain: {len(logs)}/{RETRAIN_MIN_SAMPLES}")
            return None, None

        X, y = [], []
        outcome_map = {"home_win": 0, "draw": 1, "away_win": 2}

        for log in logs:
            pred = session.query(Prediction).get(log.prediction_id)
            if not pred or not pred.model_breakdown:
                continue
            breakdown = pred.model_breakdown
            features = [
                breakdown.get("home_win", 0.33),
                breakdown.get("draw", 0.33),
                breakdown.get("away_win", 0.33),
                pred.confidence or 0.33,
                pred.edge or 0.0,
            ]
            X.append(features)
            y.append(outcome_map.get(log.actual_outcome, 0))

        return np.array(X), np.array(y)
    finally:
        session.close()


def should_retrain() -> bool:
    """Cek apakah model perlu diretrain."""
    accuracy = get_accuracy()
    brier = get_average_brier_score()
    if accuracy < RETRAIN_THRESHOLD_ACCURACY:
        print(f"Retrain needed: accuracy {accuracy:.1%} < {RETRAIN_THRESHOLD_ACCURACY:.1%}")
        return True
    if brier > 0.25:
        print(f"Retrain needed: brier score {brier:.4f} > 0.25")
        return True
    return False


def auto_retrain():
    """Auto retrain jika performa menurun."""
    if not should_retrain():
        print("Model masih performa baik, tidak perlu retrain.")
        return

    print("Memulai retrain model...")
    X, y = collect_training_data()
    if X is None:
        return

    train_xgboost(X, y)
    print(f"Retrain selesai dengan {len(X)} samples.")


if __name__ == "__main__":
    auto_retrain()
