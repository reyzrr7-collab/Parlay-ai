import numpy as np
import xgboost as xgb
import os
import pickle
from data.preprocessor import features_to_array, build_feature_vector

MODEL_PATH = "/opt/parlay-ai/models/xgboost_model.pkl"


def train_xgboost(X: np.ndarray, y: np.ndarray):
    """Train XGBoost model. y: 0=home win, 1=draw, 2=away win."""
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42
    )
    model.fit(X, y)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print("XGBoost model saved.")
    return model


def load_xgboost():
    """Load model yang sudah ditraining."""
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def xgboost_predict(match_data: dict) -> dict:
    """Prediksi dengan XGBoost."""
    model = load_xgboost()
    if model is None:
        # Fallback ke prediksi default jika belum ada model
        return {
            "home_win": 0.40,
            "draw": 0.28,
            "away_win": 0.32,
            "model": "xgboost_default"
        }

    features = build_feature_vector(match_data)
    X = features_to_array(features).reshape(1, -1)
    probs = model.predict_proba(X)[0]

    # probs[0]=home win, probs[1]=draw, probs[2]=away win
    return {
        "home_win": round(float(probs[0]), 4),
        "draw": round(float(probs[1]), 4),
        "away_win": round(float(probs[2]), 4),
        "model": "xgboost"
    }
