"""
models/xgboost_model.py
───────────────────────
XGBoost model untuk prediksi hasil pertandingan.
- Training dari data historis
- Prediksi probabilitas 1X2
- Auto-save & load model
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Optional
from data.preprocessor import FEATURE_NAMES

log = logging.getLogger("xgboost_model")
MODEL_PATH = "xgboost_model.pkl"


class FootballXGBoost:
    """
    XGBoost wrapper untuk prediksi sepak bola.
    Target: 0=Away Win, 1=Draw, 2=Home Win
    """

    def __init__(self):
        self.model  = None
        self.fitted = False

    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Latih model dari feature matrix X dan label y.
        y: 0=away, 1=draw, 2=home
        """
        try:
            from xgboost import XGBClassifier
            from sklearn.model_selection import cross_val_score
            from sklearn.calibration import CalibratedClassifierCV

            base = XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric="mlogloss",
                random_state=42,
            )

            # Kalibrasi probabilitas — penting untuk prediksi
            self.model = CalibratedClassifierCV(base, method="isotonic", cv=3)
            self.model.fit(X, y)
            self.fitted = True

            # Cross-validation accuracy
            cv_scores = cross_val_score(base, X, y, cv=5, scoring="accuracy")
            stats = {
                "cv_accuracy": round(float(cv_scores.mean()) * 100, 1),
                "cv_std":      round(float(cv_scores.std()) * 100, 2),
                "n_samples":   len(y),
            }

            self.save()
            log.info("✅ XGBoost trained — CV accuracy: %.1f%%", stats["cv_accuracy"])
            return stats

        except ImportError:
            log.error("xgboost tidak terinstall. pip install xgboost scikit-learn")
            return {}
        except Exception as e:
            log.error("XGBoost training error: %s", e)
            return {}

    def predict(self, X: np.ndarray) -> Optional[dict]:
        """Prediksi probabilitas 1X2."""
        if not self.fitted:
            return None
        try:
            proba = self.model.predict_proba(X)[0]
            # Classes: 0=away, 1=draw, 2=home
            return {
                "home_win": round(float(proba[2]) * 100, 1),
                "draw":     round(float(proba[1]) * 100, 1),
                "away_win": round(float(proba[0]) * 100, 1),
            }
        except Exception as e:
            log.error("XGBoost predict error: %s", e)
            return None

    def feature_importance(self) -> dict:
        """Tampilkan feature importance."""
        if not self.fitted:
            return {}
        try:
            base = self.model.estimators_[0] if hasattr(self.model, "estimators_") else self.model
            if hasattr(base, "feature_importances_"):
                importance = base.feature_importances_
                return dict(sorted(
                    zip(FEATURE_NAMES, importance),
                    key=lambda x: x[1], reverse=True
                ))
        except:
            pass
        return {}

    def save(self, path: str = MODEL_PATH) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        log.info("Model disimpan: %s", path)

    def load(self, path: str = MODEL_PATH) -> bool:
        if not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as f:
                self.model = pickle.load(f)
            self.fitted = True
            log.info("Model dimuat: %s", path)
            return True
        except Exception as e:
            log.error("Load model error: %s", e)
            return False


def prepare_training_data(matches_df: pd.DataFrame) -> tuple:
    """
    Siapkan data training dari DataFrame historis.
    matches_df harus punya kolom:
    home_goals, away_goals, home_xg, away_xg,
    home_form, away_form, dll.
    """
    from data.preprocessor import build_match_features

    X_list, y_list = [], []
    for _, row in matches_df.iterrows():
        try:
            features = build_match_features(
                home_form     = row.get("home_form", []),
                away_form     = row.get("away_form", []),
                home_stats    = row.get("home_stats", {}),
                away_stats    = row.get("away_stats", {}),
                h2h           = row.get("h2h", {}),
                home_xg       = row.get("home_xg", {}),
                away_xg       = row.get("away_xg", {}),
                home_fatigue  = row.get("home_fatigue", 0),
                away_fatigue  = row.get("away_fatigue", 0),
                home_injuries = row.get("home_injuries", 0),
                away_injuries = row.get("away_injuries", 0),
            )
            hg = row["home_goals"]
            ag = row["away_goals"]
            label = 2 if hg > ag else (1 if hg == ag else 0)
            X_list.append(features.flatten())
            y_list.append(label)
        except Exception as e:
            log.warning("Skip row error: %s", e)

    if not X_list:
        return np.array([]), np.array([])
    return np.array(X_list), np.array(y_list)


# Instance global
xgb_model = FootballXGBoost()
xgb_model.load()  # coba load jika sudah pernah ditraining
