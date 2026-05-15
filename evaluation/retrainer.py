"""
evaluation/retrainer.py
───────────────────────
Auto retrain XGBoost dari data prediksi yang sudah dievaluasi.
Dijalankan otomatis setiap 2 minggu oleh scheduler.
"""

import sqlite3
import logging
import numpy as np
import pandas as pd
from database.models import DB_PATH
from models.xgboost_model import xgb_model

log = logging.getLogger("retrainer")

MIN_SAMPLES_TO_RETRAIN = 50


def retrain_if_enough_data() -> dict:
    """
    Cek jumlah data — retrain jika sudah cukup.
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT * FROM prediction_log WHERE actual_result IS NOT NULL",
        conn
    )
    conn.close()

    n = len(df)
    if n < MIN_SAMPLES_TO_RETRAIN:
        msg = f"Data belum cukup ({n}/{MIN_SAMPLES_TO_RETRAIN} sampel)"
        log.info(msg)
        return {"status": "skip", "message": msg, "samples": n}

    log.info("🔄 Retrain dimulai dengan %d sampel...", n)

    # Bangun feature matrix dari kolom yang tersedia
    feature_cols = [
        "model_prob_home", "model_prob_draw", "model_prob_away",
        "confidence", "edge", "pinnacle_prob",
        "dixon_home", "dixon_draw", "dixon_away",
    ]
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].fillna(0).values

    label_map = {"home": 2, "draw": 1, "away": 0}
    y = df["actual_result"].map(label_map).dropna().values

    if len(y) != len(X):
        X = X[:len(y)]

    stats = xgb_model.train(X, y)
    log.info("✅ Retrain selesai: %s", stats)
    return {"status": "retrained", "samples": n, **stats}
