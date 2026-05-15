"""
database/queries.py
───────────────────
Semua operasi CRUD ke SQLite:
- Memory percakapan
- User context & facts
- Prediksi log & evaluasi
- Parlay log
"""

import json
import logging
from datetime import datetime
from typing import Optional, List, Dict
from database.models import get_conn, USE_POSTGRES

log = logging.getLogger("queries")

def _ph(use_postgres: bool) -> str:
    """Placeholder — %s untuk postgres, ? untuk sqlite."""
    return "%s" if use_postgres else "?"

PH = _ph(USE_POSTGRES)


# ── Memory ───────────────────────────────────────────────────────────────────

def save_message(session_id: str, role: str, content: str) -> None:
    conn = get_conn()
    conn.cursor().execute(
        f"INSERT INTO conversation_history (session_id, role, content) VALUES ({PH},{PH},{PH})",
        (session_id, role, content)
    )
    conn.commit()
    conn.close()


def load_recent_messages(session_id: str, limit: int = 10) -> str:
    conn = get_conn()
    c = conn.cursor()

    c.execute(
        f"SELECT fact_key, fact_value FROM user_facts WHERE session_id = {PH} ORDER BY updated_at DESC",
        (session_id,)
    )
    facts = c.fetchall()

    c.execute(
        f"""SELECT role, content FROM conversation_history
            WHERE session_id = {PH} ORDER BY timestamp DESC LIMIT {PH}""",
        (session_id, limit)
    )
    rows = c.fetchall()
    conn.close()

    parts = []
    if facts:
        lines = "\n".join(f"  • {k}: {v}" for k, v in facts)
        parts.append(f"[FAKTA KUNCI USER]\n{lines}")
    if rows:
        lines = "\n".join(f"{r.capitalize()}: {c}" for r, c in reversed(rows))
        parts.append(f"[RIWAYAT PERCAKAPAN]\n{lines}")
    return "\n\n".join(parts)


def get_user_context(session_id: str) -> dict:
    conn = get_conn()
    row = conn.cursor().execute(
        f"SELECT user_name, preferences FROM user_context WHERE session_id = {PH}",
        (session_id,)
    ).fetchone()
    conn.close()
    return {"user_name": row[0], "preferences": row[1]} if row else {"user_name": None, "preferences": None}


def upsert_user_fact(session_id: str, key: str, value: str, source: str = "") -> None:
    conn = get_conn()
    if USE_POSTGRES:
        conn.cursor().execute(
            f"""INSERT INTO user_facts (session_id, fact_key, fact_value, source, updated_at)
                VALUES ({PH},{PH},{PH},{PH},{PH})
                ON CONFLICT(session_id, fact_key)
                DO UPDATE SET fact_value=EXCLUDED.fact_value,
                              source=EXCLUDED.source,
                              updated_at=EXCLUDED.updated_at""",
            (session_id, key, value, source, datetime.now())
        )
    else:
        conn.cursor().execute(
            f"""INSERT INTO user_facts (session_id, fact_key, fact_value, source, updated_at)
                VALUES ({PH},{PH},{PH},{PH},{PH})
                ON CONFLICT(session_id, fact_key)
                DO UPDATE SET fact_value=excluded.fact_value,
                              source=excluded.source,
                              updated_at=excluded.updated_at""",
            (session_id, key, value, source, datetime.now())
        )
    conn.commit()
    conn.close()


def upsert_user_name(session_id: str, name: str) -> None:
    conn = get_conn()
    if USE_POSTGRES:
        conn.cursor().execute(
            f"""INSERT INTO user_context (session_id, user_name, last_active)
                VALUES ({PH},{PH},{PH})
                ON CONFLICT(session_id)
                DO UPDATE SET user_name=EXCLUDED.user_name, last_active=EXCLUDED.last_active""",
            (session_id, name, datetime.now())
        )
    else:
        conn.cursor().execute(
            f"""INSERT INTO user_context (session_id, user_name, last_active)
                VALUES ({PH},{PH},{PH})
                ON CONFLICT(session_id)
                DO UPDATE SET user_name=excluded.user_name, last_active=excluded.last_active""",
            (session_id, name, datetime.now())
        )
    conn.commit()
    conn.close()


def show_user_facts(session_id: str) -> None:
    conn = get_conn()
    rows = conn.cursor().execute(
        f"SELECT fact_key, fact_value, updated_at FROM user_facts WHERE session_id = {PH} ORDER BY updated_at DESC",
        (session_id,)
    ).fetchall()
    conn.close()
    if not rows:
        print("(Belum ada fakta tersimpan)")
        return
    print("\n📋 Fakta User Tersimpan:")
    print("─" * 50)
    for k, v, ts in rows:
        print(f"  {k:<20} → {v}  [{ts}]")
    print("─" * 50)


# ── Prediksi Log ─────────────────────────────────────────────────────────────

def save_prediction(
    match_name: str, prediction: str, confidence: float,
    probs: dict, dixon: dict, xgb: dict,
    pinnacle_prob: float, edge: float, kelly: float,
    match_id: str = "", league: str = "", pick_type: str = "1X2"
) -> int:
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        f"""INSERT INTO prediction_log
            (match_id, match_name, league, prediction, pick_type, confidence,
             model_prob_home, model_prob_draw, model_prob_away,
             dixon_home, dixon_draw, dixon_away,
             xgb_home, xgb_draw, xgb_away,
             pinnacle_prob, edge, kelly_fraction)
            VALUES ({','.join([PH]*18)})""",
        (match_id, match_name, league, prediction, pick_type, confidence,
         probs.get("home",0), probs.get("draw",0), probs.get("away",0),
         dixon.get("home",0), dixon.get("draw",0), dixon.get("away",0),
         xgb.get("home",0), xgb.get("draw",0), xgb.get("away",0),
         pinnacle_prob, edge, kelly)
    )
    if USE_POSTGRES:
        pred_id = c.fetchone()[0] if c.rowcount else 0
    else:
        pred_id = c.lastrowid
    conn.commit()
    conn.close()
    return pred_id


def evaluate_prediction(pred_id: int, actual_result: str) -> dict:
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        f"SELECT prediction, model_prob_home, model_prob_draw, model_prob_away FROM prediction_log WHERE id={PH}",
        (pred_id,)
    )
    row = c.fetchone()
    if not row:
        conn.close()
        return {}
    pred, ph, pd_v, pa = row
    was_correct = 1 if pred.lower() in actual_result.lower() else 0
    outcome_map = {"home": [1,0,0], "draw": [0,1,0], "away": [0,0,1]}
    actual_vec = outcome_map.get(actual_result.lower(), [0,0,0])
    pred_vec   = [ph or 0, pd_v or 0, pa or 0]
    brier = sum((p - a)**2 for p, a in zip(pred_vec, actual_vec)) / 3
    c.execute(
        f"UPDATE prediction_log SET actual_result={PH}, was_correct={PH}, brier_score={PH}, evaluated_at={PH} WHERE id={PH}",
        (actual_result, was_correct, round(brier,4), datetime.now(), pred_id)
    )
    conn.commit()
    conn.close()
    return {"was_correct": was_correct, "brier_score": round(brier, 4)}


def get_accuracy_stats() -> dict:
    conn = get_conn()
    row = conn.cursor().execute(
        "SELECT COUNT(*), SUM(was_correct), AVG(brier_score), AVG(edge) FROM prediction_log WHERE actual_result IS NOT NULL"
    ).fetchone()
    conn.close()
    total, correct, avg_brier, avg_edge = row
    if not total:
        return {"total": 0, "accuracy": 0, "avg_brier": 0, "avg_edge": 0}
    return {
        "total":     total,
        "accuracy":  round((correct / total) * 100, 1),
        "avg_brier": round(avg_brier or 0, 4),
        "avg_edge":  round((avg_edge or 0) * 100, 2)
    }


# ── Parlay Log ────────────────────────────────────────────────────────────────

def save_parlay(picks: list, cum_prob: float) -> int:
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        f"INSERT INTO parlay_log (picks_json, total_legs, cum_probability) VALUES ({PH},{PH},{PH})",
        (json.dumps(picks), len(picks), cum_prob)
    )
    pid = c.fetchone()[0] if USE_POSTGRES else c.lastrowid
    conn.commit()
    conn.close()
    return pid


def evaluate_parlay(parlay_id: int, all_correct: bool) -> None:
    conn = get_conn()
    conn.cursor().execute(
        f"UPDATE parlay_log SET all_correct={PH}, evaluated_at={PH} WHERE id={PH}",
        (1 if all_correct else 0, datetime.now(), parlay_id)
    )
    conn.commit()
    conn.close()
