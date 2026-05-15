"""
database/models.py
─────────────────
Inisialisasi semua tabel SQLite untuk:
- Memory percakapan agent
- Profil & fakta user
- Log prediksi (feedback loop)
- Log parlay
"""

import os
import logging
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("database")

# ── Auto-detect: PostgreSQL (Railway) atau SQLite (lokal) ────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "")
USE_POSTGRES  = DATABASE_URL.startswith("postgresql") or DATABASE_URL.startswith("postgres")


def get_conn():
    """Return koneksi DB — otomatis pilih PostgreSQL atau SQLite."""
    if USE_POSTGRES:
        import psycopg2
        url = DATABASE_URL.replace("postgres://", "postgresql://", 1)
        return psycopg2.connect(url)
    else:
        import sqlite3
        return sqlite3.connect("agent_memory.db")


def _serial(use_postgres: bool) -> str:
    return "SERIAL PRIMARY KEY" if use_postgres else "INTEGER PRIMARY KEY AUTOINCREMENT"

def _now(use_postgres: bool) -> str:
    return "NOW()" if use_postgres else "CURRENT_TIMESTAMP"


def init_db() -> None:
    conn = get_conn()
    c = conn.cursor()
    s = _serial(USE_POSTGRES)

    c.execute(f'''
        CREATE TABLE IF NOT EXISTS conversation_history (
            id          {s},
            session_id  TEXT        NOT NULL,
            role        TEXT        NOT NULL,
            content     TEXT        NOT NULL,
            timestamp   TIMESTAMP   DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    c.execute(f'''
        CREATE TABLE IF NOT EXISTS user_context (
            session_id  TEXT PRIMARY KEY,
            user_name   TEXT,
            preferences TEXT,
            last_active TIMESTAMP
        )
    ''')

    c.execute(f'''
        CREATE TABLE IF NOT EXISTS user_facts (
            id          {s},
            session_id  TEXT NOT NULL,
            fact_key    TEXT NOT NULL,
            fact_value  TEXT NOT NULL,
            source      TEXT,
            updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(session_id, fact_key)
        )
    ''')

    c.execute(f'''
        CREATE TABLE IF NOT EXISTS prediction_log (
            id               {s},
            match_id         TEXT,
            match_name       TEXT NOT NULL,
            league           TEXT,
            prediction       TEXT NOT NULL,
            pick_type        TEXT,
            confidence       FLOAT,
            model_prob_home  FLOAT,
            model_prob_draw  FLOAT,
            model_prob_away  FLOAT,
            dixon_home       FLOAT,
            dixon_draw       FLOAT,
            dixon_away       FLOAT,
            xgb_home         FLOAT,
            xgb_draw         FLOAT,
            xgb_away         FLOAT,
            pinnacle_prob    FLOAT,
            edge             FLOAT,
            kelly_fraction   FLOAT,
            actual_result    TEXT,
            was_correct      INTEGER,
            brier_score      FLOAT,
            predicted_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            evaluated_at     TIMESTAMP
        )
    ''')

    c.execute(f'''
        CREATE TABLE IF NOT EXISTS parlay_log (
            id              {s},
            picks_json      TEXT NOT NULL,
            total_legs      INTEGER,
            cum_probability FLOAT,
            all_correct     INTEGER,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            evaluated_at    TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()
    mode = "PostgreSQL" if USE_POSTGRES else "SQLite"
    log.info("✅ Database diinisialisasi (%s) — 5 tabel.", mode)
