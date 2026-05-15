"""
scheduler.py
────────────
Cron job otomatis:
- Setiap hari jam 09:00 → ambil jadwal pertandingan hari ini
- H-2 jam sebelum kick-off → jalankan analisis & kirim ke Telegram
- Setiap Senin → retrain XGBoost dari data terbaru
- Setiap malam → evaluasi prediksi yang match-nya sudah selesai
"""

import os
import time
import logging
import schedule
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

from database.models import init_db
from data.collector import get_fixtures_today
from agent.graph import run_parlay_graph
from evaluation.retrainer import retrain_if_enough_data
from evaluation.tracker import print_accuracy_report

load_dotenv()
log = logging.getLogger("scheduler")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

FOOTBALL_API_KEY = os.getenv("FOOTBALL_API_KEY", "")
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Liga yang dipantau
LEAGUES_TO_WATCH = [
    {"id": 39,  "name": "EPL"},
    {"id": 140, "name": "La Liga"},
    {"id": 78,  "name": "Bundesliga"},
    {"id": 135, "name": "Serie A"},
    {"id": 61,  "name": "Ligue 1"},
    {"id": 2,   "name": "Champions League"},
]


# ── Jobs ──────────────────────────────────────────────────────────────────────

def job_scan_fixtures():
    """Jam 09:00 — scan semua pertandingan hari ini."""
    log.info("🔍 Scanning fixtures hari ini...")
    all_matches = []
    for league in LEAGUES_TO_WATCH:
        fixtures = get_fixtures_today(league["id"])
        for f in fixtures:
            kickoff = f["fixture"].get("date", "")
            all_matches.append({
                "id":       str(f["fixture"]["id"]),
                "home":     f["teams"]["home"]["name"],
                "home_id":  f["teams"]["home"]["id"],
                "away":     f["teams"]["away"]["name"],
                "away_id":  f["teams"]["away"]["id"],
                "league":   league["name"],
                "kickoff":  kickoff,
            })

    log.info("📅 Total %d pertandingan hari ini", len(all_matches))

    # Schedule analisis H-2 jam sebelum kick-off
    for match in all_matches:
        _schedule_pre_match(match)


def _schedule_pre_match(match: dict):
    """Schedule analisis 2 jam sebelum kick-off."""
    try:
        kickoff = datetime.fromisoformat(
            match["kickoff"].replace("Z", "+00:00")
        )
        # Convert ke local time
        local_kickoff = kickoff.astimezone()
        analysis_time = local_kickoff - timedelta(hours=2)

        if analysis_time <= datetime.now().astimezone():
            # Sudah lewat, jalankan sekarang
            job_analyze_match(match)
            return

        time_str = analysis_time.strftime("%H:%M")
        schedule.every().day.at(time_str).do(
            job_analyze_match, match
        ).tag(f"match_{match['id']}")

        log.info("⏰ Scheduled: %s vs %s @ %s (analisis %s)",
                 match["home"], match["away"],
                 local_kickoff.strftime("%H:%M"), time_str)

    except Exception as e:
        log.error("Schedule error [%s]: %s", match.get("id"), e)


def job_analyze_match(match: dict):
    """Jalankan analisis untuk satu pertandingan dan kirim ke Telegram."""
    log.info("⚽ Analisis: %s vs %s", match["home"], match["away"])
    try:
        result = run_parlay_graph([match])
        if result:
            _kirim_notif_telegram(match, result)
    except Exception as e:
        log.error("Analisis error [%s vs %s]: %s",
                  match["home"], match["away"], e)
    # Hapus job setelah dijalankan
    schedule.clear(f"match_{match['id']}")


def job_retrain():
    """Senin 06:00 — retrain XGBoost dari data terbaru."""
    log.info("🔄 Retrain model mingguan...")
    result = retrain_if_enough_data()
    log.info("Retrain result: %s", result)


def job_evaluasi():
    """Setiap malam 23:30 — tampilkan laporan akurasi."""
    log.info("📊 Evaluasi akurasi harian...")
    print_accuracy_report()


def job_evaluasi_hasil():
    """
    Setiap malam 23:00 — cek hasil pertandingan dan update database.
    Evaluasi prediksi yang match-nya sudah selesai.
    """
    log.info("🔍 Mengecek hasil pertandingan selesai...")
    try:
        # Ambil fixture yang sudah FT hari ini
        today = datetime.now().strftime("%Y-%m-%d")
        resp = requests.get(
            "https://v3.football.api-sports.io/fixtures",
            headers={"x-apisports-key": FOOTBALL_API_KEY},
            params={"date": today, "status": "FT"},
            timeout=15
        )
        results = resp.json().get("response", [])
        log.info("%d pertandingan selesai hari ini", len(results))
        # TODO: match dengan prediction_log dan update actual_result
    except Exception as e:
        log.error("Evaluasi hasil error: %s", e)


# ── Telegram ──────────────────────────────────────────────────────────────────

def _kirim_notif_telegram(match: dict, parlay_result: dict):
    """Kirim notifikasi analisis ke Telegram."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        import asyncio
        from telegram import Bot
        from telegram.constants import ParseMode

        picks = parlay_result.get("picks", [])
        pick_lines = "\n".join(
            f"  {i+1}. {p.get('match_name', '')} → *{p.get('prediction', '')}* "
            f"({p.get('confidence', 0)}%) @ {p.get('odds', 0):.2f}"
            for i, p in enumerate(picks)
        )

        msg = f"""⚽ *PRE-MATCH ALERT*
🏟️ {match['home']} vs {match['away']}
🏆 {match.get('league', '')}

📊 *Status* : {parlay_result.get('status', 'N/A')}
🎯 *Legs*   : {parlay_result.get('total_legs', 0)}
📈 *Odds*   : {parlay_result.get('total_odds', 0)}x
📉 *Prob*   : {parlay_result.get('cum_probability', 0)}%

{pick_lines}

_Analisis otomatis oleh Parlay AI v3.0_"""

        async def send():
            bot = Bot(token=TELEGRAM_TOKEN)
            await bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=msg,
                parse_mode=ParseMode.MARKDOWN
            )

        asyncio.run(send())
        log.info("✅ Notifikasi Telegram terkirim")

    except Exception as e:
        log.error("Telegram notif error: %s", e)


# ── Main Scheduler ────────────────────────────────────────────────────────────

def run_scheduler():
    init_db()
    log.info("🚀 Scheduler dimulai...")
    log.info("⏰ Jadwal:")
    log.info("  09:00 — Scan fixtures hari ini")
    log.info("  H-2   — Analisis pre-match otomatis")
    log.info("  Senin 06:00 — Retrain model")
    log.info("  23:00 — Evaluasi hasil pertandingan")
    log.info("  23:30 — Laporan akurasi harian")

    # Register semua jadwal
    schedule.every().day.at("09:00").do(job_scan_fixtures)
    schedule.every().monday.at("06:00").do(job_retrain)
    schedule.every().day.at("23:00").do(job_evaluasi_hasil)
    schedule.every().day.at("23:30").do(job_evaluasi)

    # Jalankan scan pertama saat startup
    job_scan_fixtures()

    # Loop utama
    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    run_scheduler()
