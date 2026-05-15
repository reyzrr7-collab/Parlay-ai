import schedule
import time
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_daily_analysis():
    """Jalankan analisis harian lengkap."""
    logger.info("=== Memulai analisis harian ===")
    try:
        from agent.graph import run_agent
        from database.queries import save_parlay
        from parlay.generator import print_parlay

        result = run_agent()
        logger.info("Agent selesai berjalan.")
        logger.info(result[:500])
    except Exception as e:
        logger.error(f"Error dalam analisis harian: {e}")


def run_evaluation():
    """Evaluasi prediksi kemarin."""
    logger.info("=== Memulai evaluasi prediksi kemarin ===")
    try:
        from evaluation.tracker import evaluate_yesterday_predictions
        from evaluation.metrics import print_summary
        from evaluation.retrainer import auto_retrain

        evaluate_yesterday_predictions()
        print_summary()
        auto_retrain()
    except Exception as e:
        logger.error(f"Error dalam evaluasi: {e}")


def setup_schedule():
    """Setup jadwal cron."""
    # Analisis harian 2 jam sebelum kickoff pertama (biasanya 13:00 UTC untuk EPL)
    schedule.every().day.at("11:00").do(run_daily_analysis)

    # Evaluasi tiap pagi
    schedule.every().day.at("06:00").do(run_evaluation)

    logger.info("Scheduler started:")
    logger.info("  - Analisis harian: 11:00 UTC (2 jam sebelum kickoff)")
    logger.info("  - Evaluasi: 06:00 UTC")


def run():
    """Main loop scheduler."""
    setup_schedule()
    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    run()
