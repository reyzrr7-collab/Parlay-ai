#!/usr/bin/env python3
"""
Parlay AI - Main Entry Point
Sistem agentic untuk prediksi parlay sepak bola
"""
import argparse
import sys
import os
from dotenv import load_dotenv

load_dotenv()


def cmd_init():
    """Inisialisasi database."""
    print("Inisialisasi database...")
    from database.models import init_db
    init_db()
    print("✅ Database siap.")


def cmd_run():
    """Jalankan analisis hari ini."""
    print("Menjalankan analisis harian...")
    from agent.graph import run_agent
    result = run_agent()
    print(result)


def cmd_schedule():
    """Jalankan scheduler."""
    print("Menjalankan scheduler...")
    from scheduler import run
    run()


def cmd_evaluate():
    """Evaluasi prediksi kemarin."""
    from evaluation.tracker import evaluate_yesterday_predictions
    from evaluation.metrics import print_summary
    evaluate_yesterday_predictions()
    print_summary()


def cmd_test():
    """Test koneksi semua API."""
    print("Testing koneksi API...")

    # Test NVIDIA
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model="nvidia/llama-3.1-nemotron-70b-instruct",
            openai_api_base="https://integrate.api.nvidia.com/v1",
            openai_api_key=os.getenv("NGC_API_KEY"),
            max_tokens=50
        )
        llm.invoke("Hello")
        print("✅ NVIDIA API OK")
    except Exception as e:
        print(f"❌ NVIDIA API Error: {e}")

    # Test API-Football
    try:
        from data.collector import get_today_fixtures
        fixtures = get_today_fixtures()
        print(f"✅ API-Football OK ({len(fixtures)} fixtures today)")
    except Exception as e:
        print(f"❌ API-Football Error: {e}")

    # Test OddsAPI
    try:
        from data.odds import get_match_odds
        odds = get_match_odds()
        print(f"✅ OddsAPI OK ({len(odds)} matches)")
    except Exception as e:
        print(f"❌ OddsAPI Error: {e}")

    # Test Database
    try:
        from database.models import Session
        session = Session()
        session.execute("SELECT 1")
        session.close()
        print("✅ PostgreSQL OK")
    except Exception as e:
        print(f"❌ PostgreSQL Error: {e}")

    # Test Redis
    try:
        import redis
        r = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379))
        )
        r.ping()
        print("✅ Redis OK")
    except Exception as e:
        print(f"❌ Redis Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Parlay AI - Football Prediction System")
    parser.add_argument("command", choices=["init", "run", "schedule", "evaluate", "test"],
                        help="Command yang dijalankan")
    args = parser.parse_args()

    commands = {
        "init": cmd_init,
        "run": cmd_run,
        "schedule": cmd_schedule,
        "evaluate": cmd_evaluate,
        "test": cmd_test
    }

    commands[args.command]()


if __name__ == "__main__":
    main()
