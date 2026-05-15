from database.queries import log_evaluation, get_all_predictions, save_match
from data.collector import get_today_fixtures
from datetime import datetime, timedelta
import requests
import os

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
HEADERS = {
    "X-RapidAPI-Key": RAPIDAPI_KEY,
    "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
}


def get_match_result(fixture_id: int) -> dict:
    """Ambil hasil pertandingan yang sudah selesai."""
    url = "https://api-football-v1.p.rapidapi.com/v3/fixtures"
    params = {"id": fixture_id}
    try:
        res = requests.get(url, headers=HEADERS, params=params, timeout=10)
        data = res.json().get("response", [])
        if data:
            match = data[0]
            status = match["fixture"]["status"]["short"]
            if status == "FT":
                goals = match["goals"]
                return {
                    "status": "finished",
                    "home_score": goals["home"],
                    "away_score": goals["away"],
                    "outcome": "home_win" if goals["home"] > goals["away"]
                               else ("draw" if goals["home"] == goals["away"] else "away_win")
                }
    except Exception as e:
        print(f"Error fetching result: {e}")
    return {"status": "not_finished"}


def evaluate_yesterday_predictions():
    """Evaluasi semua prediksi kemarin."""
    from database.models import Session, Prediction, Match
    session = Session()
    try:
        yesterday = datetime.utcnow().date() - timedelta(days=1)
        predictions = session.query(Prediction).join(
            Match, Prediction.match_id == Match.id
        ).filter(
            Match.match_date >= datetime.combine(yesterday, datetime.min.time()),
            Match.match_date < datetime.combine(datetime.utcnow().date(), datetime.min.time())
        ).all()

        for pred in predictions:
            match = session.query(Match).get(pred.match_id)
            if not match:
                continue
            result = get_match_result(match.api_match_id)
            if result["status"] != "finished":
                continue

            actual = result["outcome"]
            correct = (pred.predicted_outcome == actual)
            brier = brier_score(pred, actual)

            log_evaluation(pred.id, actual, correct, brier)
            print(f"Evaluated: {match.home_team} vs {match.away_team} → {actual} | Correct: {correct}")
    finally:
        session.close()


def brier_score(prediction, actual_outcome: str) -> float:
    """Hitung Brier Score untuk prediksi."""
    outcome_map = {"home_win": 0, "draw": 1, "away_win": 2}
    probs = [prediction.home_win_prob, prediction.draw_prob, prediction.away_win_prob]
    actual_idx = outcome_map.get(actual_outcome, 0)
    actuals = [1 if i == actual_idx else 0 for i in range(3)]
    bs = sum((p - a) ** 2 for p, a in zip(probs, actuals)) / 3
    return round(bs, 4)
