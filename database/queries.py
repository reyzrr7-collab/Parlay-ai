from database.models import Session, Match, MatchData, Prediction, Parlay, EvaluationLog
from datetime import datetime


def save_match(api_match_id, home_team, away_team, league, match_date):
    session = Session()
    try:
        existing = session.query(Match).filter_by(api_match_id=api_match_id).first()
        if existing:
            return existing.id
        match = Match(
            api_match_id=api_match_id,
            home_team=home_team,
            away_team=away_team,
            league=league,
            match_date=match_date
        )
        session.add(match)
        session.commit()
        return match.id
    finally:
        session.close()


def save_match_data(match_id, data: dict):
    session = Session()
    try:
        md = MatchData(match_id=match_id, **data)
        session.add(md)
        session.commit()
    finally:
        session.close()


def save_prediction(match_id, prediction: dict):
    session = Session()
    try:
        pred = Prediction(match_id=match_id, **prediction)
        session.add(pred)
        session.commit()
        return pred.id
    finally:
        session.close()


def save_parlay(parlay: dict):
    session = Session()
    try:
        p = Parlay(**parlay)
        session.add(p)
        session.commit()
        return p.id
    finally:
        session.close()


def get_today_matches():
    session = Session()
    try:
        today = datetime.utcnow().date()
        return session.query(Match).filter(
            Match.match_date >= datetime.combine(today, datetime.min.time()),
            Match.status == "scheduled"
        ).all()
    finally:
        session.close()


def get_prediction_by_match(match_id):
    session = Session()
    try:
        return session.query(Prediction).filter_by(match_id=match_id).first()
    finally:
        session.close()


def log_evaluation(prediction_id, actual_outcome, correct, brier_score):
    session = Session()
    try:
        log = EvaluationLog(
            prediction_id=prediction_id,
            actual_outcome=actual_outcome,
            correct=correct,
            brier_score=brier_score
        )
        session.add(log)
        session.commit()
    finally:
        session.close()


def get_all_predictions(limit=100):
    session = Session()
    try:
        return session.query(Prediction).order_by(Prediction.created_at.desc()).limit(limit).all()
    finally:
        session.close()
