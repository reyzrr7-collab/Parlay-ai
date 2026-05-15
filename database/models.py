from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

DB_URL = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"

engine = create_engine(DB_URL)
Session = sessionmaker(bind=engine)
Base = declarative_base()


class Match(Base):
    __tablename__ = "matches"

    id = Column(Integer, primary_key=True)
    api_match_id = Column(Integer, unique=True)
    home_team = Column(String)
    away_team = Column(String)
    league = Column(String)
    match_date = Column(DateTime)
    home_score = Column(Integer, nullable=True)
    away_score = Column(Integer, nullable=True)
    status = Column(String, default="scheduled")
    created_at = Column(DateTime, default=datetime.utcnow)


class MatchData(Base):
    __tablename__ = "match_data"

    id = Column(Integer, primary_key=True)
    match_id = Column(Integer)
    home_form = Column(JSON)
    away_form = Column(JSON)
    h2h = Column(JSON)
    home_xg_avg = Column(Float)
    away_xg_avg = Column(Float)
    home_lineup = Column(JSON)
    away_lineup = Column(JSON)
    injuries = Column(JSON)
    home_odds = Column(Float)
    draw_odds = Column(Float)
    away_odds = Column(Float)
    weather = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True)
    match_id = Column(Integer)
    home_win_prob = Column(Float)
    draw_prob = Column(Float)
    away_win_prob = Column(Float)
    predicted_outcome = Column(String)
    confidence = Column(Float)
    edge = Column(Float)
    value_bet = Column(Boolean, default=False)
    model_breakdown = Column(JSON)
    agent_reasoning = Column(String)
    red_flag = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class Parlay(Base):
    __tablename__ = "parlays"

    id = Column(Integer, primary_key=True)
    match_ids = Column(JSON)
    legs = Column(Integer)
    cumulative_prob = Column(Float)
    combined_odds = Column(Float)
    selections = Column(JSON)
    result = Column(String, nullable=True)
    roi = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class EvaluationLog(Base):
    __tablename__ = "evaluation_logs"

    id = Column(Integer, primary_key=True)
    prediction_id = Column(Integer)
    actual_outcome = Column(String)
    correct = Column(Boolean)
    brier_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    Base.metadata.create_all(engine)
    print("Database tables created successfully.")


if __name__ == "__main__":
    init_db()
