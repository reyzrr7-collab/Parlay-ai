from models.dixon_coles import dixon_coles_predict
from models.bayesian import bayesian_predict
from models.xgboost_model import xgboost_predict


WEIGHTS = {
    "dixon_coles": 0.40,
    "bayesian": 0.30,
    "xgboost": 0.30
}


def ensemble_predict(match_data: dict) -> dict:
    """Gabungkan semua model dengan weighted average."""
    dc = dixon_coles_predict(match_data)
    bayes = bayesian_predict(match_data)
    xgb = xgboost_predict(match_data)

    w_dc = WEIGHTS["dixon_coles"]
    w_b = WEIGHTS["bayesian"]
    w_x = WEIGHTS["xgboost"]

    home_win = round(dc["home_win"] * w_dc + bayes["home_win"] * w_b + xgb["home_win"] * w_x, 4)
    draw = round(dc["draw"] * w_dc + bayes["draw"] * w_b + xgb["draw"] * w_x, 4)
    away_win = round(dc["away_win"] * w_dc + bayes["away_win"] * w_b + xgb["away_win"] * w_x, 4)

    total = home_win + draw + away_win
    home_win = round(home_win / total, 4)
    draw = round(draw / total, 4)
    away_win = round(away_win / total, 4)

    predicted_outcome = max(
        [("home_win", home_win), ("draw", draw), ("away_win", away_win)],
        key=lambda x: x[1]
    )[0]

    return {
        "home_win": home_win,
        "draw": draw,
        "away_win": away_win,
        "predicted_outcome": predicted_outcome,
        "confidence": max(home_win, draw, away_win),
        "breakdown": {
            "dixon_coles": dc,
            "bayesian": bayes,
            "xgboost": xgb
        }
    }
