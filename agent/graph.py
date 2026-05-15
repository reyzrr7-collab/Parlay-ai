"""
agent/graph.py
──────────────
LangGraph Multi-Agent Workflow:

Orchestrator Agent
    ├── Data Agent       → kumpul semua data
    ├── Model Agent      → jalankan Dixon-Coles + XGBoost + Bayesian
    ├── Odds Agent       → analisis odds + value bet
    ├── News Agent       → cari berita terkini via Tavily
    └── Parlay Agent     → build kombinasi parlay optimal

Catatan: install langchain-openai dan langgraph
pip install langchain langchain-openai langgraph
"""

import os
import logging
from typing import TypedDict, List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("graph")


# ── State yang dibagikan antar agent ─────────────────────────────────────────

class ParlayState(TypedDict):
    matches:        List[dict]       # input: list pertandingan
    raw_data:       Dict[str, Any]   # output data agent
    model_results:  Dict[str, Any]   # output model agent
    odds_analysis:  Dict[str, Any]   # output odds agent
    news_context:   Dict[str, Any]   # output news agent
    final_parlay:   Dict[str, Any]   # output parlay agent
    errors:         List[str]


# ── Setup LLM ─────────────────────────────────────────────────────────────────

def get_llm():
    try:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="nvidia/nemotron-3-super-120b-a12b",
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.getenv("NGC_API_KEY", ""),
            temperature=0.6,
            model_kwargs={"top_p": 0.95},
        )
    except ImportError:
        log.error("langchain-openai tidak terinstall. pip install langchain-openai")
        return None


# ── Node 1: Data Agent ────────────────────────────────────────────────────────

def data_agent(state: ParlayState) -> dict:
    """Kumpulkan semua data untuk setiap pertandingan."""
    from data.collector import (
        get_team_form, get_head_to_head,
        get_injuries, get_fatigue_score
    )
    from data.scraper import get_understat_team_stats
    from data.odds import get_best_odds_for_match

    raw_data = {}
    errors   = list(state.get("errors", []))

    for match in state["matches"]:
        mid = match.get("id", f"{match.get('home')}_vs_{match.get('away')}")
        try:
            raw_data[mid] = {
                "home_form":     get_team_form(match.get("home_id", 0)),
                "away_form":     get_team_form(match.get("away_id", 0)),
                "h2h":           get_head_to_head(
                                     match.get("home_id", 0),
                                     match.get("away_id", 0)
                                 ),
                "home_injuries": get_injuries(match.get("home_id", 0)),
                "away_injuries": get_injuries(match.get("away_id", 0)),
                "home_fatigue":  get_fatigue_score(match.get("home_id", 0)),
                "away_fatigue":  get_fatigue_score(match.get("away_id", 0)),
                "home_xg":       get_understat_team_stats(
                                     match.get("home", ""),
                                     match.get("league", "EPL")
                                 ),
                "away_xg":       get_understat_team_stats(
                                     match.get("away", ""),
                                     match.get("league", "EPL")
                                 ),
                "match_info":    match,
            }
            log.info("✅ Data terkumpul: %s vs %s",
                     match.get("home"), match.get("away"))
        except Exception as e:
            err = f"Data error [{mid}]: {e}"
            log.error(err)
            errors.append(err)

    return {"raw_data": raw_data, "errors": errors}


# ── Node 2: Model Agent ───────────────────────────────────────────────────────

def model_agent(state: ParlayState) -> dict:
    """Jalankan ensemble model untuk setiap pertandingan."""
    from models.ensemble import ensemble_predict
    from data.preprocessor import build_match_features

    model_results = {}

    for mid, data in state["raw_data"].items():
        try:
            match    = data["match_info"]
            home_xg  = data["home_xg"].get("xg_avg", 1.4) or 1.4
            away_xg  = data["away_xg"].get("xg_avg", 1.1) or 1.1

            features = build_match_features(
                home_form     = data["home_form"],
                away_form     = data["away_form"],
                home_stats    = {},
                away_stats    = {},
                h2h           = data["h2h"],
                home_xg       = data["home_xg"],
                away_xg       = data["away_xg"],
                home_fatigue  = data["home_fatigue"],
                away_fatigue  = data["away_fatigue"],
                home_injuries = len(data["home_injuries"]),
                away_injuries = len(data["away_injuries"]),
            )

            result = ensemble_predict(
                home_team  = match.get("home", ""),
                away_team  = match.get("away", ""),
                features   = features,
                home_xg    = home_xg,
                away_xg    = away_xg,
            )
            model_results[mid] = result
            log.info("✅ Model selesai: %s vs %s — %s (%.0f%%)",
                     match.get("home"), match.get("away"),
                     result.get("prediction"), result.get("confidence", 0))

        except Exception as e:
            log.error("Model error [%s]: %s", mid, e)
            model_results[mid] = {
                "prediction": "N/A", "confidence": 0,
                "home_win": 33, "draw": 33, "away_win": 34
            }

    return {"model_results": model_results}


# ── Node 3: Odds Agent ────────────────────────────────────────────────────────

def odds_agent(state: ParlayState) -> dict:
    """Analisis odds dan deteksi value bet."""
    from parlay.value_bet import analyze_value
    from data.odds import get_best_odds_for_match

    odds_analysis = {}

    for mid, model_res in state["model_results"].items():
        data  = state["raw_data"].get(mid, {})
        match = data.get("match_info", {})
        try:
            odds_data = get_best_odds_for_match(
                match.get("home", ""),
                match.get("away", ""),
                match.get("league", "EPL")
            )
            market_odds = odds_data.get("avg_odds", {})

            pred  = model_res.get("prediction", "")
            prob  = max(
                model_res.get("home_win", 0),
                model_res.get("draw", 0),
                model_res.get("away_win", 0)
            ) / 100

            value = analyze_value(pred, prob, market_odds)
            odds_analysis[mid] = {
                "market_odds":    market_odds,
                "value_analysis": value,
                "odds_raw":       odds_data,
            }
            log.info("📊 Odds: %s | Edge: %.1f%%",
                     mid, value.get("edge", 0))

        except Exception as e:
            log.error("Odds error [%s]: %s", mid, e)
            odds_analysis[mid] = {"market_odds": {}, "value_analysis": {}}

    return {"odds_analysis": odds_analysis}


# ── Node 4: News Agent ────────────────────────────────────────────────────────

def news_agent(state: ParlayState) -> dict:
    """Cari berita terkini untuk setiap pertandingan via Tavily."""
    import os
    news_context = {}
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY", ""))

        for mid, data in state["raw_data"].items():
            match = data.get("match_info", {})
            query = (f"{match.get('home', '')} vs {match.get('away', '')} "
                     f"injury lineup news {match.get('league', '')}")
            try:
                results = client.search(query=query, max_results=3)
                snippets = [
                    r.get("content", "")[:200]
                    for r in results.get("results", [])
                ]
                news_context[mid] = " | ".join(snippets)
            except Exception as e:
                log.warning("Tavily error [%s]: %s", mid, e)
                news_context[mid] = ""

    except ImportError:
        log.warning("Tavily tidak terinstall — news agent dilewati.")

    return {"news_context": news_context}


# ── Node 5: Parlay Agent ──────────────────────────────────────────────────────

def parlay_agent(state: ParlayState) -> dict:
    """Build rekomendasi parlay dari semua hasil agent sebelumnya."""
    from parlay.generator import generate_parlay

    candidates = []
    for mid, model_res in state["model_results"].items():
        data       = state["raw_data"].get(mid, {})
        odds_res   = state["odds_analysis"].get(mid, {})
        news       = state["news_context"].get(mid, "")
        match      = data.get("match_info", {})
        value      = odds_res.get("value_analysis", {})
        market_odds= odds_res.get("market_odds", {})

        pred = model_res.get("prediction", "")
        conf = model_res.get("confidence", 0)

        # Tentukan odds sesuai prediksi
        if "Home" in pred:
            odds = market_odds.get("home", 1.5)
        elif "Away" in pred:
            odds = market_odds.get("away", 2.5)
        else:
            odds = market_odds.get("draw", 3.2)

        candidates.append({
            "match_name":     f"{match.get('home','')} vs {match.get('away','')}",
            "prediction":     pred,
            "confidence":     conf,
            "odds":           odds or 1.5,
            "edge":           value.get("edge", 0),
            "kelly":          value.get("kelly", 0),
            "home_injuries":  len(data.get("home_injuries", [])),
            "away_injuries":  len(data.get("away_injuries", [])),
            "odds_movement":  0,
            "news_context":   news[:200],
        })

    final = generate_parlay(candidates)
    log.info("🎯 Parlay: %s | %d legs | %.1f%%",
             final.get("status"), final.get("total_legs", 0),
             final.get("cum_probability", 0))

    return {"final_parlay": final}


# ── Compile Graph ─────────────────────────────────────────────────────────────

def build_graph():
    """Bangun dan compile LangGraph workflow."""
    try:
        from langgraph.graph import StateGraph, END

        workflow = StateGraph(ParlayState)

        workflow.add_node("data_agent",   data_agent)
        workflow.add_node("model_agent",  model_agent)
        workflow.add_node("odds_agent",   odds_agent)
        workflow.add_node("news_agent",   news_agent)
        workflow.add_node("parlay_agent", parlay_agent)

        workflow.set_entry_point("data_agent")
        workflow.add_edge("data_agent",   "model_agent")
        workflow.add_edge("model_agent",  "odds_agent")
        workflow.add_edge("odds_agent",   "news_agent")
        workflow.add_edge("news_agent",   "parlay_agent")
        workflow.add_edge("parlay_agent", END)

        app = workflow.compile()
        log.info("✅ LangGraph compiled — 5 agents")
        return app

    except ImportError:
        log.error("LangGraph tidak terinstall. pip install langgraph langchain-openai")
        return None


def run_parlay_graph(matches: list) -> dict:
    """
    Jalankan full multi-agent workflow untuk list pertandingan.

    matches = [
        {
            "id": "match_001",
            "home": "Liverpool", "home_id": 40,
            "away": "Arsenal",   "away_id": 42,
            "league": "EPL",
        },
        ...
    ]
    """
    app = build_graph()
    if not app:
        log.error("Graph tidak bisa dibuild.")
        return {}

    initial_state: ParlayState = {
        "matches":       matches,
        "raw_data":      {},
        "model_results": {},
        "odds_analysis": {},
        "news_context":  {},
        "final_parlay":  {},
        "errors":        [],
    }

    result = app.invoke(initial_state)
    return result.get("final_parlay", {})
