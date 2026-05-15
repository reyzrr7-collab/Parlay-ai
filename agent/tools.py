"""
agent/tools.py
──────────────
Semua tools yang tersedia untuk ReAct agent:
- Kalkulator
- Tavily search
- Football stats (API-Football)
- xG stats (Understat)
- Team form + home/away split
- Head-to-head
- Injuries
- Fatigue score
- FBref advanced stats
- Odds + value bet
- Ensemble model prediction
- Parlay generator
"""

import os
import json
import math
import logging
import re
import requests
import asyncio
from typing import Optional
from tavily import TavilyClient
from dotenv import load_dotenv

from data.collector import (
    get_team_form, get_team_stats, get_head_to_head,
    get_injuries, get_fixture_odds, get_fatigue_score,
)
from data.scraper import get_understat_team_stats, get_fbref_team_stats
from data.odds import get_best_odds_for_match, odds_to_prob
from data.preprocessor import build_match_features
from models.ensemble import ensemble_predict
from parlay.value_bet import analyze_value
from parlay.generator import generate_parlay

load_dotenv()
log = logging.getLogger("tools")
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY", ""))


# ── TOOL DEFINITIONS (format untuk ReAct parser) ─────────────────────────────

TOOLS = {
    "kalkulator": {
        "description": "Evaluasi ekspresi matematika. Input: ekspresi string. Contoh: '2.5 * 1.4'",
        "func": None,
    },
    "tavily_search": {
        "description": "Cari informasi terkini di web. Input: query string.",
        "func": None,
    },
    "get_team_form": {
        "description": "Ambil N pertandingan terakhir tim. Input: {'team_id': int, 'last_n': int}",
        "func": None,
    },
    "get_xg_stats": {
        "description": "Ambil statistik xG tim dari Understat. Input: {'team_name': str, 'league': str}",
        "func": None,
    },
    "get_head_to_head": {
        "description": "Histori H2H dua tim. Input: {'team1_id': int, 'team2_id': int}",
        "func": None,
    },
    "get_injuries": {
        "description": "Cek injury & suspense pemain. Input: {'team_id': int}",
        "func": None,
    },
    "get_fatigue_score": {
        "description": "Hitung fatigue score berdasarkan kepadatan jadwal. Input: {'team_id': int}",
        "func": None,
    },
    "get_fbref_stats": {
        "description": "Advanced stats dari FBref (PPDA, xG, dll). Input: {'team_name': str, 'league': str}",
        "func": None,
    },
    "get_odds": {
        "description": "Ambil odds terbaik untuk pertandingan. Input: {'home_team': str, 'away_team': str, 'sport': str}",
        "func": None,
    },
    "run_ensemble_model": {
        "description": "Jalankan ensemble model (Dixon-Coles + XGBoost). Input: {'home_team': str, 'away_team': str, 'home_xg': float, 'away_xg': float}",
        "func": None,
    },
    "analyze_value_bet": {
        "description": "Analisis value bet & Kelly Criterion. Input: {'prediction': str, 'model_prob': float, 'market_odds': dict}",
        "func": None,
    },
    "generate_parlay": {
        "description": "Generate rekomendasi parlay dari list pertandingan. Input: {'matches': list}",
        "func": None,
    },
}


def get_tools_description() -> str:
    lines = ["Tools yang tersedia:"]
    for name, info in TOOLS.items():
        lines.append(f"- {name}: {info['description']}")
    return "\n".join(lines)


# ── Implementasi Tools ────────────────────────────────────────────────────────

def tool_kalkulator(expression: str) -> str:
    allowed = re.compile(r'^[\d\s\+\-\*\/\(\)\.\%\*\*]+$')
    if not allowed.match(expression.strip()):
        return "Error: ekspresi tidak diizinkan"
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(round(result, 6))
    except ZeroDivisionError:
        return "Error: division by zero"
    except Exception as e:
        return f"Error: {e}"


def tool_tavily_search(query: str) -> str:
    try:
        results = tavily.search(query=query, max_results=3)
        if not results or not results.get("results"):
            return "Tidak ada hasil ditemukan."
        snippets = []
        for r in results["results"][:3]:
            snippets.append(f"[{r.get('title', '')}]\n{r.get('content', '')[:300]}")
        return "\n\n".join(snippets)
    except Exception as e:
        log.error("Tavily error: %s", e)
        return f"Search error: {e}"


def tool_get_team_form(team_id: int, last_n: int = 10) -> str:
    form = get_team_form(team_id, last_n)
    if not form:
        return f"Tidak ada data form untuk team_id={team_id}"
    summary = []
    for m in form:
        summary.append(
            f"{m['date']} {'H' if m['is_home'] else 'A'} vs {m['opponent']}: "
            f"{m['scored']}-{m['conceded']} ({m['result']})"
        )
    results_str = " ".join(m["result"] for m in form)
    return f"Form terkini (terbaru ke lama): {results_str}\n" + "\n".join(summary)


def tool_get_xg_stats(team_name: str, league: str = "EPL") -> str:
    data = get_understat_team_stats(team_name, league)
    if not data:
        return f"Tidak ada xG data untuk {team_name}"
    return (
        f"xG Stats {data.get('team', team_name)} ({league}):\n"
        f"  xG per match  : {data.get('xg_avg', 'N/A')}\n"
        f"  xGA per match : {data.get('xga_avg', 'N/A')}\n"
        f"  npxG per match: {data.get('npxg_avg', 'N/A')}\n"
        f"  PPDA          : {data.get('ppda_avg', 'N/A')} (lebih rendah = pressing lebih intens)\n"
        f"  Matches       : {data.get('matches', 'N/A')}"
    )


def tool_get_head_to_head(team1_id: int, team2_id: int) -> str:
    h2h = get_head_to_head(team1_id, team2_id)
    if not h2h:
        return "Tidak ada data H2H"
    return (
        f"Head-to-Head ({h2h['total_matches']} pertemuan terakhir):\n"
        f"  Tim 1 menang  : {h2h['team1_wins']}\n"
        f"  Tim 2 menang  : {h2h['team2_wins']}\n"
        f"  Seri          : {h2h['draws']}\n"
        f"  Rata-rata gol : {h2h['avg_goals']} per match"
    )


def tool_get_injuries(team_id: int) -> str:
    injuries = get_injuries(team_id)
    if not injuries:
        return f"Tidak ada injury report untuk team_id={team_id}"
    lines = [f"Injury Report ({len(injuries)} pemain):"]
    for inj in injuries[:5]:
        lines.append(f"  - {inj['player']} ({inj['type']}): {inj['reason']}")
    return "\n".join(lines)


def tool_get_fatigue(team_id: int) -> str:
    score = get_fatigue_score(team_id)
    level = "🔴 Tinggi" if score >= 0.6 else ("🟡 Sedang" if score >= 0.4 else "🟢 Rendah")
    return f"Fatigue Score: {score:.2f} → {level}"


def tool_get_fbref(team_name: str, league: str = "EPL") -> str:
    data = get_fbref_team_stats(team_name, league)
    if not data:
        return f"Tidak ada FBref data untuk {team_name}"
    return (
        f"FBref Stats {data.get('team', team_name)}:\n"
        f"  xG total    : {data.get('xg', 'N/A')}\n"
        f"  xGA total   : {data.get('xga', 'N/A')}\n"
        f"  xG diff     : {data.get('xg_diff', 'N/A')}\n"
        f"  Possession  : {data.get('possession', 'N/A')}%\n"
        f"  Matches     : {data.get('matches', 'N/A')}"
    )


def tool_get_odds(home_team: str, away_team: str, sport: str = "EPL") -> str:
    data = get_best_odds_for_match(home_team, away_team, sport)
    if not data or not data.get("avg_odds"):
        return f"Tidak ada odds data untuk {home_team} vs {away_team}"
    odds = data["avg_odds"]
    probs = odds_to_prob(odds.get("home", 0), odds.get("draw", 0), odds.get("away", 0))
    return (
        f"Odds {home_team} vs {away_team}:\n"
        f"  Home : {odds.get('home', 'N/A')} (implied prob: {round(probs['home']*100,1)}%)\n"
        f"  Draw : {odds.get('draw', 'N/A')} (implied prob: {round(probs['draw']*100,1)}%)\n"
        f"  Away : {odds.get('away', 'N/A')} (implied prob: {round(probs['away']*100,1)}%)\n"
        f"  Overround: {probs.get('overround', 'N/A')}%"
    )


def tool_run_ensemble(home_team: str, away_team: str,
                      home_xg: float = 1.4, away_xg: float = 1.1) -> str:
    result = ensemble_predict(home_team, away_team, home_xg=home_xg, away_xg=away_xg)
    breakdown = result.get("breakdown", {})
    lines = [
        f"🤖 Ensemble Model — {home_team} vs {away_team}:",
        f"  Prediksi    : {result.get('prediction', 'N/A')}",
        f"  Home Win    : {result.get('home_win', 0)}%",
        f"  Draw        : {result.get('draw', 0)}%",
        f"  Away Win    : {result.get('away_win', 0)}%",
        f"  Confidence  : {result.get('confidence', 0)}%",
        f"  Best Score  : {result.get('best_score', 'N/A')}",
        f"  Models Used : {', '.join(result.get('models_used', []))}",
    ]
    if breakdown:
        lines.append("  Breakdown per model:")
        for model, vals in breakdown.items():
            lines.append(f"    {model}: H{vals['home']}% D{vals['draw']}% A{vals['away']}% (w:{vals['weight']}%)")
    return "\n".join(lines)


def tool_analyze_value(prediction: str, model_prob: float,
                       market_odds: dict) -> str:
    result = analyze_value(prediction, model_prob / 100, market_odds)
    status = "✅ VALUE BET" if result["is_value"] else "❌ Tidak ada value"
    return (
        f"Value Bet Analysis:\n"
        f"  Status        : {status}\n"
        f"  Edge vs pasar : {result.get('edge', 0)}%\n"
        f"  Expected Value: {result.get('ev', 0)}%\n"
        f"  Kelly Fraction: {result.get('kelly_pct', 0)}% bankroll\n"
        f"  Model prob    : {result.get('model_prob', 0)}%\n"
        f"  Market prob   : {result.get('market_prob', 0)}%"
    )


def tool_generate_parlay(matches: list) -> str:
    result = generate_parlay(matches)
    return (
        f"🎯 PARLAY RECOMMENDATION:\n"
        f"  Status         : {result.get('status')}\n"
        f"  Total Legs     : {result.get('total_legs', 0)}\n"
        f"  Total Odds     : {result.get('total_odds', 0)}x\n"
        f"  Kumulatif Prob : {result.get('cum_probability', 0)}%\n"
        f"  Expected Value : {result.get('expected_value', 0)}%\n\n"
        + result.get("summary", "")
    )


# ── Tool Dispatcher ───────────────────────────────────────────────────────────

def execute_tool(tool_name: str, tool_input: str) -> str:
    """Eksekusi tool berdasarkan nama dan input string."""
    try:
        inp = json.loads(tool_input) if tool_input.strip().startswith("{") else {"query": tool_input}
    except json.JSONDecodeError:
        inp = {"query": tool_input}

    try:
        if tool_name == "kalkulator":
            return tool_kalkulator(inp.get("expression", tool_input))

        elif tool_name == "tavily_search":
            return tool_tavily_search(inp.get("query", tool_input))

        elif tool_name == "get_team_form":
            return tool_get_team_form(inp.get("team_id", 0), inp.get("last_n", 10))

        elif tool_name == "get_xg_stats":
            return tool_get_xg_stats(inp.get("team_name", ""), inp.get("league", "EPL"))

        elif tool_name == "get_head_to_head":
            return tool_get_head_to_head(inp.get("team1_id", 0), inp.get("team2_id", 0))

        elif tool_name == "get_injuries":
            return tool_get_injuries(inp.get("team_id", 0))

        elif tool_name == "get_fatigue_score":
            return tool_get_fatigue(inp.get("team_id", 0))

        elif tool_name == "get_fbref_stats":
            return tool_get_fbref(inp.get("team_name", ""), inp.get("league", "EPL"))

        elif tool_name == "get_odds":
            return tool_get_odds(
                inp.get("home_team", ""), inp.get("away_team", ""),
                inp.get("sport", "EPL")
            )

        elif tool_name == "run_ensemble_model":
            return tool_run_ensemble(
                inp.get("home_team", ""), inp.get("away_team", ""),
                float(inp.get("home_xg", 1.4)), float(inp.get("away_xg", 1.1))
            )

        elif tool_name == "analyze_value_bet":
            return tool_analyze_value(
                inp.get("prediction", ""),
                float(inp.get("model_prob", 50)),
                inp.get("market_odds", {})
            )

        elif tool_name == "generate_parlay":
            return tool_generate_parlay(inp.get("matches", []))

        else:
            return f"Tool '{tool_name}' tidak dikenal."

    except Exception as e:
        log.error("Tool '%s' error: %s", tool_name, e)
        return f"Error menjalankan {tool_name}: {e}"
