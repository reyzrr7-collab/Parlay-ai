import os
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from agent.tools import ALL_TOOLS
from agent.prompts import NEMOTRON_SYSTEM_PROMPT, PARLAY_BUILDER_PROMPT
from dotenv import load_dotenv

load_dotenv()

NGC_API_KEY = os.getenv("NGC_API_KEY")

llm = ChatOpenAI(
    model="nvidia/llama-3.1-nemotron-70b-instruct",
    openai_api_base="https://integrate.api.nvidia.com/v1",
    openai_api_key=NGC_API_KEY,
    temperature=0.1,
    max_tokens=2000
).bind_tools(ALL_TOOLS)


class AgentState(TypedDict):
    messages: List
    fixtures: List
    analyses: List
    parlay: dict


def scanner_node(state: AgentState) -> AgentState:
    """Node 1: Scan pertandingan hari ini."""
    messages = [
        SystemMessage(content=NEMOTRON_SYSTEM_PROMPT),
        HumanMessage(content="Gunakan tool get_fixtures_today untuk ambil semua pertandingan hari ini di EPL (league_id=39). Laporkan hasilnya.")
    ]
    response = llm.invoke(messages)
    state["messages"] = messages + [response]
    return state


def analyzer_node(state: AgentState) -> AgentState:
    """Node 2: Analisis setiap pertandingan."""
    messages = state["messages"] + [
        HumanMessage(content=(
            "Untuk setiap pertandingan yang ditemukan, gunakan tool analyze_match dan check_value_bet. "
            "Identifikasi pertandingan mana yang memiliki value bet dan confidence tinggi."
        ))
    ]
    response = llm.invoke(messages)
    state["messages"] = messages + [response]
    return state


def reasoning_node(state: AgentState) -> AgentState:
    """Node 3: Nemotron deep reasoning."""
    messages = state["messages"] + [
        HumanMessage(content=(
            "Berdasarkan data yang sudah dikumpulkan, lakukan analisis kualitatif mendalam:\n"
            "1. Cek faktor non-statistik (motivasi, absensi, cuaca, berita)\n"
            "2. Identifikasi red flag yang bisa membatalkan prediksi\n"
            "3. Beri penilaian final: MASUK / SKIP untuk setiap pertandingan\n"
            "Gunakan tool get_weather dan get_latest_news jika diperlukan."
        ))
    ]
    response = llm.invoke(messages)
    state["messages"] = messages + [response]
    return state


def parlay_builder_node(state: AgentState) -> AgentState:
    """Node 4: Build parlay optimal."""
    messages = state["messages"] + [
        HumanMessage(content=PARLAY_BUILDER_PROMPT)
    ]
    response = llm.invoke(messages)
    state["messages"] = messages + [response]
    return state


def should_use_tools(state: AgentState) -> str:
    """Router: cek apakah perlu pakai tools."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "continue"


def build_graph():
    """Build LangGraph workflow."""
    workflow = StateGraph(AgentState)

    tool_node = ToolNode(ALL_TOOLS)

    workflow.add_node("scanner", scanner_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("analyzer", analyzer_node)
    workflow.add_node("reasoning", reasoning_node)
    workflow.add_node("parlay_builder", parlay_builder_node)

    workflow.set_entry_point("scanner")
    workflow.add_conditional_edges("scanner", should_use_tools, {"tools": "tools", "continue": "analyzer"})
    workflow.add_edge("tools", "analyzer")
    workflow.add_conditional_edges("analyzer", should_use_tools, {"tools": "tools", "continue": "reasoning"})
    workflow.add_edge("reasoning", "parlay_builder")
    workflow.add_edge("parlay_builder", END)

    return workflow.compile()


def run_agent():
    """Jalankan agen lengkap."""
    graph = build_graph()
    initial_state = AgentState(messages=[], fixtures=[], analyses=[], parlay={})
    result = graph.invoke(initial_state)
    final_message = result["messages"][-1]
    return final_message.content


if __name__ == "__main__":
    print(run_agent())
