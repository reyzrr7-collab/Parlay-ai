"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         FOOTBALL PARLAY AGENT v3.0                                          ║
║         ReAct + Self-Reflection + Dixon-Coles + XGBoost + Parlay Logic      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import re
import os
import json
import time
import sqlite3
import logging
import asyncio
import requests
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv
from telegram import Bot
from telegram.constants import ParseMode

from database.models import init_db
from database.queries import (
    save_message, load_recent_messages,
    get_user_context, upsert_user_fact, upsert_user_name, show_user_facts
)
from agent.tools import execute_tool, get_tools_description
from agent.prompts import REACT_SYSTEM, EXTRACTION_SYSTEM, REFLECTION_SYSTEM
from evaluation.tracker import print_accuracy_report

load_dotenv()

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
NVIDIA_API_KEY  = os.getenv("NGC_API_KEY", "")
NVIDIA_CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
TELEGRAM_TOKEN  = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID= os.getenv("TELEGRAM_CHAT_ID", "")

MODEL_NAME      = "nvidia/nemotron-3-super-120b-a12b"
MODEL_MINI      = "nvidia/llama-3.1-nemotron-nano-8b-v1"
MAX_ITERATIONS  = 10
LLM_TIMEOUT     = 120
LLM_MINI_TIMEOUT= 60

# Rate limiter
_last_req = 0.0
RATE_LIMIT = 0.5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("main")


# ══════════════════════════════════════════════════════════════════════════════
#  LLM CALLS
# ══════════════════════════════════════════════════════════════════════════════

def _rate_limit():
    global _last_req
    elapsed = time.time() - _last_req
    if elapsed < RATE_LIMIT:
        time.sleep(RATE_LIMIT - elapsed)
    _last_req = time.time()


def panggil_llm(system: str, user: str, num_predict: int = 4096,
                temperature: float = 0.1, streaming: bool = False,
                retries: int = 2) -> Optional[str]:
    _rate_limit()
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "max_tokens": num_predict,
        "temperature": temperature,
        "top_p": 0.7,
        "stream": streaming,
    }
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }
    for attempt in range(1, retries + 2):
        try:
            resp = requests.post(NVIDIA_CHAT_URL, json=payload, headers=headers,
                                 timeout=LLM_TIMEOUT, stream=streaming)
            resp.raise_for_status()
            if streaming:
                chunks = []
                print("\n🤖 Agent:", flush=True)
                for line in resp.iter_lines():
                    if not line:
                        continue
                    decoded = line.decode() if isinstance(line, bytes) else line
                    if decoded.startswith("data:"):
                        ds = decoded[5:].strip()
                        if ds == "[DONE]":
                            break
                        try:
                            delta = json.loads(ds)["choices"][0]["delta"].get("content", "")
                            if delta:
                                print(delta, end="", flush=True)
                                chunks.append(delta)
                        except:
                            pass
                print()
                raw = "".join(chunks)
            else:
                raw = resp.json()["choices"][0]["message"]["content"]

            # Log think block, strip dari output
            think = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
            if think:
                log.debug("[THINK] %s", think.group(1)[:300])
            return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        except requests.Timeout:
            log.warning("Timeout (percobaan %d)", attempt)
            if attempt <= retries:
                time.sleep(3 * attempt)
        except requests.RequestException as e:
            log.error("LLM error: %s", e)
            if attempt <= retries:
                time.sleep(3 * attempt)
    return None


def panggil_llm_mini(system: str, user: str) -> Optional[str]:
    _rate_limit()
    payload = {
        "model": MODEL_MINI,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "max_tokens": 512,
        "temperature": 0.0,
        "top_p": 0.7,
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }
    for attempt in range(1, 3):
        try:
            resp = requests.post(NVIDIA_CHAT_URL, json=payload, headers=headers,
                                 timeout=LLM_MINI_TIMEOUT)
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]
            return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        except requests.Timeout:
            if attempt < 2:
                time.sleep(2)
        except Exception as e:
            log.warning("LLM mini error: %s", e)
            if attempt < 2:
                time.sleep(2)
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  SMART FACT EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def ekstrak_fakta(pesan: str, session_id: str) -> None:
    raw = panggil_llm_mini(EXTRACTION_SYSTEM, f"Pesan: {pesan}")
    if not raw:
        return
    try:
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        fakta = json.loads(clean)
    except:
        return
    if not isinstance(fakta, dict):
        return
    for k, v in fakta.items():
        if v and str(v).strip():
            upsert_user_fact(session_id, k, str(v).strip(), source=pesan[:200])
    if "nama" in fakta and fakta["nama"]:
        upsert_user_name(session_id, fakta["nama"])


# ══════════════════════════════════════════════════════════════════════════════
#  SELF-REFLECTION
# ══════════════════════════════════════════════════════════════════════════════

def self_reflect(jawaban: str) -> dict:
    """Cek kelengkapan jawaban menggunakan LLM mini."""
    raw = panggil_llm_mini(REFLECTION_SYSTEM, f"Jawaban agent:\n{jawaban[:1500]}")
    if not raw:
        return {"complete": True, "needs_revision": False}
    try:
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        return json.loads(clean)
    except:
        return {"complete": True, "needs_revision": False}


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIDENCE INDICATOR
# ══════════════════════════════════════════════════════════════════════════════

def hitung_confidence(jawaban: str) -> str:
    """Ekstrak confidence dari jawaban dan tampilkan indikator."""
    match = re.search(r'"confidence"\s*:\s*(\d+)', jawaban)
    if not match:
        return ""
    conf = int(match.group(1))
    if conf >= 70:
        return f"🟢 Confidence: {conf}%"
    elif conf >= 55:
        return f"🟡 Confidence: {conf}%"
    else:
        return f"🔴 Confidence: {conf}%"


# ══════════════════════════════════════════════════════════════════════════════
#  REACT AGENT LOOP
# ══════════════════════════════════════════════════════════════════════════════

TOOL_PATTERN = re.compile(
    r"Action\s*:\s*(\w+)\s*\nAction Input\s*:\s*(.*?)(?=\nObservation|\nThought|\nFinal Answer|$)",
    re.DOTALL | re.IGNORECASE
)
FINAL_PATTERN = re.compile(r"Final Answer\s*:\s*(.*)", re.DOTALL | re.IGNORECASE)


def jalankan_agent(pertanyaan: str, session_id: str) -> str:
    """
    ReAct agent loop utama:
    Thought → Action → Observation → ... → Final Answer
    """
    memory  = load_recent_messages(session_id)
    tools_desc = get_tools_description()

    system = REACT_SYSTEM + f"\n\n{tools_desc}"
    if memory:
        system += f"\n\n[KONTEKS USER]\n{memory}"

    scratchpad = f"Pertanyaan: {pertanyaan}\n\nThought:"
    tool_retries: dict = {}
    final_answer = None

    for iteration in range(1, MAX_ITERATIONS + 1):
        log.info("━ Iterasi %d/%d", iteration, MAX_ITERATIONS)

        response = panggil_llm(
            system, scratchpad,
            streaming=(iteration == MAX_ITERATIONS)  # streaming di iterasi terakhir
        )

        if not response:
            log.warning("LLM tidak merespon di iterasi %d", iteration)
            break

        scratchpad += " " + response

        # Cek Final Answer
        final_match = FINAL_PATTERN.search(response)
        if final_match:
            final_answer = final_match.group(1).strip()
            log.info("✅ Final Answer ditemukan di iterasi %d", iteration)
            break

        # Cek Action
        tool_match = TOOL_PATTERN.search(response)
        if not tool_match:
            # Tidak ada action & tidak ada final answer → paksa selesai
            if iteration >= 5:
                final_answer = response
                break
            scratchpad += "\nThought:"
            continue

        tool_name  = tool_match.group(1).strip()
        tool_input = tool_match.group(2).strip()

        # Guard retry per tool
        retry_key = f"{tool_name}:{tool_input[:50]}"
        tool_retries[retry_key] = tool_retries.get(retry_key, 0) + 1
        if tool_retries[retry_key] > 2:
            obs = f"Tool '{tool_name}' sudah dicoba {tool_retries[retry_key]}x, skip."
            log.warning(obs)
        else:
            log.info("🔧 Tool: %s | Input: %s", tool_name, tool_input[:80])
            obs = execute_tool(tool_name, tool_input)
            log.info("📊 Observation: %s", str(obs)[:150])

        scratchpad += f"\nObservation: {obs}\nThought:"

    # Fallback jika tidak ada final answer
    if not final_answer:
        final_answer = scratchpad.split("Thought:")[-1].strip() or "Maaf, tidak dapat menyelesaikan analisis."

    # Self-reflection
    reflection = self_reflect(final_answer)
    if reflection.get("needs_revision") and not reflection.get("complete"):
        missing = reflection.get("missing", [])
        if missing:
            log.info("🔄 Revisi diperlukan: %s", missing)
            revision_prompt = (
                f"Jawaban sebelumnya:\n{final_answer[:1000]}\n\n"
                f"Yang masih kurang: {', '.join(missing)}\n"
                f"Lengkapi analisis dengan informasi yang kurang tersebut."
            )
            revised = panggil_llm(system, revision_prompt, streaming=True)
            if revised:
                final_answer = revised

    # Simpan ke memory
    save_message(session_id, "user", pertanyaan)
    save_message(session_id, "assistant", final_answer[:2000])
    ekstrak_fakta(pertanyaan, session_id)

    return final_answer


# ══════════════════════════════════════════════════════════════════════════════
#  TELEGRAM OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

async def _kirim_telegram_async(pesan: str) -> None:
    try:
        bot = Bot(token=TELEGRAM_TOKEN)
        for i in range(0, len(pesan), 4000):
            await bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=pesan[i:i+4000],
                parse_mode=ParseMode.MARKDOWN
            )
            await asyncio.sleep(0.3)
        log.info("✅ Telegram terkirim.")
    except Exception as e:
        log.error("❌ Telegram error: %s", e)


def kirim_telegram(pesan: str) -> None:
    try:
        asyncio.run(_kirim_telegram_async(pesan))
    except RuntimeError:
        asyncio.get_event_loop().run_until_complete(_kirim_telegram_async(pesan))


def format_telegram(pertanyaan: str, jawaban: str, session_id: str) -> str:
    now = datetime.now().strftime("%d %b %Y %H:%M")
    conf_str = hitung_confidence(jawaban)
    return f"""⚽ *FOOTBALL PARLAY AGENT v3.0*
📅 {now} | 🔑 `{session_id[:8]}`
{conf_str}
━━━━━━━━━━━━━━━━━━━━

❓ *Pertanyaan:*
_{pertanyaan}_

━━━━━━━━━━━━━━━━━━━━
🤖 *Analisis AI:*

{jawaban[:3000]}

━━━━━━━━━━━━━━━━━━━━
_Powered by NVIDIA Nemotron 120B_
_⚠️ Untuk referensi saja — bukan keputusan mutlak._"""


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    init_db()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║      ⚽  FOOTBALL PARLAY AGENT v3.0  ⚽                  ║")
    print("║  ReAct + Dixon-Coles + XGBoost + Parlay Logic            ║")
    print("╚══════════════════════════════════════════════════════════╝")

    session_input = input(
        "\nMasukkan Session ID untuk lanjut sesi lama,\n"
        "atau Enter untuk sesi baru: "
    ).strip()
    session_id = session_input or datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n📌 Session ID: {session_id}")

    ctx = get_user_context(session_id)
    if ctx["user_name"]:
        print(f"👋 Selamat datang kembali, {ctx['user_name']}!")
    else:
        print("💡 Tips: Sebutkan nama dan tim favorit Anda.")

    print("\nKetik 'exit' untuk keluar.")
    print("Ketik 'fakta' untuk melihat profil Anda.")
    print("Ketik 'akurasi' untuk laporan akurasi prediksi.")
    print("─" * 57 + "\n")

    while True:
        try:
            user_input = input("Anda: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nSampai jumpa! Data tersimpan. ⚽")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "keluar"):
            print("Sampai jumpa! ⚽")
            break
        if user_input.lower() == "fakta":
            show_user_facts(session_id)
            continue
        if user_input.lower() == "akurasi":
            print_accuracy_report()
            continue

        jawaban = jalankan_agent(user_input, session_id)

        # Kirim ke Telegram jika jawaban berkualitas
        layak_tg = (
            jawaban
            and len(jawaban) > 100
            and TELEGRAM_TOKEN
            and not jawaban.startswith("Error")
            and not jawaban.startswith("⚠️")
        )
        if layak_tg:
            kirim_telegram(format_telegram(user_input, jawaban, session_id))

        print("\n" + "─" * 57)
