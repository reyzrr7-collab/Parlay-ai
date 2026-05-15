# ⚽ Football Parlay Agent v3.0

ReAct + Dixon-Coles + XGBoost + Bayesian + Multi-Agent

---

## 📁 Struktur Project

```
parlay-ai/
├── .env                    ← API keys (isi dulu ini)
├── requirements.txt
├── main.py                 ← Jalankan ini untuk chat
├── scheduler.py            ← Jalankan ini untuk auto-prediksi
│
├── data/
│   ├── collector.py        ← API-Football (form, H2H, injury)
│   ├── scraper.py          ← Understat (xG) + FBref (advanced)
│   ├── odds.py             ← OddsAPI + Pinnacle
│   └── preprocessor.py     ← Feature engineering
│
├── models/
│   ├── dixon_coles.py      ← Model statistik utama
│   ├── bayesian.py         ← Hierarchical Bayesian
│   ├── xgboost_model.py    ← Machine learning
│   └── ensemble.py         ← Gabungkan semua model
│
├── agent/
│   ├── tools.py            ← Semua tools ReAct agent
│   ├── prompts.py          ← System prompts
│   └── graph.py            ← LangGraph multi-agent
│
├── parlay/
│   ├── filter.py           ← Filter kandidat parlay
│   ├── value_bet.py        ← Kelly Criterion + edge
│   └── generator.py        ← Build kombinasi optimal
│
├── database/
│   ├── models.py           ← Skema tabel SQLite
│   └── queries.py          ← CRUD operations
│
└── evaluation/
    ├── tracker.py          ← Log prediksi
    ├── metrics.py          ← Brier Score, RPS, ROI
    └── retrainer.py        ← Auto retrain XGBoost
```

---

## 🚀 Setup di VPS DO 2GB

### 1. Install system dependencies
```bash
apt update && apt upgrade -y
apt install python3 python3-pip python3-venv git -y
```

### 2. Clone & setup
```bash
git clone <repo-url> /opt/parlay-ai
cd /opt/parlay-ai
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Isi .env
```bash
nano .env
# Isi semua API keys
```

### 4. Jalankan
```bash
# Mode chat interaktif
python main.py

# Mode scheduler otomatis
python scheduler.py
```

### 5. Jalankan sebagai service (24/7)
```bash
nano /etc/systemd/system/parlay-ai.service
```
```ini
[Unit]
Description=Parlay AI
After=network.target

[Service]
User=root
WorkingDirectory=/opt/parlay-ai
ExecStart=/opt/parlay-ai/venv/bin/python scheduler.py
Restart=always

[Install]
WantedBy=multi-user.target
```
```bash
systemctl enable parlay-ai
systemctl start parlay-ai
```

---

## 🔑 API Keys yang Diperlukan

| Key | Daftar di | Gratis |
|-----|-----------|--------|
| NGC_API_KEY | https://build.nvidia.com | ✅ |
| FOOTBALL_API_KEY | https://rapidapi.com/api-sports | ✅ (100 req/hari) |
| TAVILY_API_KEY | https://tavily.com | ✅ |
| ODDS_API_KEY | https://the-odds-api.com | ✅ (500 req/bulan) |
| TELEGRAM_TOKEN | @BotFather di Telegram | ✅ |

---

## 💬 Contoh Penggunaan

```
Anda: Prediksi Liverpool vs Arsenal besok

Anda: Analisis 3 pertandingan EPL hari ini untuk parlay

Anda: akurasi   (→ tampilkan laporan akurasi)

Anda: fakta     (→ tampilkan profil tersimpan)
```

---

## ⚠️ Disclaimer

Untuk referensi dan edukasi saja.
Tidak ada sistem prediksi yang akurat 100%.
Sepak bola selalu punya faktor tak terduga.
