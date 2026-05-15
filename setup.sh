#!/bin/bash
# Setup script untuk VPS DO 2GB
# Jalankan: bash setup.sh

set -e
echo "=== PARLAY AI SETUP ==="

# Buat direktori project
sudo mkdir -p /opt/parlay-ai/logs
sudo chown -R $USER:$USER /opt/parlay-ai

# Copy semua file
cp -r . /opt/parlay-ai/
cd /opt/parlay-ai

# Install dependencies Python
pip3 install -r requirements.txt

# Setup PostgreSQL
sudo -u postgres psql -c "CREATE DATABASE parlay_ai;" 2>/dev/null || echo "DB already exists"
sudo -u postgres psql -c "CREATE USER parlay_user WITH PASSWORD 'ganti_password_ini';" 2>/dev/null || echo "User already exists"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE parlay_ai TO parlay_user;" 2>/dev/null || true

# Copy .env
if [ ! -f .env ]; then
    cp .env.example .env
    echo "⚠️  Edit .env dan isi semua API keys sebelum lanjut!"
    exit 1
fi

# Init database
python3 main.py init

# Setup systemd service
sudo tee /etc/systemd/system/parlay-ai.service > /dev/null <<EOF
[Unit]
Description=Parlay AI Football Prediction
After=network.target postgresql.service redis.service

[Service]
Type=simple
User=$USER
WorkingDirectory=/opt/parlay-ai
ExecStart=/usr/bin/python3 main.py schedule
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable parlay-ai
sudo systemctl start parlay-ai

echo ""
echo "✅ Setup selesai!"
echo "Cek status: sudo systemctl status parlay-ai"
echo "Lihat log  : tail -f /opt/parlay-ai/logs/scheduler.log"
echo "Test API   : python3 main.py test"
echo "Run manual : python3 main.py run"
