#!/bin/bash
# start.sh: IT Support System'i Docker Compose ile başlatır.

# --- Yapılandırma ---
# Docker Compose dosyasının adı
DOCKER_COMPOSE_FILE="docker-compose.yml"

# Başlangıç mesajı
echo "🚀 IT Destek Sistemi (RAG Chatbot) başlatılıyor..."

# 1. Ön Koşul Kontrolü (Gerekli API Anahtarlarının Kontrolü)
# .env dosyasının varlığını kontrol et
if [ ! -f .env ]; then
    echo "🚨 Hata: .env dosyası bulunamadı."
    echo "Lütfen 'cp env.example .env' komutunu çalıştırın ve API anahtarlarınızı yapılandırın."
    exit 1
fi

# .env dosyasında anahtarların gerçekten ayarlanıp ayarlanmadığını kontrol et
if grep -q "your_actual_openai_api_key_here" .env || grep -q "your_actual_gemini_api_key_here" .env; then
    echo "🚨 Hata: .env dosyasında API anahtarları yapılandırılmamış."
    echo "Lütfen OPENAI_API_KEY ve GEMINI_API_KEY değerlerini güncelleyin."
    exit 1
fi

# 2. Servisleri Başlatma
# Docker Compose ile servisleri (backend, frontend, vectordatabase) oluştur ve başlat
echo "🐳 Docker Compose servisleri oluşturuluyor ve arka planda başlatılıyor..."
# -d: Arka planda çalıştır (detached mode)
# --build: Görüntüler eski ise yeniden oluştur
# --force-recreate: Kapsayıcıları yeniden oluştur (gerekirse temiz bir başlangıç için)
docker-compose -f "${DOCKER_COMPOSE_FILE}" up -d --build --force-recreate

# Docker Compose'un başarılı bir şekilde başlatılıp başlatılmadığını kontrol et
if [ $? -ne 0 ]; then
    echo "❌ Hata: Docker Compose başlatılamadı. Hata mesajları için yukarıyı kontrol edin."
    exit 1
fi

# 3. Servis Durumu Kontrolü (Opsiyonel ama iyi bir pratik)
echo "🔍 Servis durumu kontrol ediliyor (birkaç saniye sürebilir)..."
sleep 5 # Servislerin başlaması için kısa bir bekleme süresi

# Tüm servislerin çalıştığını teyit etmek için logs komutunu kullanabiliriz
# docker-compose ps -a | grep -q "Up"

echo "✅ Başlatma başarılı!"

# 4. Erişim Bilgileri
echo "---"
echo "🌐 Servis Adresleri:"
echo "   - Frontend (Chatbot): http://localhost:8501"
echo "   - Backend API (FastAPI): http://localhost:8000"
echo "   - Qdrant Vector DB (Web UI): http://localhost:6333"
echo "---"
echo "Servis loglarını izlemek için: 'docker-compose logs -f'"
echo "Tüm servisleri durdurmak için: 'docker-compose down'"
