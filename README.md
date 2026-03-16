# BodySwapAI — Full-Body Swap Service

> Полная замена персонажа в видео с помощью **Wan2.2-Animate-14B** (Replace mode).
> Кинематографическое качество: адаптация к освещению, сохранение движений и выражений лица.

## 🚀 Быстрый старт

### 1. Клонировать и настроить

```bash
cd swap-service
cp .env.example .env   # или отредактировать существующий .env
```

### 2. Запуск через Docker Compose (API + Redis + MinIO)

```bash
docker compose up -d
```

Сервисы:
- **API**: http://localhost:8000
- **MinIO Console**: http://localhost:9001 (minioadmin / minioadmin)

### 3. Запуск для разработки (без Docker)

```bash
# Установить зависимости
pip install -r requirements.txt

# Запустить Redis (нужен для очереди задач)
redis-server &

# Запустить API
uvicorn app.main:app --reload --port 8000

# Запустить Celery worker (в отдельном терминале)
celery -A app.workers.celery_worker worker --loglevel=info --concurrency=1
```

## 📁 Структура проекта

```
swap-service/
├── app/
│   ├── main.py              # FastAPI entry point
│   ├── config.py             # Settings (.env)
│   ├── api/
│   │   ├── routes.py         # REST API endpoints
│   │   └── schemas.py        # Pydantic models
│   ├── pipeline/
│   │   ├── orchestrator.py   # Pipeline coordinator
│   │   ├── generate.py       # Wan2.2-Animate-14B Replace
│   │   ├── segment.py        # Background segmentation (SAM 2)
│   │   ├── face.py           # Face swap + enhancement
│   │   └── composite.py      # Video compositing + assembly
│   ├── workers/
│   │   ├── celery_worker.py  # Async task processing
│   │   └── runpod_handler.py # RunPod GPU serverless handler
│   └── storage/
│       └── s3.py             # MinIO / S3 client
├── web/                      # Web UI
├── docker-compose.yml        # Local stack
├── Dockerfile                # API server image
├── Dockerfile.gpu            # RunPod GPU worker image (Wan2.2)
├── start.sh                  # GPU worker entrypoint
├── download_models.py        # One-time model download to Network Volume
└── .env                      # Environment config
```

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/swap` | Upload photo + video, create job |
| `GET` | `/api/status/{job_id}` | Get job status & progress |
| `GET` | `/api/result/{job_id}` | Get result download URL |
| `GET` | `/api/jobs` | List recent jobs |
| `DELETE` | `/api/jobs/{job_id}` | Delete job |

## ☁️ RunPod Deployment

Подробная инструкция: [README_runpod_setup.md](README_runpod_setup.md)

```bash
# Собрать GPU-образ (лёгкий, без моделей)
docker build -f Dockerfile.gpu -t bodyswap-gpu:wan22 .

# Запушить в Docker Hub
docker tag bodyswap-gpu:wan22 snak1o/bodyswap-gpu:wan22
docker push snak1o/bodyswap-gpu:wan22
```

## ⚙️ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RUNPOD_API_KEY` | — | RunPod API key |
| `RUNPOD_ENDPOINT_ID` | — | RunPod serverless endpoint ID |
| `MODELS_DIR` | `/runpod-volume/models` | Path to models on Network Volume |
| `WAN_MODEL_NAME` | `Wan-AI/Wan2.2-Animate-14B` | HuggingFace model name |
| `WAN_REPLACE_FLAG` | `true` | Use Replace mode |
| `WAN_USE_RELIGHTING_LORA` | `true` | Adapt lighting to scene |
| `WAN_OFFLOAD_MODEL` | `false` | Enable model offloading (for smaller GPUs) |
| `WAN_RESOLUTION_W` | `1280` | Output width |
| `WAN_RESOLUTION_H` | `720` | Output height |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection |
| `MAX_VIDEO_DURATION_SEC` | `120` | Max video length |
