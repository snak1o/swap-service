# BodySwapAI — Full-Body Swap Service

> Перенос персонажа с фото на видео с повторением всех движений 1 в 1.

## 🚀 Быстрый старт

### 1. Клонировать и настроить

```bash
cd swap-service
cp .env.example .env   # или отредактировать существующий .env
```

### 2. Запуск через Docker Compose

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
│   │   ├── pose.py           # Pose extraction (DWPose)
│   │   ├── segment.py        # Background segmentation (SAM 2)
│   │   ├── generate.py       # Body generation (Animate Anyone 2)
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
├── Dockerfile.gpu            # RunPod GPU worker image
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

```bash
# Собрать GPU-образ
docker build -f Dockerfile.gpu -t bodyswap-gpu:latest .

# Запушить в Docker Hub / GHCR
docker tag bodyswap-gpu:latest your-registry/bodyswap-gpu:latest
docker push your-registry/bodyswap-gpu:latest

# Создать Serverless Endpoint на RunPod Dashboard
# → Image: your-registry/bodyswap-gpu:latest
# → GPU: A40 или A100
```

## ⚙️ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RUNPOD_API_KEY` | — | RunPod API key |
| `RUNPOD_ENDPOINT_ID` | — | RunPod serverless endpoint ID |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection |
| `MINIO_ENDPOINT` | `localhost:9000` | MinIO endpoint |
| `MAX_VIDEO_DURATION_SEC` | `120` | Max video length |
| `SKIP_FRAMES` | `0` | Process every Nth frame |
