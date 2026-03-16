# RunPod Deployment Guide — Wan2.2-Animate-14B

## Автоматический деплой (CI/CD)

### Как работает

```
git push → GitHub Actions → Docker build → Docker Hub → RunPod Endpoint обновляется
```

Ты просто пушишь код — всё остальное происходит автоматически.

### Настройка (один раз)

#### 1. GitHub Secrets

Зайди в **GitHub → Settings → Secrets and variables → Actions → New repository secret**

| Secret | Значение | Где взять |
|---|---|---|
| `DOCKER_USERNAME` | `snak1o` | Твой Docker Hub username |
| `DOCKER_PASSWORD` | `...` | Docker Hub → Account Settings → Security → Access Tokens |
| `RUNPOD_API_KEY` | `rpa_...` | RunPod → Settings → API Keys |
| `RUNPOD_ENDPOINT_ID` | `...` | RunPod → Serverless → Endpoints → твой endpoint |

#### 2. Network Volume (один раз)

1. **RunPod → Storage → Network Volumes → Create Volume**
   - Name: `bodyswap-models`
   - Size: **100 GB**
   - Data Center: выбери тот, где есть **H100 SXM 80GB**

#### 3. Скачать модели (один раз!)

Запусти Pod с Volume и выполни:

```bash
pip install "huggingface_hub[cli]" insightface gfpgan

# Wan2.2-Animate-14B (~50GB, ~30-60 мин)
huggingface-cli download Wan-AI/Wan2.2-Animate-14B \
    --local-dir /runpod-volume/models/Wan2.2-Animate-14B

# SAM 2.1
wget -q -P /runpod-volume/models \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# GFPGAN
wget -q -P /runpod-volume/models \
    https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth

# InsightFace
python -c "
import os; os.environ['INSIGHTFACE_HOME']='/runpod-volume/models/insightface'
from insightface.app import FaceAnalysis
fa = FaceAnalysis(name='buffalo_l', root='/runpod-volume/models/insightface')
fa.prepare(ctx_id=-1)
print('Done!')
"

# Проверить
du -sh /runpod-volume/models/*
```

**Останови Pod** — модели сохранены на Volume навсегда.

#### 4. Serverless Endpoint (один раз)

1. **RunPod → Serverless → Templates → New Template**
   - Name: `BodySwap-Wan22`
   - Image: `snak1o/bodyswap-gpu:latest`
   - Container Disk: **20 GB**

2. **RunPod → Serverless → Endpoints → New Endpoint**
   - Template: `BodySwap-Wan22`
   - **GPU: H100 SXM 80GB**
   - **Network Volume: `bodyswap-models`**
   - Workers: Min `0`, Max `4`
   - Idle Timeout: `60s`

3. Скопируй **Endpoint ID** → добавь в GitHub Secrets

---

### Использование

```bash
# Внёс изменения в код
git add .
git commit -m "update pipeline"
git push

# GitHub Actions автоматически:
# 1. Соберёт Docker-образ
# 2. Запушит в Docker Hub
# 3. Обновит RunPod Endpoint через API
# Через ~5-10 мин новый образ будет на RunPod
```

### Ручной запуск деплоя

GitHub → Actions → Deploy to RunPod → Run workflow

---

## Стоимость

| Компонент | Цена |
|---|---|
| H100 SXM Serverless | ~$0.00116/сек (~$4.18/час) |
| Network Volume 100GB | ~$0.07/час (~$50/мес) |
| Idle (0 workers) | $0 (только Volume) |
| GitHub Actions | бесплатно (2000 мин/мес) |

## API Тест

```bash
curl -X POST "https://api.runpod.ai/v2/{ENDPOINT_ID}/run" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "mode": "full_video",
      "reference_image": "<base64_photo>",
      "video_base64": "<base64_video>"
    }
  }'
```
