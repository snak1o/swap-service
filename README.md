# Wan2.2 Replace — Body Swap Service

Замена персонажа в видео через Wan2.2-Animate-14B.

## Быстрый старт (RunPod)

### 1. Создай Pod на RunPod
- **Template:** RunPod PyTorch 2.4
- **GPU:** H100 SXM 80GB
- **Network Volume:** прикрепи (100+ GB)

### 2. Установка (один раз)
```bash
cd /workspace
git clone https://github.com/ТВОЙ_ЮЗЕР/swap-service.git
cd swap-service
bash setup.sh
```

### 3. Запуск
```bash
python server.py
```

Сервер стартует на порте `8000`.
URL: `https://POD_ID-8000.proxy.runpod.net`

## API

### POST /swap
Заменить персонажа в видео.

```bash
curl -X POST https://POD_ID-8000.proxy.runpod.net/swap \
  -F "photo=@face.jpg" \
  -F "video=@source.mp4" \
  -o result.mp4
```

### Python
```python
import requests

resp = requests.post("https://POD_ID-8000.proxy.runpod.net/swap",
    files={
        "photo": open("face.jpg", "rb"),
        "video": open("source.mp4", "rb"),
    })

with open("result.mp4", "wb") as f:
    f.write(resp.content)
```

### GET /health
```json
{"status": "ok", "model_loaded": true, "gpu": {"name": "H100", "vram_total_mb": "81559"}}
```

## Обновление
```bash
cd /workspace/swap-service
git pull
# Перезапусти server.py
```
