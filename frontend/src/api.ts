export interface JobStatus {
  id: string
  status: 'IN_QUEUE' | 'IN_PROGRESS' | 'COMPLETED' | 'FAILED' | 'CANCELLED' | 'TIMED_OUT'
  output?: {
    video?: string   // "data:video/mp4;base64,..."
    error?: string
  }
  error?: string
}

// RunPod Serverless config
const RUNPOD_ENDPOINT = import.meta.env.VITE_RUNPOD_ENDPOINT || ''
const RUNPOD_API_KEY = import.meta.env.VITE_RUNPOD_API_KEY || ''

function runpodHeaders() {
  return {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${RUNPOD_API_KEY}`,
  }
}

/**
 * Convert a File to a base64 data URL string.
 */
function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result as string)
    reader.onerror = () => reject(new Error('Failed to read file'))
    reader.readAsDataURL(file)
  })
}

/**
 * Get video dimensions from a File.
 */
function getVideoDimensions(file: File): Promise<{ width: number; height: number }> {
  return new Promise((resolve, reject) => {
    const video = document.createElement('video')
    video.preload = 'metadata'
    video.onloadedmetadata = () => {
      URL.revokeObjectURL(video.src)
      // Round to nearest multiple of 8 (model requirement)
      const w = Math.round(video.videoWidth / 8) * 8
      const h = Math.round(video.videoHeight / 8) * 8
      resolve({ width: w, height: h })
    }
    video.onerror = () => reject(new Error('Cannot read video metadata'))
    video.src = URL.createObjectURL(file)
  })
}

/**
 * Start a swap job on RunPod Serverless (Wan Animate template).
 */
async function startSwapJob(imageB64: string, videoB64: string, width: number, height: number): Promise<string> {
  // Strip data URL prefix (e.g. "data:image/jpeg;base64,") and fix padding
  const cleanBase64 = (dataUrl: string) => {
    const raw = dataUrl.includes(',') ? dataUrl.split(',')[1] : dataUrl
    // Fix padding
    const pad = raw.length % 4
    return pad ? raw + '='.repeat(4 - pad) : raw
  }

  const res = await fetch(`${RUNPOD_ENDPOINT}/run`, {
    method: 'POST',
    headers: runpodHeaders(),
    body: JSON.stringify({
      input: {
        image_base64: cleanBase64(imageB64),
        video_base64: cleanBase64(videoB64),
        prompt: '视频中的人在做动作',
        seed: Math.floor(Math.random() * 999999),
        width,
        height,
        fps: 16,
        cfg: 1.0,
        steps: 6,
      },
    }),
  })

  if (!res.ok) {
    const data = await res.json().catch(() => null)
    throw new Error(data?.error || `RunPod error: ${res.status}`)
  }

  const data = await res.json()
  return data.id
}

/**
 * Check job status.
 */
export async function getJobStatus(jobId: string): Promise<JobStatus> {
  const res = await fetch(`${RUNPOD_ENDPOINT}/status/${jobId}`, {
    headers: runpodHeaders(),
  })

  if (!res.ok) {
    throw new Error(`Status check failed: ${res.status}`)
  }

  return res.json()
}

/**
 * Cancel a running job.
 */
export async function cancelJob(jobId: string): Promise<void> {
  await fetch(`${RUNPOD_ENDPOINT}/cancel/${jobId}`, {
    method: 'POST',
    headers: runpodHeaders(),
  })
}

/**
 * Full swap flow:
 * 1. Encode files as base64
 * 2. Send to RunPod
 * 3. Poll for status
 * 4. Return result blob
 */
export async function swapVideo(
  photo: File,
  video: File,
  onProgress?: (message: string, percent: number) => void,
  onJobId?: (jobId: string) => void,
): Promise<Blob> {
  // Step 1: Detect resolution + encode files
  onProgress?.('Подготовка файлов...', 5)
  const [imageB64, videoB64, dims] = await Promise.all([
    fileToBase64(photo),
    fileToBase64(video),
    getVideoDimensions(video),
  ])

  // Step 2: Start job
  onProgress?.(`Отправка в RunPod (${dims.width}×${dims.height})...`, 15)
  const jobId = await startSwapJob(imageB64, videoB64, dims.width, dims.height)
  onJobId?.(jobId)

  // Step 3: Poll for completion
  onProgress?.('Обработка запущена...', 20)
  const result = await pollJobUntilDone(jobId, onProgress)

  if (result.output?.error) {
    throw new Error(result.output.error)
  }

  if (!result.output?.video) {
    throw new Error('No video in result')
  }

  // Step 4: Decode base64 data URL to blob
  onProgress?.('Декодирование результата...', 98)
  const videoData = result.output.video
  const base64 = videoData.includes(',') ? videoData.split(',')[1] : videoData
  const binaryStr = atob(base64)
  const bytes = new Uint8Array(binaryStr.length)
  for (let i = 0; i < binaryStr.length; i++) {
    bytes[i] = binaryStr.charCodeAt(i)
  }

  return new Blob([bytes], { type: 'video/mp4' })
}

/**
 * Poll RunPod job status until done.
 */
async function pollJobUntilDone(
  jobId: string,
  onProgress?: (message: string, percent: number) => void,
): Promise<JobStatus> {
  const startTime = Date.now()
  const timeout = 2 * 60 * 60 * 1000 // 2 hours

  while (Date.now() - startTime < timeout) {
    await sleep(3000)

    const status = await getJobStatus(jobId)

    if (status.status === 'COMPLETED') {
      onProgress?.('Готово!', 100)
      return status
    }

    if (status.status === 'FAILED' || status.status === 'TIMED_OUT') {
      throw new Error(status.error || status.output?.error || 'Job failed')
    }

    if (status.status === 'CANCELLED') {
      throw new Error('Job cancelled')
    }

    if (status.status === 'IN_PROGRESS') {
      const elapsed = Math.round((Date.now() - startTime) / 1000)
      const mins = Math.floor(elapsed / 60)
      const secs = elapsed % 60
      onProgress?.(`Обработка... (${mins}:${secs.toString().padStart(2, '0')})`,
        Math.min(90, 25 + elapsed * 0.3))
    }

    if (status.status === 'IN_QUEUE') {
      onProgress?.('В очереди, ожидаем GPU...', 22)
    }
  }

  throw new Error('Job timed out after 2 hours')
}

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}
