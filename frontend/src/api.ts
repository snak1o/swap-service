export interface SwapResponse {
  success: boolean
  error?: string
}

export interface HealthResponse {
  status: string
  model_loaded: boolean
  gpu: {
    name: string
    vram_total_mb: string
    vram_used_mb: string
  } | null
}

const API_BASE = import.meta.env.VITE_API_URL || '/api'

export async function checkHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`)
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`)
  return res.json()
}

export async function swapVideo(
  photo: File,
  video: File,
  onProgress?: (percent: number) => void
): Promise<Blob> {
  const formData = new FormData()
  formData.append('photo', photo)
  formData.append('video', video)

  const xhr = new XMLHttpRequest()

  return new Promise((resolve, reject) => {
    xhr.open('POST', `${API_BASE}/swap`)

    xhr.responseType = 'blob'

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) {
        onProgress(Math.round((e.loaded / e.total) * 100))
      }
    }

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(xhr.response as Blob)
      } else {
        reject(new Error(`Swap failed: ${xhr.status} ${xhr.statusText}`))
      }
    }

    xhr.onerror = () => reject(new Error('Network error'))
    xhr.ontimeout = () => reject(new Error('Request timeout'))

    xhr.timeout = 0 // no timeout — inference takes long

    xhr.send(formData)
  })
}
