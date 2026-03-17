import { io, Socket } from 'socket.io-client'

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

export interface JobUpdate {
  job_id: string
  message: string
  percent: number
  done: boolean
  error?: string
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
  onUploadProgress?: (percent: number) => void,
  onJobUpdate?: (update: JobUpdate) => void
): Promise<Blob> {
  const formData = new FormData()
  formData.append('photo', photo)
  formData.append('video', video)

  // Step 1: Upload files via XHR to track upload progress
  const jobId = await new Promise<string>((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', `${API_BASE}/swap`)
    xhr.responseType = 'json'

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onUploadProgress) {
        onUploadProgress(Math.round((e.loaded / e.total) * 100))
      }
    }

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        const data = xhr.response
        if (data?.job_id) {
          resolve(data.job_id)
        } else {
          reject(new Error(data?.error || 'No job_id in response'))
        }
      } else {
        reject(new Error(`Upload failed: ${xhr.status} ${xhr.statusText}`))
      }
    }

    xhr.onerror = () => reject(new Error('Network error'))
    xhr.ontimeout = () => reject(new Error('Upload timeout'))
    xhr.timeout = 0

    xhr.send(formData)
  })

  // Step 2: Connect to Socket.IO and listen for job updates
  return new Promise<Blob>((resolve, reject) => {
    const socket: Socket = io(API_BASE, {
      transports: ['websocket', 'polling'],
      reconnection: true,
      reconnectionAttempts: 10,
      reconnectionDelay: 2000,
    })

    const cleanup = () => {
      socket.off('job_update')
      socket.off('connect_error')
      socket.disconnect()
    }

    socket.on('job_update', async (update: JobUpdate) => {
      if (update.job_id !== jobId) return

      if (onJobUpdate) onJobUpdate(update)

      if (update.done) {
        cleanup()

        if (update.error) {
          reject(new Error(update.error))
          return
        }

        // Step 3: Download result
        try {
          const res = await fetch(`${API_BASE}/result/${jobId}`)
          if (!res.ok) {
            const errData = await res.json().catch(() => null)
            throw new Error(errData?.error || `Download failed: ${res.status}`)
          }
          const blob = await res.blob()
          resolve(blob)
        } catch (err) {
          reject(err)
        }
      }
    })

    socket.on('connect_error', (err) => {
      console.error('Socket.IO connection error:', err)
    })

    // Timeout after 2 hours in case the job hangs
    setTimeout(() => {
      cleanup()
      reject(new Error('Job timed out after 2 hours'))
    }, 2 * 60 * 60 * 1000)
  })
}
