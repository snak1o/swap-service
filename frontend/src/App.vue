<template>
  <div class="app-container">
    <!-- Header -->
    <header class="header">
      <h1>🎭 Wan2.2 Body Swap</h1>
      <p>Загрузи фото и видео — получи видео с заменённым персонажем</p>
      <div class="status-badge">
        <span class="status-dot" :class="serverStatus" />
        <span>{{ statusText }}</span>
      </div>
    </header>

    <!-- Upload Grid -->
    <div class="upload-grid">
      <!-- Photo Upload -->
      <div
        class="upload-card"
        :class="{ 'has-file': photoPreview, dragover: photoDragover }"
        @click="triggerUpload('photo')"
        @dragover.prevent="photoDragover = true"
        @dragleave.prevent="photoDragover = false"
        @drop.prevent="handleDrop($event, 'photo')"
      >
        <template v-if="!photoPreview">
          <div class="upload-icon">📸</div>
          <div class="upload-label">Референсное фото</div>
          <div class="upload-hint">JPG, PNG • Лицо персонажа</div>
        </template>
        <div v-else class="preview-container">
          <img :src="photoPreview" alt="Photo preview" />
          <div class="preview-name">
            <span>{{ photoFile?.name }}</span>
            <button class="remove-btn" @click.stop="removeFile('photo')">✕</button>
          </div>
        </div>
        <input
          ref="photoInput"
          type="file"
          accept="image/*"
          @change="handleFileSelect($event, 'photo')"
        />
      </div>

      <!-- Video Upload -->
      <div
        class="upload-card"
        :class="{ 'has-file': videoPreview, dragover: videoDragover }"
        @click="triggerUpload('video')"
        @dragover.prevent="videoDragover = true"
        @dragleave.prevent="videoDragover = false"
        @drop.prevent="handleDrop($event, 'video')"
      >
        <template v-if="!videoPreview">
          <div class="upload-icon">🎬</div>
          <div class="upload-label">Исходное видео</div>
          <div class="upload-hint">MP4 • Видео с персонажем</div>
        </template>
        <div v-else class="preview-container">
          <video :src="videoPreview" muted loop autoplay playsinline />
          <div class="preview-name">
            <span>{{ videoFile?.name }}</span>
            <button class="remove-btn" @click.stop="removeFile('video')">✕</button>
          </div>
        </div>
        <input
          ref="videoInput"
          type="file"
          accept="video/*"
          @change="handleFileSelect($event, 'video')"
        />
      </div>
    </div>

    <!-- Swap Button -->
    <button
      class="swap-btn"
      :class="{ processing: isProcessing }"
      :disabled="!canSwap"
      @click="startSwap"
    >
      <template v-if="!isProcessing">
        🔄 Заменить персонажа
      </template>
      <template v-else>
        ⏳ {{ progressMessage }}
      </template>
    </button>

    <!-- Progress -->
    <div v-if="isProcessing" class="progress-container">
      <div class="progress-bar">
        <div class="progress-fill" :style="{ width: progressPercent + '%' }" />
      </div>
      <div class="progress-text">{{ progressMessage }}</div>
    </div>

    <!-- Error -->
    <div v-if="error" class="error-toast">
      ⚠️ {{ error }}
    </div>

    <!-- Result -->
    <div v-if="resultUrl" class="result-section">
      <div class="result-card">
        <h3>✅ Результат</h3>
        <video class="result-video" :src="resultUrl" controls autoplay loop />
        <div class="result-actions">
          <button class="download-btn" @click="downloadResult">
            ⬇️ Скачать видео
          </button>
          <button class="reset-btn" @click="resetAll">
            🔄 Новый запрос
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { swapVideo, checkHealth } from './api'

// --- State ---
const photoFile = ref<File | null>(null)
const videoFile = ref<File | null>(null)
const photoPreview = ref<string>('')
const videoPreview = ref<string>('')
const photoDragover = ref(false)
const videoDragover = ref(false)

const isProcessing = ref(false)
const uploadPercent = ref(0)
const progressMessage = ref('')
const progressPercent = ref(0)
const error = ref('')
const resultUrl = ref('')
const resultBlob = ref<Blob | null>(null)

const serverStatus = ref<'checking' | 'online' | 'offline'>('checking')
const statusText = ref('Проверяю сервер...')

const photoInput = ref<HTMLInputElement | null>(null)
const videoInput = ref<HTMLInputElement | null>(null)

// --- Computed ---
const canSwap = computed(() => {
  return photoFile.value && videoFile.value && !isProcessing.value
})

// --- Methods ---
function triggerUpload(type: 'photo' | 'video') {
  if (type === 'photo' && !photoPreview.value) {
    photoInput.value?.click()
  } else if (type === 'video' && !videoPreview.value) {
    videoInput.value?.click()
  }
}

function handleFileSelect(event: Event, type: 'photo' | 'video') {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  if (file) setFile(file, type)
}

function handleDrop(event: DragEvent, type: 'photo' | 'video') {
  photoDragover.value = false
  videoDragover.value = false
  const file = event.dataTransfer?.files?.[0]
  if (file) setFile(file, type)
}

function setFile(file: File, type: 'photo' | 'video') {
  error.value = ''
  const url = URL.createObjectURL(file)

  if (type === 'photo') {
    if (photoPreview.value) URL.revokeObjectURL(photoPreview.value)
    photoFile.value = file
    photoPreview.value = url
  } else {
    if (videoPreview.value) URL.revokeObjectURL(videoPreview.value)
    videoFile.value = file
    videoPreview.value = url
  }
}

function removeFile(type: 'photo' | 'video') {
  if (type === 'photo') {
    if (photoPreview.value) URL.revokeObjectURL(photoPreview.value)
    photoFile.value = null
    photoPreview.value = ''
    if (photoInput.value) photoInput.value.value = ''
  } else {
    if (videoPreview.value) URL.revokeObjectURL(videoPreview.value)
    videoFile.value = null
    videoPreview.value = ''
    if (videoInput.value) videoInput.value.value = ''
  }
}

async function startSwap() {
  if (!photoFile.value || !videoFile.value) return

  isProcessing.value = true
  error.value = ''
  resultUrl.value = ''
  resultBlob.value = null
  uploadPercent.value = 0
  progressPercent.value = 0
  progressMessage.value = 'Загрузка файлов...'

  try {
    const blob = await swapVideo(
      photoFile.value,
      videoFile.value,
      (percent) => {
        uploadPercent.value = percent
        if (percent < 100) {
          progressPercent.value = percent * 0.3 // 0-30% = upload
          progressMessage.value = `Загрузка файлов: ${percent}%`
        } else {
          progressPercent.value = 35
          progressMessage.value = 'Wan2.2 обрабатывает видео... это займёт несколько минут'
          startProgressSimulation()
        }
      }
    )

    resultBlob.value = blob
    resultUrl.value = URL.createObjectURL(blob)
    progressPercent.value = 100
    progressMessage.value = 'Готово!'
  } catch (err: any) {
    error.value = err.message || 'Произошла ошибка'
  } finally {
    isProcessing.value = false
  }
}

let progressInterval: ReturnType<typeof setInterval> | null = null

function startProgressSimulation() {
  if (progressInterval) clearInterval(progressInterval)
  progressInterval = setInterval(() => {
    if (progressPercent.value < 90) {
      progressPercent.value += 0.5
      const minutes = Math.round((90 - progressPercent.value) / 3)
      progressMessage.value = `Wan2.2 обрабатывает видео... ~${minutes} мин.`
    }
  }, 5000)
}

function downloadResult() {
  if (!resultBlob.value) return
  const a = document.createElement('a')
  a.href = URL.createObjectURL(resultBlob.value)
  a.download = `swap_result_${Date.now()}.mp4`
  a.click()
  URL.revokeObjectURL(a.href)
}

function resetAll() {
  removeFile('photo')
  removeFile('video')
  if (resultUrl.value) URL.revokeObjectURL(resultUrl.value)
  resultUrl.value = ''
  resultBlob.value = null
  error.value = ''
  progressPercent.value = 0
  progressMessage.value = ''
  if (progressInterval) clearInterval(progressInterval)
}

async function checkServerStatus() {
  try {
    const health = await checkHealth()
    serverStatus.value = 'online'
    statusText.value = health.gpu
      ? `${health.gpu.name} • ${health.model_loaded ? 'Модель загружена' : 'Модель не найдена'}`
      : 'Сервер онлайн'
  } catch {
    serverStatus.value = 'offline'
    statusText.value = 'Сервер недоступен'
  }
}

onMounted(() => {
  checkServerStatus()
  setInterval(checkServerStatus, 30000)
})
</script>
