/**
 * BodySwapAI — Web UI Application
 */

const API_BASE = '';

// ---- State ----
let state = {
    photoFile: null,
    videoFile: null,
    currentJobId: null,
    pollInterval: null,
};

// ---- DOM Elements ----
const $ = (sel) => document.querySelector(sel);
const photoInput = $('#photo-input');
const videoInput = $('#video-input');
const photoDropzone = $('#photo-dropzone');
const videoDropzone = $('#video-dropzone');
const photoPreview = $('#photo-preview');
const videoPreview = $('#video-preview');
const submitBtn = $('#submit-btn');
const uploadSection = $('#upload-section');
const processingSection = $('#processing-section');
const resultSection = $('#result-section');
const errorSection = $('#error-section');
const progressFill = $('#progress-fill');
const progressStep = $('#progress-step');
const progressPercent = $('#progress-percent');
const progressMessage = $('#progress-message');
const resultVideo = $('#result-video');
const resultDownload = $('#result-download');
const resultMeta = $('#result-meta');
const errorMessage = $('#error-message');

// ---- Step labels ----
const STEP_LABELS = {
    pending: 'Ожидание',
    uploading: 'Загрузка',
    processing: 'Обработка',
    pose_extraction: 'Извлечение позы',
    segmentation: 'Сегментация фона',
    body_generation: 'Генерация тела',
    face_refinement: 'Улучшение лица',
    compositing: 'Композитинг',
    post_processing: 'Пост-обработка',
    completed: 'Готово ✅',
    failed: 'Ошибка ❌',
};

// ---- File Upload Handling ----
function setupDropzone(dropzone, input, type) {
    dropzone.addEventListener('click', () => input.click());

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        const file = e.dataTransfer.files[0];
        if (file) handleFile(file, type);
    });

    input.addEventListener('change', () => {
        if (input.files[0]) handleFile(input.files[0], type);
    });
}

function handleFile(file, type) {
    if (type === 'photo') {
        state.photoFile = file;
        photoPreview.src = URL.createObjectURL(file);
        photoPreview.hidden = false;
        photoDropzone.classList.add('has-file');
    } else {
        state.videoFile = file;
        videoPreview.src = URL.createObjectURL(file);
        videoPreview.hidden = false;
        videoDropzone.classList.add('has-file');
    }
    updateSubmitBtn();
}

function updateSubmitBtn() {
    submitBtn.disabled = !(state.photoFile && state.videoFile);
}

// ---- Submit Job ----
async function submitJob() {
    const formData = new FormData();
    formData.append('photo', state.photoFile);
    formData.append('video', state.videoFile);
    formData.append('scene_detection', $('#opt-scene').checked);
    formData.append('temporal_smoothing', $('#opt-smooth').checked);
    formData.append('upscale', $('#opt-upscale').checked);
    formData.append('skip_frames', $('#opt-skip').value);

    try {
        submitBtn.disabled = true;
        submitBtn.querySelector('.btn__text').textContent = '⏳ Загрузка...';

        const response = await fetch(`${API_BASE}/api/swap`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Upload failed');
        }

        const data = await response.json();
        state.currentJobId = data.job_id;

        showSection('processing');
        startPolling();

    } catch (error) {
        showError(error.message);
    }
}

// ---- Polling ----
function startPolling() {
    if (state.pollInterval) clearInterval(state.pollInterval);

    state.pollInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_BASE}/api/status/${state.currentJobId}`);
            const data = await response.json();

            updateProgress(data);

            if (data.status === 'completed') {
                clearInterval(state.pollInterval);
                showResult(data);
            } else if (data.status === 'failed') {
                clearInterval(state.pollInterval);
                showError(data.error || data.message);
            }
        } catch (error) {
            console.error('Polling error:', error);
        }
    }, 2000);
}

function updateProgress(data) {
    const percent = Math.round(data.progress * 100);
    progressFill.style.width = `${percent}%`;
    progressPercent.textContent = `${percent}%`;
    progressStep.textContent = STEP_LABELS[data.status] || data.current_step || data.status;
    progressMessage.textContent = data.message || '';

    // Update pipeline step dots
    const steps = document.querySelectorAll('.step');
    const stepOrder = [
        'pending', 'pose_extraction', 'segmentation',
        'body_generation', 'face_refinement', 'compositing', 'post_processing',
    ];

    const currentIdx = stepOrder.indexOf(data.status);
    steps.forEach((stepEl, i) => {
        stepEl.classList.remove('active', 'done');
        if (i < currentIdx) {
            stepEl.classList.add('done');
        } else if (i === currentIdx) {
            stepEl.classList.add('active');
        }
    });
}

// ---- Show Result ----
function showResult(data) {
    showSection('result');

    const resultUrl = data.result_url;
    resultVideo.src = resultUrl;
    resultDownload.href = resultUrl;
    resultMeta.textContent = `Job: ${data.job_id}`;
}

// ---- Show Error ----
function showError(message) {
    showSection('error');
    errorMessage.textContent = message;
}

// ---- Section Switching ----
function showSection(section) {
    uploadSection.hidden = section !== 'upload';
    processingSection.hidden = section !== 'processing';
    resultSection.hidden = section !== 'result';
    errorSection.hidden = section !== 'error';
}

// ---- Reset ----
function resetUI() {
    state = { photoFile: null, videoFile: null, currentJobId: null, pollInterval: null };

    photoPreview.hidden = true;
    videoPreview.hidden = true;
    photoDropzone.classList.remove('has-file');
    videoDropzone.classList.remove('has-file');
    photoInput.value = '';
    videoInput.value = '';
    submitBtn.disabled = true;
    submitBtn.querySelector('.btn__text').textContent = '🚀 Начать обработку';
    progressFill.style.width = '0%';

    document.querySelectorAll('.step').forEach(s => s.classList.remove('active', 'done'));

    showSection('upload');
}

// ---- Init ----
document.addEventListener('DOMContentLoaded', () => {
    setupDropzone(photoDropzone, photoInput, 'photo');
    setupDropzone(videoDropzone, videoInput, 'video');

    submitBtn.addEventListener('click', submitJob);
    $('#new-job-btn').addEventListener('click', resetUI);
    $('#retry-btn').addEventListener('click', resetUI);
});
