/**
 * VOXGUARD AI — Frontend Interactive Engine
 * Handles Drag & Drop, Live Mic Recording, Canvas Audio Visualization, 
 * Async Backend REST API communication, and Forensic Dashboard rendering.
 */

document.addEventListener('DOMContentLoaded', () => {
    // Global State
    let selectedFile = null;
    let currentAnalysisData = null;
    let mediaRecorder = null;
    let audioChunks = [];
    let recordTimerInterval = null;
    let recordStartTime = 0;
    let audioContext = null;
    let analyserNode = null;
    let micCanvasAnimFrame = null;

    // DOM Elements
    const backendStatusText = document.getElementById('backend-status-text');
    const modelStatusBadge = document.getElementById('model-status-badge');
    const modelStatusText = document.getElementById('model-status-text');
    
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const filePreviewCard = document.getElementById('file-preview-card');
    const previewFilename = document.getElementById('preview-filename');
    const previewFilesize = document.getElementById('preview-filesize');
    const audioPlayer = document.getElementById('audio-player');
    
    const analyzeBtn = document.getElementById('analyze-btn');
    const removeFileBtn = document.getElementById('remove-file-btn');
    
    const recordBtn = document.getElementById('record-btn');
    const recordBtnText = document.getElementById('record-btn-text');
    const stopRecordBtn = document.getElementById('stop-record-btn');
    const recordTimer = document.getElementById('record-timer');
    const recorderStatus = document.getElementById('recorder-status');
    const micCanvas = document.getElementById('mic-canvas');
    
    const loadingPanel = document.getElementById('loading-panel');
    const loadingStatusTitle = document.getElementById('loading-status-title');
    const resultsPanel = document.getElementById('results-panel');
    
    const confidenceScore = document.getElementById('confidence-score');
    const gaugeCircle = document.getElementById('gauge-circle');
    const verdictBadge = document.getElementById('verdict-badge');
    const verdictDesc = document.getElementById('verdict-desc');
    const waveformImg = document.getElementById('waveform-img');
    const spectrogramFilename = document.getElementById('spectrogram-filename');
    const resultTimestamp = document.getElementById('result-timestamp');
    
    const metricPitch = document.getElementById('metric-pitch');
    const metricSpectral = document.getElementById('metric-spectral');
    const metricNoise = document.getElementById('metric-noise');
    const metricPhase = document.getElementById('metric-phase');
    const downloadReportBtn = document.getElementById('download-report-btn');

    // 1. Initial Health Check
    checkBackendHealth();
    loadHistoryFromStorage();

    function checkBackendHealth() {
        fetch('/api/health')
            .then(res => res.json())
            .then(data => {
                if (data.status === 'ok') {
                    backendStatusText.textContent = 'Backend Online';
                    if (data.model_loaded) {
                        modelStatusBadge.style.borderColor = 'rgba(16, 185, 129, 0.4)';
                        modelStatusText.textContent = 'TensorFlow Model Ready';
                    } else {
                        modelStatusBadge.style.borderColor = 'rgba(245, 158, 11, 0.4)';
                        modelStatusText.textContent = 'Heuristic Fallback Mode';
                    }
                }
            })
            .catch(() => {
                backendStatusText.textContent = 'Backend Offline';
                backendStatusText.style.color = 'var(--accent-fake)';
            });
    }

    // 2. Tab Navigation
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            
            btn.classList.add('active');
            const tabId = btn.getAttribute('data-tab');
            document.getElementById(tabId).classList.add('active');
        });
    });

    // 3. Drag & Drop File Handling
    if (dropzone) {
        ['dragenter', 'dragover'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                dropzone.classList.add('dragover');
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                dropzone.classList.remove('dragover');
            });
        });

        dropzone.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleSelectedFile(files[0]);
            }
        });
    }

    if (fileInput) {
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleSelectedFile(e.target.files[0]);
            }
        });
    }

    function handleSelectedFile(file) {
        selectedFile = file;
        previewFilename.textContent = file.name;
        previewFilesize.textContent = formatBytes(file.size);
        
        filePreviewCard.style.display = 'flex';

        // Audio Preview setup
        if (file.type.startsWith('audio/') || file.name.match(/\.(wav|mp3|m4a|ogg|webm)$/i)) {
            const objectUrl = URL.createObjectURL(file);
            audioPlayer.src = objectUrl;
            audioPlayer.style.display = 'block';
        } else {
            audioPlayer.style.display = 'none';
        }
    }

    if (removeFileBtn) {
        removeFileBtn.addEventListener('click', () => {
            selectedFile = null;
            filePreviewCard.style.display = 'none';
            if (fileInput) fileInput.value = '';
        });
    }

    // 4. Live Microphone Recording Logic
    if (recordBtn) {
        recordBtn.addEventListener('click', toggleRecording);
    }

    if (stopRecordBtn) {
        stopRecordBtn.addEventListener('click', stopRecording);
    }

    async function toggleRecording() {
        if (!mediaRecorder || mediaRecorder.state === 'inactive') {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];

                mediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0) {
                        audioChunks.push(event.data);
                    }
                };

                mediaRecorder.onstop = () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    const file = new File([audioBlob], `mic_recording_${Date.now()}.wav`, { type: 'audio/wav' });
                    handleSelectedFile(file);
                    recorderStatus.textContent = 'Recording completed! Ready for voice analysis.';
                    stopMicCanvasVisualizer();
                };

                mediaRecorder.start(100);
                recordBtn.classList.add('recording');
                recordBtnText.textContent = 'Recording...';
                stopRecordBtn.disabled = false;
                recorderStatus.textContent = 'Speak into your microphone...';

                // Timer
                recordStartTime = Date.now();
                recordTimerInterval = setInterval(updateRecordTimer, 1000);

                // Audio Canvas Visualizer
                startMicCanvasVisualizer(stream);

            } catch (err) {
                recorderStatus.textContent = 'Microphone access denied or unsupported.';
                console.error(err);
            }
        }
    }

    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
            mediaRecorder.stream.getTracks().forEach(track => track.stop());
            clearInterval(recordTimerInterval);
            recordTimer.textContent = '00:00';
            recordBtn.classList.remove('recording');
            recordBtnText.textContent = 'Start Recording';
            stopRecordBtn.disabled = true;
        }
    }

    function updateRecordTimer() {
        const elapsedSec = Math.floor((Date.now() - recordStartTime) / 1000);
        const mins = String(Math.floor(elapsedSec / 60)).padStart(2, '0');
        const secs = String(elapsedSec % 60).padStart(2, '0');
        recordTimer.textContent = `${mins}:${secs}`;
    }

    function startMicCanvasVisualizer(stream) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const source = audioContext.createMediaStreamSource(stream);
        analyserNode = audioContext.createAnalyser();
        analyserNode.fftSize = 64;
        source.connect(analyserNode);

        const canvasCtx = micCanvas.getContext('2d');
        const bufferLength = analyserNode.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);

        function draw() {
            micCanvasAnimFrame = requestAnimationFrame(draw);
            analyserNode.getByteFrequencyData(dataArray);

            canvasCtx.fillStyle = '#0a0f1d';
            canvasCtx.fillRect(0, 0, micCanvas.width, micCanvas.height);

            const barWidth = (micCanvas.width / bufferLength) * 2.5;
            let x = 0;

            for (let i = 0; i < bufferLength; i++) {
                const barHeight = (dataArray[i] / 255) * micCanvas.height;
                const gradient = canvasCtx.createLinearGradient(0, micCanvas.height, 0, 0);
                gradient.addColorStop(0, '#00f2fe');
                gradient.addColorStop(1, '#3a7bd5');

                canvasCtx.fillStyle = gradient;
                canvasCtx.fillRect(x, micCanvas.height - barHeight, barWidth, barHeight);

                x += barWidth + 2;
            }
        }
        draw();
    }

    function stopMicCanvasVisualizer() {
        if (micCanvasAnimFrame) cancelAnimationFrame(micCanvasAnimFrame);
        if (audioContext) audioContext.close();
    }

    // 5. Sample Picker
    window.loadSample = function(sampleFileName, label) {
        // Fetch sample image from static/samples and convert to file
        fetch(`/static/samples/${sampleFileName}`)
            .then(res => res.blob())
            .then(blob => {
                const file = new File([blob], sampleFileName, { type: 'image/png' });
                handleSelectedFile(file);
                alert(`Loaded sample: ${label}`);
            })
            .catch(err => alert('Error loading sample file: ' + err));
    };

    // 6. Trigger Analysis
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', runAnalysis);
    }

    function runAnalysis() {
        if (!selectedFile) return;

        // Show Loading UI
        loadingPanel.style.display = 'block';
        resultsPanel.style.display = 'none';
        filePreviewCard.style.display = 'none';

        // Animated Step Progress
        setStep(1);
        setTimeout(() => setStep(2), 600);
        setTimeout(() => setStep(3), 1200);

        const formData = new FormData();
        formData.append('file', selectedFile);

        fetch('/api/predict', {
            method: 'POST',
            body: formData
        })
        .then(res => {
            if (!res.ok) throw new Error('Analysis failed');
            return res.json();
        })
        .then(data => {
            if (data.status === 'success') {
                setTimeout(() => {
                    loadingPanel.style.display = 'none';
                    renderResults(data);
                }, 1600);
            } else {
                alert('Analysis Error: ' + (data.message || 'Unknown error'));
                loadingPanel.style.display = 'none';
            }
        })
        .catch(err => {
            alert('Failed to connect to backend server.');
            loadingPanel.style.display = 'none';
            console.error(err);
        });
    }

    function setStep(stepNum) {
        for (let i = 1; i <= 3; i++) {
            const stepEl = document.getElementById(`step-${i}`);
            if (i <= stepNum) {
                stepEl.classList.add('active');
            } else {
                stepEl.classList.remove('active');
            }
        }
    }

    // 7. Render Forensic Dashboard
    function renderResults(data) {
        currentAnalysisData = data;
        resultsPanel.style.display = 'block';

        const isReal = data.prediction === 'real';
        const confidence = data.confidence;

        // Gauge Animation
        confidenceScore.textContent = `${confidence}%`;
        const circumference = 2 * Math.PI * 70; // 440
        const offset = circumference - (confidence / 100) * circumference;
        gaugeCircle.style.strokeDashoffset = offset;
        gaugeCircle.style.stroke = isReal ? 'var(--accent-real)' : 'var(--accent-fake)';

        // Verdict Badge
        if (isReal) {
            verdictBadge.textContent = 'AUTHENTIC HUMAN VOICE';
            verdictBadge.className = 'verdict-badge real';
            verdictDesc.textContent = 'Spectral features match natural human vocal tract acoustics.';
        } else {
            verdictBadge.textContent = 'SYNTHETIC DEEPFAKE VOICE';
            verdictBadge.className = 'verdict-badge fake';
            verdictDesc.textContent = 'Neural network detected pitch discontinuities & synthetic phase artifacts.';
        }

        // Waveform Image
        waveformImg.src = `${data.waveform_url}?t=${Date.now()}`;
        spectrogramFilename.textContent = data.filename;
        resultTimestamp.textContent = `Scanned at ${new Date().toLocaleTimeString()}`;

        // Metrics
        metricPitch.textContent = data.metrics.pitch_stability;
        metricSpectral.textContent = data.metrics.spectral_flatness;
        metricNoise.textContent = data.metrics.noise_floor;
        metricPhase.textContent = data.metrics.phase_coherence;

        // Save to History
        saveHistoryItem({
            timestamp: new Date().toLocaleTimeString(),
            filename: data.filename,
            prediction: data.prediction,
            confidence: data.confidence,
            size: data.file_size
        });
    }

    window.resetAnalysis = function() {
        resultsPanel.style.display = 'none';
        selectedFile = null;
        if (fileInput) fileInput.value = '';
    };

    // 8. Download Forensic Report (JSON)
    if (downloadReportBtn) {
        downloadReportBtn.addEventListener('click', () => {
            if (!currentAnalysisData) return;
            const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(currentAnalysisData, null, 2));
            const dlAnchor = document.createElement('a');
            dlAnchor.setAttribute("href", dataStr);
            dlAnchor.setAttribute("download", `VoxGuard_Report_${currentAnalysisData.filename}.json`);
            document.body.appendChild(dlAnchor);
            dlAnchor.click();
            dlAnchor.remove();
        });
    }

    // 9. History Table LocalStorage Management
    function saveHistoryItem(item) {
        let history = JSON.parse(localStorage.getItem('voxguard_history') || '[]');
        history.unshift(item);
        if (history.length > 10) history = history.slice(0, 10);
        localStorage.setItem('voxguard_history', JSON.stringify(history));
        renderHistoryTable(history);
    }

    function loadHistoryFromStorage() {
        const history = JSON.parse(localStorage.getItem('voxguard_history') || '[]');
        renderHistoryTable(history);
    }

    function renderHistoryTable(history) {
        const tbody = document.getElementById('history-table-body');
        if (!tbody) return;

        if (history.length === 0) {
            tbody.innerHTML = `<tr class="empty-row"><td colspan="5" style="text-align: center; color: var(--text-muted);">No recent voice analyses found</td></tr>`;
            return;
        }

        tbody.innerHTML = history.map(item => `
            <tr>
                <td>${item.timestamp}</td>
                <td><strong>${item.filename}</strong></td>
                <td><span class="status-tag ${item.prediction}">${item.prediction.toUpperCase()}</span></td>
                <td><strong>${item.confidence}%</strong></td>
                <td>${item.size || 'N/A'}</td>
            </tr>
        `).join('');
    }

    window.clearHistory = function() {
        localStorage.removeItem('voxguard_history');
        renderHistoryTable([]);
    };

    // Helper: Bytes formatter
    function formatBytes(bytes, decimals = 1) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
    }
});
