/**
 * ============================================================
 * NEUROSIGN AI - GERMAN TRAFFIC SIGN RECOGNITION FRONTEND LOGIC
 * ============================================================
 */

document.addEventListener('DOMContentLoaded', () => {
    // Application State
    let gtsrbClasses = {};
    let selectedImageFile = null;
    let selectedImageBase64 = null;
    let mediaStream = null;

    // DOM Elements
    const fileInput = document.getElementById('file-input');
    const dropzone = document.getElementById('dropzone');
    const dropzonePrompt = document.getElementById('dropzone-prompt');
    const previewWrapper = document.getElementById('preview-wrapper');
    const imagePreview = document.getElementById('image-preview');
    const btnRemoveImage = document.getElementById('btn-remove-image');
    const btnPredict = document.getElementById('btn-predict');
    
    // Output Elements
    const outputPlaceholder = document.getElementById('output-placeholder');
    const outputLoader = document.getElementById('output-loader');
    const outputContent = document.getElementById('output-content');
    const latencyBadge = document.getElementById('latency-badge');

    const resultIcon = document.getElementById('result-icon');
    const resultCategory = document.getElementById('result-category');
    const resultName = document.getElementById('result-name');
    const resultGerman = document.getElementById('result-german');
    const resultConfidence = document.getElementById('result-confidence');
    const resultDescription = document.getElementById('result-description');
    const topKList = document.getElementById('top-k-list');

    // Camera Elements
    const btnOpenCamera = document.getElementById('btn-open-camera');
    const cameraModal = document.getElementById('camera-modal');
    const btnCloseCamera = document.getElementById('btn-close-camera');
    const btnCancelCamera = document.getElementById('btn-cancel-camera');
    const btnCaptureSnapshot = document.getElementById('btn-capture-snapshot');
    const webcamVideo = document.getElementById('webcam-video');
    const webcamCanvas = document.getElementById('webcam-canvas');

    // Matrix Modal
    const matrixModal = document.getElementById('matrix-modal');
    const openMatrixModal = document.getElementById('open-matrix-modal');
    const btnCloseMatrix = document.getElementById('btn-close-matrix');

    // ============================================================
    // 1. INITIALIZATION & DATA FETCHING
    // ============================================================
    initApp();

    async function initApp() {
        await fetchClasses();
        await fetchAnalytics();
        initPresets();
        setupEventListeners();
        setupNavigation();
    }

    async function fetchClasses() {
        try {
            const res = await fetch('/api/classes');
            const data = await res.json();
            if (data.status === 'success' || data.classes) {
                gtsrbClasses = data.classes || {};
                renderCatalogGrid(gtsrbClasses);
            }
        } catch (err) {
            console.error('Failed to load GTSRB classes:', err);
        }
    }




    async function fetchAnalytics() {
        // Fetch test results
        try {
            const res = await fetch('/api/test-results');
            const metrics = await res.json();
            if (metrics.accuracy) {
                const accPercent = (metrics.accuracy * 100).toFixed(2) + '%';
                document.getElementById('stat-accuracy').innerText = accPercent;
                document.getElementById('metric-accuracy').innerText = accPercent;
                document.getElementById('metric-precision').innerText = (metrics.precision * 100).toFixed(2) + '%';
                document.getElementById('metric-recall').innerText = (metrics.recall * 100).toFixed(2) + '%';
                document.getElementById('metric-f1').innerText = (metrics.f1_score * 100).toFixed(2) + '%';
            }
        } catch (err) {
            console.warn('Test metrics fetch error:', err);
        }

        // Fetch history graphs
        try {
            const res = await fetch('/api/history');
            const history = await res.json();
            if (history.status === 'success') {
                renderCharts(history.data);
            }
        } catch (err) {
            console.warn('History fetch error:', err);
        }
    }

    // ============================================================
    // 2. CHART.JS ANALYTICS
    // ============================================================
    function renderCharts(data) {
        const epochs = Array.from({ length: data.train_loss.length }, (_, i) => i + 1);

        // Common Chart Config
        const chartOptions = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#94a3b8', font: { family: 'Plus Jakarta Sans', size: 12 } }
                }
            },
            scales: {
                x: {
                    ticks: { color: '#64748b' },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' }
                },
                y: {
                    ticks: { color: '#64748b' },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' }
                }
            }
        };

        // Accuracy Chart
        const accCtx = document.getElementById('accuracyChart').getContext('2d');
        new Chart(accCtx, {
            type: 'line',
            data: {
                labels: epochs,
                datasets: [
                    {
                        label: 'Train Accuracy (%)',
                        data: data.train_accuracy,
                        borderColor: '#6366f1',
                        backgroundColor: 'rgba(99, 102, 241, 0.1)',
                        borderWidth: 2,
                        tension: 0.3,
                        pointRadius: 0
                    },
                    {
                        label: 'Valid Accuracy (%)',
                        data: data.valid_accuracy,
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        borderWidth: 2,
                        tension: 0.3,
                        pointRadius: 0
                    }
                ]
            },
            options: chartOptions
        });

        // Loss Chart
        const lossCtx = document.getElementById('lossChart').getContext('2d');
        new Chart(lossCtx, {
            type: 'line',
            data: {
                labels: epochs,
                datasets: [
                    {
                        label: 'Train Loss',
                        data: data.train_loss,
                        borderColor: '#8b5cf6',
                        borderWidth: 2,
                        tension: 0.3,
                        pointRadius: 0
                    },
                    {
                        label: 'Valid Loss',
                        data: data.valid_loss,
                        borderColor: '#06b6d4',
                        borderWidth: 2,
                        tension: 0.3,
                        pointRadius: 0
                    }
                ]
            },
            options: chartOptions
        });



    }

    // ============================================================
    // 3. SIGN CATALOG EXPLORER
    // ============================================================
    function renderCatalogGrid(classesDict, filterCategory = 'all', searchQuery = '') {
        const container = document.getElementById('sign-grid-container');
        container.innerHTML = '';

        const searchLower = searchQuery.toLowerCase().trim();

        Object.keys(classesDict).forEach(classId => {
            const item = classesDict[classId];
            const category = item.category;

            // Category Filter Check
            if (filterCategory !== 'all' && category !== filterCategory) {
                return;
            }

            // Search Query Check
            if (searchQuery) {
                const matchName = item.name.toLowerCase().includes(searchLower);
                const matchGerman = item.german_name.toLowerCase().includes(searchLower);
                const matchId = classId.toString() === searchLower;
                if (!matchName && !matchGerman && !matchId) return;
            }

            const catClass = getCategoryCSSClass(category);

            const card = document.createElement('div');
            card.className = 'sign-card';
            card.innerHTML = `
                <div class="sign-card-header">
                    <span class="sign-class-id">Class ${classId}</span>
                    <span class="sign-cat-badge ${catClass}">${category}</span>
                </div>
                <div class="sign-card-body">
                    <div class="sign-emoji-large">${item.icon}</div>
                    <div class="sign-names-group">
                        <span class="sign-name-en">${item.name}</span>
                        <span class="sign-name-de">${item.german_name}</span>
                    </div>
                </div>
                <p class="sign-desc-text">${item.description}</p>
            `;

            container.appendChild(card);
        });

        if (container.children.length === 0) {
            container.innerHTML = `<p class="text-muted text-center" style="grid-column: 1/-1; padding: 2rem;">No traffic signs matched your search criteria.</p>`;
        }
    }

    function getCategoryCSSClass(category) {
        switch (category) {
            case 'Speed Limit': return 'cat-speed-limit';
            case 'Priority': return 'cat-priority';
            case 'Warning': return 'cat-warning';
            case 'Prohibitory': return 'cat-prohibitory';
            case 'Mandatory': return 'cat-mandatory';
            case 'De-restriction': return 'cat-de-restriction';
            default: return 'cat-warning';
        }
    }

    // Filter Buttons logic
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const category = btn.dataset.category;
            const searchVal = document.getElementById('catalog-search-input').value;
            renderCatalogGrid(gtsrbClasses, category, searchVal);
        });
    });

    document.getElementById('catalog-search-input').addEventListener('input', (e) => {
        const activeCategory = document.querySelector('.filter-btn.active').dataset.category;
        renderCatalogGrid(gtsrbClasses, activeCategory, e.target.value);
    });

    // ============================================================
    // 4. IMAGE UPLOAD & PRESET SELECTION
    // ============================================================
    function setupEventListeners() {
        // Dropzone click trigger
        dropzone.addEventListener('click', (e) => {
            if (e.target !== btnRemoveImage && !btnRemoveImage.contains(e.target)) {
                fileInput.click();
            }
        });

        fileInput.addEventListener('change', handleFileSelect);

        // Drag & Drop
        dropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropzone.classList.add('drag-over');
        });

        dropzone.addEventListener('dragleave', () => {
            dropzone.classList.remove('drag-over');
        });

        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropzone.classList.remove('drag-over');
            if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                processImageFile(e.dataTransfer.files[0]);
            }
        });

        // Remove Preview
        btnRemoveImage.addEventListener('click', (e) => {
            e.stopPropagation();
            resetImageSelection();
        });

        // Predict Button Click
        btnPredict.addEventListener('click', runModelPrediction);

        // Webcam Modal
        btnOpenCamera.addEventListener('click', startWebcam);
        btnCloseCamera.addEventListener('click', stopWebcam);
        btnCancelCamera.addEventListener('click', stopWebcam);
        btnCaptureSnapshot.addEventListener('click', captureWebcamSnapshot);

        // Matrix Lightbox Modal
        openMatrixModal.addEventListener('click', () => matrixModal.classList.remove('hidden'));
        btnCloseMatrix.addEventListener('click', () => matrixModal.classList.add('hidden'));
    }

    function handleFileSelect(e) {
        if (e.target.files && e.target.files[0]) {
            processImageFile(e.target.files[0]);
        }
    }

    function processImageFile(file) {
        selectedImageFile = file;
        selectedImageBase64 = null;

        const reader = new FileReader();
        reader.onload = (e) => {
            displayPreview(e.target.result);
        };
        reader.readAsDataURL(file);
    }

    // Cropper Elements
    const btnOpenCrop = document.getElementById('btn-open-crop');
    const cropModal = document.getElementById('crop-modal');
    const btnCloseCrop = document.getElementById('btn-close-crop');
    const btnCancelCrop = document.getElementById('btn-cancel-crop');
    const btnApplyCrop = document.getElementById('btn-apply-crop');
    const cropperImg = document.getElementById('cropper-img');
    let cropperInstance = null;

    function displayPreview(imageSrc) {
        imagePreview.src = imageSrc;
        dropzonePrompt.classList.add('hidden');
        previewWrapper.classList.remove('hidden');
        btnPredict.disabled = false;
        if (btnOpenCrop) btnOpenCrop.classList.remove('hidden');
    }

    function resetImageSelection() {
        selectedImageFile = null;
        selectedImageBase64 = null;
        fileInput.value = '';
        imagePreview.src = '';
        previewWrapper.classList.add('hidden');
        dropzonePrompt.classList.remove('hidden');
        btnPredict.disabled = true;
        if (btnOpenCrop) btnOpenCrop.classList.add('hidden');

        // Reset Output Workspace
        outputContent.classList.add('hidden');
        outputLoader.classList.add('hidden');
        outputPlaceholder.classList.remove('hidden');
        latencyBadge.innerText = '-- ms';
    }

    // Cropper Event Handlers
    btnOpenCrop.addEventListener('click', openCropper);
    btnCloseCrop.addEventListener('click', closeCropper);
    btnCancelCrop.addEventListener('click', closeCropper);
    btnApplyCrop.addEventListener('click', applyCropper);

    function openCropper() {
        if (!imagePreview.src) return;
        cropperImg.src = imagePreview.src;
        cropModal.classList.remove('hidden');

        if (cropperInstance) {
            cropperInstance.destroy();
        }

        setTimeout(() => {
            cropperInstance = new Cropper(cropperImg, {
                aspectRatio: 1,
                viewMode: 1,
                autoCropArea: 0.8,
                responsive: true
            });
        }, 150);
    }

    function closeCropper() {
        if (cropperInstance) {
            cropperInstance.destroy();
            cropperInstance = null;
        }
        cropModal.classList.add('hidden');
    }

    function applyCropper() {
        if (!cropperInstance) return;
        const canvas = cropperInstance.getCroppedCanvas({
            width: 128,
            height: 128
        });

        if (canvas) {
            const dataUrl = canvas.toDataURL('image/png');
            selectedImageFile = null;
            selectedImageBase64 = dataUrl;
            displayPreview(dataUrl);
            closeCropper();
            runModelPrediction();
        }
    }


    // ============================================================
    // 5. REAL GTSRB SAMPLE PRESETS
    // ============================================================
    async function initPresets() {
        const container = document.getElementById('sample-presets');
        container.innerHTML = '';

        try {
            const res = await fetch('/api/samples');
            const data = await res.json();
            
            if (data.status === 'success' && data.samples && data.samples.length > 0) {
                data.samples.forEach(sample => {
                    const chip = document.createElement('div');
                    chip.className = 'preset-chip';
                    chip.innerHTML = `
                        <span class="preset-icon">${sample.icon}</span>
                        <span class="preset-title">${sample.name}</span>
                    `;

                    chip.addEventListener('click', () => {
                        selectedImageFile = null;
                        selectedImageBase64 = sample.data_url;
                        displayPreview(sample.data_url);
                        runModelPrediction();
                    });
                    container.appendChild(chip);
                });
                return;
            }
        } catch (err) {
            console.warn('Failed to fetch real sample images:', err);
        }

        // Fallback default presets
        const defaultList = [
            { classId: 14, name: 'Stop', icon: '🛑' },
            { classId: 2, name: 'Speed 50', icon: '🚗' },
            { classId: 13, name: 'Yield', icon: '🔻' }
        ];
        defaultList.forEach(preset => {
            const chip = document.createElement('div');
            chip.className = 'preset-chip';
            chip.innerHTML = `<span class="preset-icon">${preset.icon}</span><span class="preset-title">${preset.name}</span>`;
            container.appendChild(chip);
        });
    }


    // ============================================================
    // 6. LIVE WEBCAM SNAPSHOT MODAL
    // ============================================================
    async function startWebcam() {
        try {
            mediaStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
            webcamVideo.srcObject = mediaStream;
            cameraModal.classList.remove('hidden');
        } catch (err) {
            alert('Unable to access webcam: ' + err.message);
        }
    }

    function stopWebcam() {
        if (mediaStream) {
            mediaStream.getTracks().forEach(track => track.stop());
            mediaStream = null;
        }
        cameraModal.classList.add('hidden');
    }

    function captureWebcamSnapshot() {
        webcamCanvas.width = webcamVideo.videoWidth || 640;
        webcamCanvas.height = webcamVideo.videoHeight || 480;
        const ctx = webcamCanvas.getContext('2d');
        ctx.drawImage(webcamVideo, 0, 0, webcamCanvas.width, webcamCanvas.height);
        
        const dataUrl = webcamCanvas.toDataURL('image/png');
        selectedImageFile = null;
        selectedImageBase64 = dataUrl;
        displayPreview(dataUrl);
        stopWebcam();
    }

    // ============================================================
    // 7. REST API MODEL PREDICTION CALL
    // ============================================================
    async function runModelPrediction() {
        if (!selectedImageFile && !selectedImageBase64) return;

        // Show Loader UI
        outputPlaceholder.classList.add('hidden');
        outputContent.classList.add('hidden');
        outputLoader.classList.remove('hidden');
        btnPredict.disabled = true;

        try {
            let res;

            if (selectedImageFile) {
                const formData = new FormData();
                formData.append('file', selectedImageFile);
                res = await fetch('/api/predict', {
                    method: 'POST',
                    body: formData
                });
            } else if (selectedImageBase64) {
                res = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: selectedImageBase64 })
                });
            }

            const data = await res.json();

            if (data.status === 'success') {
                renderPredictionResults(data);
            } else {
                alert('Prediction Error: ' + data.message);
                outputLoader.classList.add('hidden');
                outputPlaceholder.classList.remove('hidden');
            }

        } catch (err) {
            console.error('Prediction API Error:', err);
            alert('Failed to connect to model server.');
            outputLoader.classList.add('hidden');
            outputPlaceholder.classList.remove('hidden');
        } finally {
            btnPredict.disabled = false;
        }
    }

    function renderPredictionResults(data) {
        const top = data.top_prediction;
        const info = top.info;
        const isValid = data.is_valid_sign;

        latencyBadge.innerText = `${data.inference_time_ms} ms`;

        if (!isValid) {
            // Non-traffic sign / Low Confidence State (< 50%)
            resultIcon.innerText = '❓';
            resultCategory.innerText = 'UNRECOGNIZED IMAGE';
            resultCategory.style.background = 'rgba(239, 68, 68, 0.2)';
            resultCategory.style.color = '#fca5a5';
            
            resultName.innerText = 'No Traffic Sign Detected';
            resultGerman.innerText = 'Kein Verkehrszeichen erkannt';
            
            resultConfidence.innerText = `${top.confidence}%`;
            resultConfidence.style.color = '#f87171';
            
            const confCircle = document.getElementById('confidence-circle');
            confCircle.style.background = 'rgba(239, 68, 68, 0.15)';
            confCircle.style.borderColor = 'rgba(239, 68, 68, 0.4)';

            const energyVal = data.top_3_energy || top.confidence;
            resultDescription.innerHTML = `<strong style="color: #f87171;">Image is not a recognized traffic sign.</strong> Combined Top-3 model energy was only <strong>${energyVal}%</strong> (Required: 35.0%).`;

        } else {
            // Valid Traffic Sign State
            resultIcon.innerText = info.icon || '🛑';
            resultCategory.innerText = info.category || 'General';
            resultCategory.style.background = 'rgba(99, 102, 241, 0.15)';
            resultCategory.style.color = '#a5b4fc';

            resultName.innerText = info.name;
            resultGerman.innerText = info.german_name;
            
            resultConfidence.innerText = `${top.confidence}%`;
            
            const confCircle = document.getElementById('confidence-circle');
            if (top.confidence >= 75.0) {
                resultConfidence.style.color = '#10b981';
                confCircle.style.background = 'rgba(16, 185, 129, 0.12)';
                confCircle.style.borderColor = 'rgba(16, 185, 129, 0.3)';
            } else {
                resultConfidence.style.color = '#f59e0b';
                confCircle.style.background = 'rgba(245, 158, 11, 0.12)';
                confCircle.style.borderColor = 'rgba(245, 158, 11, 0.3)';
            }

            resultDescription.innerText = info.description;
        }

        // Top-5 Candidate Bars
        topKList.innerHTML = '';
        data.top_k.forEach((item, index) => {
            const isTop = index === 0;
            const barRow = document.createElement('div');
            barRow.className = 'k-item';
            barRow.innerHTML = `
                <div class="k-label-row">
                    <span class="k-name">${item.info.icon} ${item.info.name}</span>
                    <span class="k-val">${item.confidence}%</span>
                </div>
                <div class="k-track">
                    <div class="k-fill ${isTop && isValid ? 'top-match' : ''}" style="width: 0%; ${!isValid ? 'background: #64748b;' : ''}"></div>
                </div>
            `;
            topKList.appendChild(barRow);

            // Animate bar fill
            setTimeout(() => {
                barRow.querySelector('.k-fill').style.width = `${item.confidence}%`;
            }, 100 + index * 50);
        });

        // Switch Workspace State
        outputLoader.classList.add('hidden');
        outputContent.classList.remove('hidden');
    }


    // ============================================================
    // 8. SCROLL SPY NAVIGATION
    // ============================================================
    function setupNavigation() {
        const sections = document.querySelectorAll('section');
        const navLinks = document.querySelectorAll('.nav-link');

        window.addEventListener('scroll', () => {
            let current = '';
            sections.forEach(section => {
                const sectionTop = section.offsetTop - 120;
                if (window.scrollY >= sectionTop) {
                    current = section.getAttribute('id');
                }
            });

            navLinks.forEach(link => {
                link.classList.remove('active');
                if (link.getAttribute('href') === `#${current}`) {
                    link.classList.add('active');
                }
            });
        });
    }
});
