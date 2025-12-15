// API 基础配置
const API_BASE_URL = 'http://localhost:8000';
let currentTaskId = null;
let currentFile = null;
let currentVideoUrl = null;
let currentSubtitleData = null;

// 多语言支持
function t(key) {
    return window.t ? window.t(key) : key;
}

// DOM 元素 - 一步生成Tab
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const startProcessBtn = document.getElementById('os-startProcessBtn');

// 初始化 - 一步生成Tab
if (uploadArea && fileInput) {
    uploadArea.addEventListener('click', () => fileInput.click());
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    fileInput.addEventListener('change', handleFileSelect);
}

if (startProcessBtn) {
    startProcessBtn.addEventListener('click', startProcessing);
}

// 下载按钮事件
const osDownloadSrtBtn = document.getElementById('os-downloadSrtBtn');
const osDownloadVttBtn = document.getElementById('os-downloadVttBtn');
const osDownloadJsonBtn = document.getElementById('os-downloadJsonBtn');

if (osDownloadSrtBtn) osDownloadSrtBtn.addEventListener('click', () => downloadSubtitle('srt', 'os'));
if (osDownloadVttBtn) osDownloadVttBtn.addEventListener('click', () => downloadSubtitle('vtt', 'os'));
if (osDownloadJsonBtn) osDownloadJsonBtn.addEventListener('click', () => downloadJson('os'));

// 语言切换
const languageSelector = document.getElementById('languageSelector');
if (languageSelector) {
    languageSelector.addEventListener('change', (e) => {
        if (window.switchLanguage) {
            window.switchLanguage(e.target.value);
        }
    });
}

// 绑定事件（在语言切换后重新绑定）
function bindEvents() {
    const diarizationCheckbox = document.getElementById('enableDiarization');
    if (diarizationCheckbox && !diarizationCheckbox.onchange) {
        // 事件已在HTML中绑定
    }
}

// 拖拽处理
function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFileSelect(e) {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
}

function handleFile(file) {
    currentFile = file;
    uploadArea.innerHTML = `
        <div class="upload-content">
            <p>${t('fileSelected')}: ${file.name}</p>
            <p class="file-hint">${t('fileSize')}: ${(file.size / 1024 / 1024).toFixed(2)} MB</p>
        </div>
    `;
    configSection.style.display = 'block';
}
    
    // 如果是视频文件，创建预览
    if (file.type.startsWith('video/')) {
        const url = URL.createObjectURL(file);
        currentVideoUrl = url;
        videoSource.src = url;
        videoPlayer.load();
    }
}

// 开始处理
async function startProcessing() {
    // 检查上传类型
    const uploadType = document.querySelector('input[name="uploadType"]:checked')?.value || 'file';
    const urlInput = document.getElementById('urlInput');
    const youtubeInput = document.getElementById('youtubeInput');
    
    // 验证输入
    if (uploadType === 'file' && !currentFile) {
        alert(t('selectFile'));
        return;
    } else if (uploadType === 'url' && (!urlInput || !urlInput.value.trim())) {
        alert(t('enterUrl'));
        return;
    } else if (uploadType === 'youtube' && (!youtubeInput || !youtubeInput.value.trim())) {
        alert(t('enterYouTubeUrl'));
        return;
    }

    const startProcessBtn = document.getElementById('os-startProcessBtn');
    const configSection = document.querySelector('#one-step-tab .section:nth-of-type(2)');
    const progressSection = document.getElementById('os-progressSection');
    const resultSection = document.getElementById('os-resultSection');
    
    if (startProcessBtn) startProcessBtn.disabled = true;
    if (configSection) configSection.style.display = 'none';
    if (progressSection) progressSection.style.display = 'block';
    if (resultSection) resultSection.style.display = 'none';

    // 收集参数
    const params = collectAllParams('os');
    
    try {
        updateProgress(10, t('uploading'));
        
        let response;
        
        if (uploadType === 'youtube') {
            // 使用YouTube接口
            const formData = new FormData();
            formData.append('youtube_url', youtubeInput.value.trim());
            
            response = await fetch(`${API_BASE_URL}/service/youtube-transcribe?${params}`, {
                method: 'POST',
                body: formData
            });
        } else if (uploadType === 'url') {
            // 使用URL接口
            const formData = new FormData();
            formData.append('url', urlInput.value.trim());
            
            response = await fetch(`${API_BASE_URL}/speech-to-text-url?${params}`, {
                method: 'POST',
                body: formData
            });
        } else {
            // 使用文件上传接口
            const formData = new FormData();
            formData.append('file', currentFile);
            
            response = await fetch(`${API_BASE_URL}/speech-to-text?${params}`, {
                method: 'POST',
                body: formData
            });
        }

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || errorData.message || `${t('processingFailed')}: ${response.statusText}`);
        }

        const data = await response.json();
        currentTaskId = data.identifier;
        
        updateProgress(20, t('processing'));
        const taskInfo = document.getElementById('os-taskInfo');
        if (taskInfo) {
            taskInfo.innerHTML = `<div><strong>${t('taskId')}:</strong> ${currentTaskId}</div>`;
        }

        // 开始轮询任务状态
        pollTaskStatus();
    } catch (error) {
        console.error(t('processingFailed') + ':', error);
        alert(t('processingFailed') + ': ' + error.message);
        if (startProcessBtn) startProcessBtn.disabled = false;
        if (progressSection) progressSection.style.display = 'none';
    }
}

// 收集所有参数
function collectAllParams(prefix) {
    const params = new URLSearchParams();
    
    // 主要参数
    const language = document.getElementById(`${prefix}-language`)?.value;
    const model = document.getElementById(`${prefix}-model`)?.value;
    const device = document.getElementById(`${prefix}-device`)?.value;
    
    if (language) params.append('language', language);
    if (model) params.append('model', model);
    if (device) params.append('device', device);
    
    // Whisper模型参数
    const task = document.getElementById(`${prefix}-task`)?.value;
    const device_index = document.getElementById(`${prefix}-device_index`)?.value;
    const threads = document.getElementById(`${prefix}-threads`)?.value;
    const batch_size = document.getElementById(`${prefix}-batch_size`)?.value;
    const chunk_size = document.getElementById(`${prefix}-chunk_size`)?.value;
    const compute_type = document.getElementById(`${prefix}-compute_type`)?.value;
    
    if (task) params.append('task', task);
    if (device_index) params.append('device_index', device_index);
    if (threads) params.append('threads', threads);
    if (batch_size) params.append('batch_size', batch_size);
    if (chunk_size) params.append('chunk_size', chunk_size);
    if (compute_type) params.append('compute_type', compute_type);
    
    // 对齐参数
    const align_model = document.getElementById(`${prefix}-align_model`)?.value;
    const interpolate_method = document.getElementById(`${prefix}-interpolate_method`)?.value;
    const return_char_alignments = document.getElementById(`${prefix}-return_char_alignments`)?.checked;
    
    if (align_model) params.append('align_model', align_model);
    if (interpolate_method) params.append('interpolate_method', interpolate_method);
    if (return_char_alignments) params.append('return_char_alignments', 'true');
    
    // 说话人分离参数
    const min_speakers = document.getElementById(`${prefix}-min_speakers`)?.value;
    const max_speakers = document.getElementById(`${prefix}-max_speakers`)?.value;
    
    if (min_speakers) params.append('min_speakers', min_speakers);
    if (max_speakers) params.append('max_speakers', max_speakers);
    
    // ASR选项
    const beam_size = document.getElementById(`${prefix}-beam_size`)?.value;
    const best_of = document.getElementById(`${prefix}-best_of`)?.value;
    const patience = document.getElementById(`${prefix}-patience`)?.value;
    const length_penalty = document.getElementById(`${prefix}-length_penalty`)?.value;
    const temperatures = document.getElementById(`${prefix}-temperatures`)?.value;
    const compression_ratio_threshold = document.getElementById(`${prefix}-compression_ratio_threshold`)?.value;
    const log_prob_threshold = document.getElementById(`${prefix}-log_prob_threshold`)?.value;
    const no_speech_threshold = document.getElementById(`${prefix}-no_speech_threshold`)?.value;
    const initial_prompt = document.getElementById(`${prefix}-initial_prompt`)?.value;
    const suppress_tokens = document.getElementById(`${prefix}-suppress_tokens`)?.value;
    const suppress_numerals = document.getElementById(`${prefix}-suppress_numerals`)?.checked;
    const hotwords = document.getElementById(`${prefix}-hotwords`)?.value;
    
    if (beam_size) params.append('beam_size', beam_size);
    if (best_of) params.append('best_of', best_of);
    if (patience) params.append('patience', patience);
    if (length_penalty) params.append('length_penalty', length_penalty);
    if (temperatures) params.append('temperatures', temperatures);
    if (compression_ratio_threshold) params.append('compression_ratio_threshold', compression_ratio_threshold);
    if (log_prob_threshold) params.append('log_prob_threshold', log_prob_threshold);
    if (no_speech_threshold) params.append('no_speech_threshold', no_speech_threshold);
    if (initial_prompt) params.append('initial_prompt', initial_prompt);
    if (suppress_tokens) params.append('suppress_tokens', suppress_tokens);
    if (suppress_numerals) params.append('suppress_numerals', 'true');
    if (hotwords) params.append('hotwords', hotwords);
    
    // VAD选项
    const vad_onset = document.getElementById(`${prefix}-vad_onset`)?.value;
    const vad_offset = document.getElementById(`${prefix}-vad_offset`)?.value;
    
    if (vad_onset) params.append('vad_onset', vad_onset);
    if (vad_offset) params.append('vad_offset', vad_offset);
    
    return params;
}

// 轮询任务状态
async function pollTaskStatus() {
    if (!currentTaskId) return;

    try {
        const response = await fetch(`${API_BASE_URL}/task/${currentTaskId}`);
        const data = await response.json();

        const taskInfo = document.getElementById('os-taskInfo');
        if (taskInfo) {
            taskInfo.innerHTML = `
                <div><strong>${t('taskId')}:</strong> ${currentTaskId}</div>
                <div><strong>${t('status')}:</strong> ${data.status}</div>
                <div><strong>${t('taskType')}:</strong> ${data.metadata?.task_type || 'N/A'}</div>
                ${data.metadata?.duration ? `<div><strong>${t('duration')}:</strong> ${data.metadata.duration.toFixed(2)} ${t('seconds')}</div>` : ''}
            `;
        }

        if (data.status === 'processing') {
            updateProgress(30 + Math.random() * 50, t('processing'));
            setTimeout(pollTaskStatus, 2000); // 2秒后再次查询
        } else if (data.status === 'completed') {
            updateProgress(100, t('completed'));
            await handleCompletedTask(data);
        } else if (data.status === 'failed') {
            throw new Error(data.error || t('processingFailed'));
        }
    } catch (error) {
        console.error(t('queryFailed') + ':', error);
        alert(t('queryFailed') + ': ' + error.message);
        const startProcessBtn = document.getElementById('os-startProcessBtn');
        if (startProcessBtn) startProcessBtn.disabled = false;
    }
}

// 更新进度
function updateProgress(percent, text) {
    const progressFill = document.getElementById('os-progressFill');
    const progressText = document.getElementById('os-progressText');
    if (progressFill) progressFill.style.width = percent + '%';
    if (progressText) progressText.textContent = text;
}

// 中文简繁转换函数（使用 OpenCC 库）
function convertChinese(text, conversionType) {
    if (conversionType === 'none' || !text) return text;
    
    try {
        // 使用 OpenCC 库进行转换
        if (typeof OpenCC !== 'undefined') {
            const converter = new OpenCC(conversionType);
            return converter.convert(text);
        } else {
            // 如果 OpenCC 未加载，使用简单映射
            console.warn(t('openccNotLoaded') || 'OpenCC library not loaded, using simple conversion');
            return simpleConvert(text, conversionType);
        }
    } catch (error) {
        console.error(t('conversionFailed') || 'Conversion failed:', error);
        return text;
    }
}

// 简单转换函数（备用方案，基于常见字符）
function simpleConvert(text, conversionType) {
    if (!text) return text;
    
    // 常见简繁转换映射（简化版，实际应用中建议使用完整的 OpenCC）
    const conversions = {
        't2s': {
            '繁體': '繁体', '簡體': '简体', '轉換': '转换', '識別': '识别',
            '語音': '语音', '處理': '处理', '結果': '结果', '蘋果': '苹果',
            '幹': '干', '憑': '凭', '你們': '你们', '億': '亿', '美元': '美元',
            '覺得': '觉得', '合適': '合适', '願意': '愿意', '挺身': '挺身',
            '而出': '而出', '十年': '十年', '三年': '三年', '最少': '最少',
            '需要': '需要', '如果': '如果', '大家': '大家', '为了': '为了',
            '小米': '小米', '我': '我', '都': '都', '不': '不', '成': '成',
            '啥': '啥', '能': '能', '干': '干'
        },
        's2t': {
            '繁体': '繁體', '简体': '簡體', '转换': '轉換', '识别': '識別',
            '语音': '語音', '处理': '處理', '结果': '結果', '苹果': '蘋果',
            '干': '幹', '凭': '憑', '你们': '你們', '亿': '億',
            '觉得': '覺得', '合适': '合適', '愿意': '願意'
        }
    };
    
    const map = conversions[conversionType] || {};
    let converted = text;
    
    // 按长度排序，先替换长的词
    const sortedEntries = Object.entries(map).sort((a, b) => b[0].length - a[0].length);
    
    for (const [from, to] of sortedEntries) {
        converted = converted.replace(new RegExp(from, 'g'), to);
    }
    
    return converted;
}

// 处理完成的任务
async function handleCompletedTask(data) {
    const result = data.result;
    
    // 应用中文转换
    const conversionType = document.getElementById('os-chineseConversion')?.value || 'none';
    if (conversionType !== 'none' && result.segments) {
        result.segments = result.segments.map(segment => ({
            ...segment,
            text: convertChinese(segment.text, conversionType),
            words: segment.words ? segment.words.map(word => ({
                ...word,
                word: convertChinese(word.word, conversionType)
            })) : undefined
        }));
    }
    
    currentSubtitleData = result;

    // 显示结果
    const resultSection = document.getElementById('os-resultSection');
    const progressSection = document.getElementById('os-progressSection');
    if (resultSection) resultSection.style.display = 'block';
    if (progressSection) progressSection.style.display = 'none';

    // 生成字幕
    if (result.segments) {
        const srtContent = generateSRT(result.segments);
        const vttContent = generateVTT(result.segments);
        
        // 创建字幕文件 Blob
        const srtBlob = new Blob([srtContent], { type: 'text/plain;charset=utf-8' });
        const vttBlob = new Blob([vttContent], { type: 'text/vtt;charset=utf-8' });
        
        const srtUrl = URL.createObjectURL(srtBlob);
        const vttUrl = URL.createObjectURL(vttBlob);
        
        const subtitleTrack = document.getElementById('os-subtitleTrack');
        const videoPlayer = document.getElementById('os-videoPlayer');
        if (subtitleTrack) subtitleTrack.src = vttUrl;
        if (videoPlayer && videoPlayer.textTracks[0]) {
            videoPlayer.textTracks[0].mode = 'showing';
        }

        // 获取文件名
        const uploadType = document.querySelector('input[name="uploadType"]:checked')?.value || 'file';
        let filename = 'result';
        if (uploadType === 'file' && currentFile) {
            filename = currentFile.name.replace(/\.[^/.]+$/, '');
        } else if (uploadType === 'youtube') {
            filename = 'youtube_video';
        } else if (uploadType === 'url') {
            filename = 'url_audio';
        }

        // 保存下载链接
        const osDownloadSrtBtn = document.getElementById('os-downloadSrtBtn');
        const osDownloadVttBtn = document.getElementById('os-downloadVttBtn');
        const osDownloadJsonBtn = document.getElementById('os-downloadJsonBtn');
        
        if (osDownloadSrtBtn) {
            osDownloadSrtBtn.dataset.url = srtUrl;
            osDownloadSrtBtn.dataset.filename = filename + '.srt';
        }
        if (osDownloadVttBtn) {
            osDownloadVttBtn.dataset.url = vttUrl;
            osDownloadVttBtn.dataset.filename = filename + '.vtt';
        }
        if (osDownloadJsonBtn) {
            osDownloadJsonBtn.dataset.data = JSON.stringify(result, null, 2);
            osDownloadJsonBtn.dataset.filename = filename + '.json';
        }

        // 显示转录文本
        const transcriptContent = document.getElementById('os-transcriptContent');
        if (transcriptContent) {
            displayTranscript(result.segments, transcriptContent, videoPlayer);
        }
    }

    const startProcessBtn = document.getElementById('os-startProcessBtn');
    if (startProcessBtn) startProcessBtn.disabled = false;
}

// 显示转录文本
function displayTranscript(segments) {
    transcriptContent.innerHTML = segments.map((segment, index) => {
        const startTime = formatTime(segment.start);
        const endTime = formatTime(segment.end);
        const speaker = segment.speaker ? `<div class="speaker">${t('speaker')}: ${segment.speaker}</div>` : '';
        
        return `
            <div class="transcript-segment" data-start="${segment.start}" data-end="${segment.end}">
                <div class="time">${startTime} - ${endTime}</div>
                <div class="text">${segment.text}</div>
                ${speaker}
            </div>
        `;
    }).join('');

    // 点击文本跳转到对应时间
    transcriptContent.querySelectorAll('.transcript-segment').forEach(segment => {
        segment.addEventListener('click', () => {
            const start = parseFloat(segment.dataset.start);
            videoPlayer.currentTime = start;
        });
    });

    // 视频播放时高亮对应文本
    videoPlayer.addEventListener('timeupdate', () => {
        const currentTime = videoPlayer.currentTime;
        transcriptContent.querySelectorAll('.transcript-segment').forEach(segment => {
            const start = parseFloat(segment.dataset.start);
            const end = parseFloat(segment.dataset.end);
            if (currentTime >= start && currentTime <= end) {
                segment.classList.add('active');
            } else {
                segment.classList.remove('active');
            }
        });
    });
}

// 生成 SRT 字幕
function generateSRT(segments) {
    return segments.map((segment, index) => {
        const start = formatTimeSRT(segment.start);
        const end = formatTimeSRT(segment.end);
        return `${index + 1}\n${start} --> ${end}\n${segment.text}\n`;
    }).join('\n');
}

// 生成 VTT 字幕
function generateVTT(segments) {
    const header = 'WEBVTT\n\n';
    const content = segments.map((segment, index) => {
        const start = formatTimeVTT(segment.start);
        const end = formatTimeVTT(segment.end);
        return `${index + 1}\n${start} --> ${end}\n${segment.text}\n`;
    }).join('\n');
    return header + content;
}

// 时间格式化
function formatTime(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

function formatTimeSRT(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    const millis = Math.floor((seconds % 1) * 1000);
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')},${millis.toString().padStart(3, '0')}`;
}

function formatTimeVTT(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    const millis = Math.floor((seconds % 1) * 1000);
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}.${millis.toString().padStart(3, '0')}`;
}

// 更新进度
function updateProgress(percent, text) {
    progressFill.style.width = percent + '%';
    progressText.textContent = text;
}

// 下载字幕
function downloadSubtitle(format, prefix = '') {
    const btnId = prefix ? `${prefix}-download${format === 'srt' ? 'Srt' : 'Vtt'}Btn` : `download${format === 'srt' ? 'Srt' : 'Vtt'}Btn`;
    const btn = document.getElementById(btnId);
    if (!btn || !btn.dataset.url) return;

    const a = document.createElement('a');
    a.href = btn.dataset.url;
    a.download = btn.dataset.filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

// 下载 JSON
function downloadJson(prefix = '') {
    const btnId = prefix ? `${prefix}-downloadJsonBtn` : 'downloadJsonBtn';
    const btn = document.getElementById(btnId);
    if (!btn || !btn.dataset.data) return;

    const blob = new Blob([btn.dataset.data], { type: 'application/json;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = btn.dataset.filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

