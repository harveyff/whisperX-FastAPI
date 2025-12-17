// ============================================================================
// 文件版本标识 - 用于确认浏览器加载的是最新版本
// ============================================================================
// 每次修改文件时更新这个hash值，确保浏览器加载最新版本
// Hash值基于文件关键内容生成，每次修改后更新
const APP_JS_HASH = 'v2.1.0-complete-rewrite-safe'; // 文件内容hash标识
const APP_JS_VERSION = '2.0.7-' + Date.now();
const APP_JS_BUILD_TIME = new Date().toISOString();

console.log('========================================');
console.log('[App.js] 文件已加载');
console.log('[App.js] 版本:', APP_JS_VERSION);
console.log('[App.js] Hash标识:', APP_JS_HASH);
console.log('[App.js] 构建时间:', APP_JS_BUILD_TIME);
console.log('[App.js] 加载时间:', new Date().toLocaleString());
console.log('[App.js] 如果Hash标识不是 v2.1.0-complete-rewrite-safe，说明文件未更新');
console.log('========================================');

// 语法检查：如果这行代码能执行，说明前面的代码没有语法错误
try {
    console.log('[App.js] 语法检查通过');
} catch (e) {
    console.error('[App.js] 语法错误:', e);
}

// ============================================================================
// API 基础配置
// ============================================================================
// 从当前host获取API基础URL
const API_BASE_URL = (() => {
    const protocol = window.location.protocol;
    const hostname = window.location.hostname;
    const port = window.location.port;
    
    // 如果当前端口是80或443（HTTP/HTTPS默认端口），或者没有端口，尝试使用8000
    // 否则使用相同端口（假设前后端在同一服务器）
    let apiPort = port;
    if (!port || port === '80' || port === '443') {
        apiPort = '8000';
    }
    
    // 如果是localhost或127.0.0.1，使用8000端口
    if (hostname === 'localhost' || hostname === '127.0.0.1') {
        apiPort = '8000';
    }
    
    return `${protocol}//${hostname}:${apiPort}`;
})();
console.log('[App.js] API_BASE_URL:', API_BASE_URL);
console.log('[App.js] 当前页面URL:', window.location.href);

// ============================================================================
// 全局状态变量
// ============================================================================
let currentTaskId = null;
let currentFile = null;
let currentVideoUrl = null;
let currentSubtitleData = null;
let isProcessing = false; // 防止重复提交
let videoLoadTimeout = null; // 视频加载超时
let isHandlingFile = false; // 防止重复处理文件
let isPreviewing = false; // 防止重复预览
let isCleaningUp = false; // 防止重复清理

// ============================================================================
// 工具函数
// ============================================================================

// 多语言支持
function t(key) {
    return window.t ? window.t(key) : key;
}

// 获取 DOM 元素（安全获取）
function getElement(id) {
    return document.getElementById(id);
}

// 更新进度
function updateProgress(percent, text) {
    const progressFill = getElement('os-progressFill');
    const progressText = getElement('os-progressText');
    if (progressFill) progressFill.style.width = percent + '%';
    if (progressText) progressText.textContent = text;
}

// 时间格式化函数
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

// ============================================================================
// 文件处理相关函数
// ============================================================================

// 处理文件选择
function handleFileSelect(e) {
    // 防止重复调用
    if (isHandlingFile) {
        console.warn('[handleFileSelect] 正在处理文件，忽略重复调用');
        return;
    }
    
    const files = e.target.files;
    if (files && files.length > 0) {
        // 使用 setTimeout 确保异步执行
        setTimeout(() => {
            handleFile(files[0]);
        }, 0);
    }
}

// 处理文件 - 完全重写，避免任何递归
function handleFile(file) {
    // 防止重复调用 - 在函数最开始检查
    if (isHandlingFile) {
        console.warn('[handleFile] 正在处理文件，忽略重复调用');
        return;
    }
    
    if (!file) {
        console.warn('[handleFile] 文件对象为空');
        return;
    }
    
    // 立即设置标志，防止递归
    isHandlingFile = true;
    
    console.log('[handleFile] 处理文件:', file.name, '大小:', (file.size / 1024 / 1024).toFixed(2), 'MB', '类型:', file.type);
    
    // 保存文件引用
    currentFile = file;
    
    // 预先计算所有需要的值
    const fileSelectedText = (window.t ? window.t('fileSelected') : 'fileSelected');
    const fileSizeText = (window.t ? window.t('fileSize') : 'fileSize');
    const fileSizeMB = (file.size / 1024 / 1024).toFixed(2);
    const fileName = file.name;
    
    // 使用 requestIdleCallback 或 setTimeout 确保完全异步
    const scheduleUpdate = (callback) => {
        if (window.requestIdleCallback) {
            window.requestIdleCallback(callback, { timeout: 100 });
        } else {
            setTimeout(callback, 0);
        }
    };
    
    scheduleUpdate(() => {
        // 再次检查标志
        if (!isHandlingFile) {
            return;
        }
        
        try {
            // DOM 操作
            const uploadArea = getElement('uploadArea');
            if (uploadArea) {
                const existingContent = uploadArea.querySelector('.upload-content');
                if (existingContent) {
                    existingContent.remove();
                }
                
                const newContent = document.createElement('div');
                newContent.className = 'upload-content';
                
                const nameP = document.createElement('p');
                nameP.textContent = `${fileSelectedText}: ${fileName}`;
                
                const sizeP = document.createElement('p');
                sizeP.className = 'file-hint';
                sizeP.textContent = `${fileSizeText}: ${fileSizeMB} MB`;
                
                newContent.appendChild(nameP);
                newContent.appendChild(sizeP);
                uploadArea.appendChild(newContent);
            }
            
            const configSection = getElement('os-configSection');
            if (configSection) {
                configSection.style.display = 'block';
            }
            
            // 处理视频预览 - 使用独立的异步调用
            scheduleUpdate(() => {
                try {
                    if (!isPreviewing && !isCleaningUp) {
                        handleVideoPreview(file);
                    }
                } catch (e) {
                    console.error('[handleFile] 视频预览处理失败:', e);
                }
                
                // 重置标志
                scheduleUpdate(() => {
                    isHandlingFile = false;
                });
            });
            
        } catch (error) {
            console.error('[handleFile] 处理文件时出错:', error);
            console.error('[handleFile] 错误堆栈:', error.stack);
            scheduleUpdate(() => {
                isHandlingFile = false;
            });
            alert('处理文件时出错: ' + (error.message || String(error)));
        }
    });
}

// 清理视频资源
function cleanupVideoResources() {
    if (isCleaningUp) {
        console.warn('[cleanupVideoResources] 正在清理中，跳过重复调用');
        return;
    }
    
    isCleaningUp = true;
    
    try {
        const videoPlayer = getElement('os-videoPlayer');
        
        if (videoLoadTimeout) {
            clearTimeout(videoLoadTimeout);
            videoLoadTimeout = null;
        }
        
        if (videoPlayer && currentTimeUpdateHandler) {
            try {
                videoPlayer.removeEventListener('timeupdate', currentTimeUpdateHandler);
                videoPlayer.removeEventListener('error', currentTimeUpdateHandler);
                currentTimeUpdateHandler = null;
            } catch (e) {
                console.warn('[cleanupVideoResources] 移除监听器失败:', e);
            }
        }
        
        if (videoPlayer) {
            try {
                videoPlayer.pause();
                const parent = videoPlayer.parentNode;
                if (parent) {
                    const newVideoPlayer = videoPlayer.cloneNode(false);
                    parent.replaceChild(newVideoPlayer, videoPlayer);
                }
                const videoSource = getElement('os-videoSource');
                if (videoSource) {
                    videoSource.src = '';
                }
            } catch (e) {
                console.warn('[cleanupVideoResources] 清理视频元素失败:', e);
            }
        }
        
        if (currentVideoUrl) {
            try {
                URL.revokeObjectURL(currentVideoUrl);
            } catch (e) {
                console.warn('[cleanupVideoResources] 清理URL失败:', e);
            }
            currentVideoUrl = null;
        }
    } finally {
        isCleaningUp = false;
    }
}

// 处理视频预览
function handleVideoPreview(file) {
    if (isPreviewing) {
        console.warn('[handleVideoPreview] 正在预览中，跳过重复调用');
        return;
    }
    
    console.log('[handleVideoPreview] 函数被调用，文件类型:', file?.type, '文件大小:', file?.size ? (file.size / 1024 / 1024).toFixed(2) + ' MB' : 'unknown');
    
    if (!file) {
        console.warn('[handleVideoPreview] 文件对象为空');
        return;
    }
    
    const videoPlayer = getElement('os-videoPlayer');
    const videoSource = getElement('os-videoSource');
    
    if (!videoPlayer || !videoSource) {
        console.log('[handleVideoPreview] 视频元素未找到，跳过预览');
        return;
    }
    
    const videoContainer = videoPlayer.closest('.video-player-container');
    
    if (file.type && file.type.startsWith('video/')) {
        isPreviewing = true;
        
        try {
            if (!isCleaningUp) {
                cleanupVideoResources();
            }
            
            const url = URL.createObjectURL(file);
            currentVideoUrl = url;
            videoSource.src = url;
            
            if (videoContainer) {
                videoContainer.style.display = 'block';
            }
            
            videoLoadTimeout = setTimeout(() => {
                try {
                    if (videoPlayer && videoSource && videoSource.src) {
                        videoPlayer.load();
                        console.log('[handleVideoPreview] 视频加载完成');
                        const errorHandler = () => {
                            console.warn('[handleVideoPreview] 视频加载错误');
                            if (videoPlayer) {
                                videoPlayer.removeEventListener('error', errorHandler);
                            }
                        };
                        videoPlayer.addEventListener('error', errorHandler, { once: true });
                    }
                } catch (e) {
                    console.error('[handleVideoPreview] 视频加载失败:', e);
                } finally {
                    videoLoadTimeout = null;
                    isPreviewing = false;
                }
            }, 200);
            
        } catch (error) {
            isPreviewing = false;
            console.error('[handleVideoPreview] 创建视频预览失败:', error);
            if (!isCleaningUp) {
                cleanupVideoResources();
            }
        }
    } else {
        if (videoContainer) {
            videoContainer.style.display = 'none';
        }
        if (!isCleaningUp) {
            cleanupVideoResources();
        }
        isPreviewing = false;
    }
}

// ============================================================================
// 拖拽处理函数
// ============================================================================

function handleDragOver(e) {
    e.preventDefault();
    const uploadArea = getElement('uploadArea');
    if (uploadArea) {
        uploadArea.classList.add('dragover');
    }
}

function handleDragLeave(e) {
    e.preventDefault();
    const uploadArea = getElement('uploadArea');
    if (uploadArea) {
        uploadArea.classList.remove('dragover');
    }
}

function handleDrop(e) {
    e.preventDefault();
    const uploadArea = getElement('uploadArea');
    if (uploadArea) {
        uploadArea.classList.remove('dragover');
    }
    
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
        // 使用 setTimeout 确保异步执行，避免递归
        setTimeout(() => {
            handleFile(files[0]);
        }, 0);
    }
}

// ============================================================================
// 参数收集函数
// ============================================================================

// 收集所有参数
function collectAllParams(prefix) {
    console.log('[collectAllParams] 开始收集参数，前缀:', prefix);
    const params = new URLSearchParams();
    
    // 辅助函数：安全获取值
    const getValue = (id, defaultValue = null) => {
        const element = getElement(id);
        if (!element) {
            console.warn(`[collectAllParams] 元素未找到: ${id}`);
            return defaultValue;
        }
        // 对于 input 元素，即使 value 是空字符串或 "0"，也应该返回实际值
        // 只有当元素不存在时才返回 defaultValue
        const value = element.value !== undefined ? element.value : defaultValue;
        console.log(`[collectAllParams] ${id}:`, value, '(元素存在:', !!element, ', 类型:', element.type || element.tagName, ')');
        return value;
    };
    
    // 主要参数
    const language = getValue(`${prefix}-language`);
    const model = getValue(`${prefix}-model`);
    const device = getValue(`${prefix}-device`);
    
    if (language !== null && language !== '') params.append('language', language);
    if (model !== null && model !== '') params.append('model', model);
    if (device !== null && device !== '') params.append('device', device);
    
    // Whisper模型参数
    const task = getValue(`${prefix}-task`);
    const device_index = getValue(`${prefix}-device_index`);
    const threads = getValue(`${prefix}-threads`);
    const batch_size = getValue(`${prefix}-batch_size`);
    const chunk_size = getValue(`${prefix}-chunk_size`);
    const compute_type = getValue(`${prefix}-compute_type`);
    
    // 对于这些参数，即使值是 "0" 或空字符串，也应该发送（让后端决定是否使用默认值）
    if (task !== null && task !== '') params.append('task', task);
    // 对于数字类型，即使值是 "0" 也要发送（使用实际值或默认值）
    if (device_index !== null) params.append('device_index', device_index || '0');
    if (threads !== null) params.append('threads', threads || '0');
    if (batch_size !== null) params.append('batch_size', batch_size || '8');
    if (chunk_size !== null) params.append('chunk_size', chunk_size || '20');
    if (compute_type !== null && compute_type !== '') params.append('compute_type', compute_type);
    
    // 对齐参数
    const align_model = getValue(`${prefix}-align_model`);
    const interpolate_method = getValue(`${prefix}-interpolate_method`);
    const return_char_alignments = getElement(`${prefix}-return_char_alignments`)?.checked;
    
    if (align_model !== null && align_model !== '') params.append('align_model', align_model);
    if (interpolate_method !== null && interpolate_method !== '') params.append('interpolate_method', interpolate_method);
    if (return_char_alignments) params.append('return_char_alignments', 'true');
    
    // 说话人分离参数
    const min_speakers = getValue(`${prefix}-min_speakers`);
    const max_speakers = getValue(`${prefix}-max_speakers`);
    
    if (min_speakers !== null && min_speakers !== '') params.append('min_speakers', min_speakers);
    if (max_speakers !== null && max_speakers !== '') params.append('max_speakers', max_speakers);
    
    // ASR选项
    const beam_size = getValue(`${prefix}-beam_size`);
    const best_of = getValue(`${prefix}-best_of`);
    const patience = getValue(`${prefix}-patience`);
    const length_penalty = getValue(`${prefix}-length_penalty`);
    const temperatures = getValue(`${prefix}-temperatures`);
    const compression_ratio_threshold = getValue(`${prefix}-compression_ratio_threshold`);
    const log_prob_threshold = getValue(`${prefix}-log_prob_threshold`);
    const no_speech_threshold = getValue(`${prefix}-no_speech_threshold`);
    const initial_prompt = getValue(`${prefix}-initial_prompt`);
    const suppress_tokens = getValue(`${prefix}-suppress_tokens`);
    const suppress_numerals = getElement(`${prefix}-suppress_numerals`)?.checked;
    const hotwords = getValue(`${prefix}-hotwords`);
    
    if (beam_size !== null && beam_size !== '') params.append('beam_size', beam_size);
    if (best_of !== null && best_of !== '') params.append('best_of', best_of);
    if (patience !== null && patience !== '') params.append('patience', patience);
    if (length_penalty !== null && length_penalty !== '') params.append('length_penalty', length_penalty);
    if (temperatures !== null && temperatures !== '') params.append('temperatures', temperatures);
    if (compression_ratio_threshold !== null && compression_ratio_threshold !== '') params.append('compression_ratio_threshold', compression_ratio_threshold);
    if (log_prob_threshold !== null && log_prob_threshold !== '') params.append('log_prob_threshold', log_prob_threshold);
    if (no_speech_threshold !== null && no_speech_threshold !== '') params.append('no_speech_threshold', no_speech_threshold);
    if (initial_prompt !== null && initial_prompt !== '') params.append('initial_prompt', initial_prompt);
    if (suppress_tokens !== null && suppress_tokens !== '') params.append('suppress_tokens', suppress_tokens);
    if (suppress_numerals) params.append('suppress_numerals', 'true');
    if (hotwords !== null && hotwords !== '') params.append('hotwords', hotwords);
    
    // VAD选项
    const vad_onset = getValue(`${prefix}-vad_onset`);
    const vad_offset = getValue(`${prefix}-vad_offset`);
    
    if (vad_onset !== null && vad_onset !== '') params.append('vad_onset', vad_onset);
    if (vad_offset !== null && vad_offset !== '') params.append('vad_offset', vad_offset);
    
    console.log('[collectAllParams] 收集到的参数:', params.toString());
    return params;
}

// ============================================================================
// 处理流程相关函数
// ============================================================================

// 开始处理
async function startProcessing() {
    // 防止重复提交
    if (isProcessing) {
        console.warn('[startProcessing] 正在处理中，忽略重复请求');
        return;
    }
    
    // 检查上传类型
    const uploadType = document.querySelector('input[name="uploadType"]:checked')?.value || 'file';
    const urlInput = getElement('urlInput');
    const youtubeInput = getElement('youtubeInput');
    
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
    
    // 检查文件大小（可选，如果文件太大可以提示）
    if (uploadType === 'file' && currentFile) {
        const fileSizeMB = currentFile.size / 1024 / 1024;
        console.log('[startProcessing] 文件大小:', fileSizeMB.toFixed(2), 'MB');
        if (fileSizeMB > 500) {
            if (!confirm(`文件较大 (${fileSizeMB.toFixed(2)} MB)，上传可能需要较长时间，是否继续？`)) {
                return;
            }
        }
    }

    // 更新UI状态
    const startProcessBtn = getElement('os-startProcessBtn');
    const configSection = document.querySelector('#one-step-tab .section:nth-of-type(2)');
    const progressSection = getElement('os-progressSection');
    const resultSection = getElement('os-resultSection');
    
    isProcessing = true;
    if (startProcessBtn) startProcessBtn.disabled = true;
    if (configSection) configSection.style.display = 'none';
    if (progressSection) progressSection.style.display = 'block';
    if (resultSection) resultSection.style.display = 'none';

    // 收集参数
    const params = collectAllParams('os');
    
    try {
        updateProgress(10, t('uploading'));
        
        let response;
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 600000); // 10分钟超时
        
        try {
            if (uploadType === 'youtube') {
                // 使用YouTube接口
                const formData = new FormData();
                formData.append('youtube_url', youtubeInput.value.trim());
                
                response = await fetch(`${API_BASE_URL}/service/youtube-transcribe?${params}`, {
                    method: 'POST',
                    body: formData,
                    signal: controller.signal
                });
            } else if (uploadType === 'url') {
                // 使用URL接口
                const formData = new FormData();
                formData.append('url', urlInput.value.trim());
                
                response = await fetch(`${API_BASE_URL}/speech-to-text-url?${params}`, {
                    method: 'POST',
                    body: formData,
                    signal: controller.signal
                });
            } else {
                // 使用文件上传接口
                const formData = new FormData();
                formData.append('file', currentFile);
                
                console.log('[startProcessing] 开始上传文件，大小:', (currentFile.size / 1024 / 1024).toFixed(2), 'MB');
                
                response = await fetch(`${API_BASE_URL}/speech-to-text?${params}`, {
                    method: 'POST',
                    body: formData,
                    signal: controller.signal
                });
            }
            
            clearTimeout(timeoutId);
        } catch (fetchError) {
            clearTimeout(timeoutId);
            if (fetchError.name === 'AbortError') {
                throw new Error('上传超时，请检查网络连接或文件大小');
            }
            throw fetchError;
        }

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            const errorMessage = errorData.detail || errorData.message || `${t('processingFailed')}: ${response.statusText}`;
            console.error('[startProcessing] 服务器错误:', response.status, errorMessage);
            throw new Error(errorMessage);
        }

        const data = await response.json();
        currentTaskId = data.identifier;
        
        updateProgress(20, t('processing'));
        const taskInfo = getElement('os-taskInfo');
        if (taskInfo) {
            taskInfo.innerHTML = `<div><strong>${t('taskId')}:</strong> ${currentTaskId}</div>`;
        }

        // 开始轮询任务状态
        pollTaskStatus();
    } catch (error) {
        console.error('[startProcessing] 处理失败:', error);
        isProcessing = false;
        alert(t('processingFailed') + ': ' + error.message);
        if (startProcessBtn) startProcessBtn.disabled = false;
        if (progressSection) progressSection.style.display = 'none';
    }
}

// 轮询任务状态
async function pollTaskStatus() {
    if (!currentTaskId) return;

    try {
        const response = await fetch(`${API_BASE_URL}/task/${currentTaskId}`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();

        const taskInfo = getElement('os-taskInfo');
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
        console.error('[pollTaskStatus] 查询失败:', error);
        isProcessing = false;
        alert(t('queryFailed') + ': ' + error.message);
        const startProcessBtn = getElement('os-startProcessBtn');
        if (startProcessBtn) startProcessBtn.disabled = false;
    }
}

// ============================================================================
// 字幕生成和处理函数
// ============================================================================

// 生成 SRT 字幕
function generateSRT(segments) {
    if (!segments || !Array.isArray(segments)) return '';
    return segments.map((segment, index) => {
        const start = formatTimeSRT(segment.start);
        const end = formatTimeSRT(segment.end);
        return `${index + 1}\n${start} --> ${end}\n${segment.text}\n`;
    }).join('\n');
}

// 生成 VTT 字幕
function generateVTT(segments) {
    if (!segments || !Array.isArray(segments)) return 'WEBVTT\n\n';
    const header = 'WEBVTT\n\n';
    const content = segments.map((segment, index) => {
        const start = formatTimeVTT(segment.start);
        const end = formatTimeVTT(segment.end);
        return `${index + 1}\n${start} --> ${end}\n${segment.text}\n`;
    }).join('\n');
    return header + content;
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
    if (!data || !data.result) {
        console.error('[handleCompletedTask] Invalid task data:', data);
        isProcessing = false;
        return;
    }
    
    // 重置处理状态
    isProcessing = false;
    
    // 先清理之前的监听器和状态
    cleanupVideoResources();
    
    // 重置转录显示标志，允许重新显示
    transcriptDisplayed = false;
    lastUpdateTime = 0;
    
    const result = data.result;
    
    // 应用中文转换
    const conversionType = getElement('os-chineseConversion')?.value || 'none';
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
    const resultSection = getElement('os-resultSection');
    const progressSection = getElement('os-progressSection');
    if (resultSection) resultSection.style.display = 'block';
    if (progressSection) progressSection.style.display = 'none';

    // 生成字幕
    if (result.segments && Array.isArray(result.segments)) {
        const srtContent = generateSRT(result.segments);
        const vttContent = generateVTT(result.segments);
        
        // 创建字幕文件 Blob
        const srtBlob = new Blob([srtContent], { type: 'text/plain;charset=utf-8' });
        const vttBlob = new Blob([vttContent], { type: 'text/vtt;charset=utf-8' });
        
        const srtUrl = URL.createObjectURL(srtBlob);
        const vttUrl = URL.createObjectURL(vttBlob);
        
        const subtitleTrack = getElement('os-subtitleTrack');
        const videoPlayer = getElement('os-videoPlayer');
        if (subtitleTrack) subtitleTrack.src = vttUrl;
        if (videoPlayer && videoPlayer.textTracks && videoPlayer.textTracks[0]) {
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
        const osDownloadSrtBtn = getElement('os-downloadSrtBtn');
        const osDownloadVttBtn = getElement('os-downloadVttBtn');
        const osDownloadJsonBtn = getElement('os-downloadJsonBtn');
        
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
        const transcriptContent = getElement('os-transcriptContent');
        if (transcriptContent) {
            displayTranscript(result.segments, transcriptContent, videoPlayer);
        }
    }

    const startProcessBtn = getElement('os-startProcessBtn');
    if (startProcessBtn) {
        startProcessBtn.disabled = false;
    }
    
    // 确保处理状态已重置
    isProcessing = false;
}

// 视频时间更新处理函数（全局变量，用于正确移除事件监听器）
let currentTimeUpdateHandler = null;
let transcriptDisplayed = false; // 防止重复显示
let lastUpdateTime = 0; // 节流：上次更新时间
const UPDATE_THROTTLE_MS = 100; // 节流间隔：100毫秒

// 显示转录文本
function displayTranscript(segments, transcriptContent, videoPlayer) {
    if (!segments || !Array.isArray(segments) || !transcriptContent) {
        console.warn('[displayTranscript] 参数无效，跳过显示');
        return;
    }
    
    // 防止重复调用
    if (transcriptDisplayed) {
        console.warn('[displayTranscript] 转录文本已显示，跳过重复显示');
        return;
    }
    
    console.log('[displayTranscript] 开始显示转录文本，段落数:', segments.length);
    transcriptDisplayed = true;
    
    // 先清除之前的内容，避免重复添加事件监听器
    transcriptContent.innerHTML = '';
    
    // 创建文档片段以提高性能
    const fragment = document.createDocumentFragment();
    const segmentsArray = segments.map((segment, index) => {
        const startTime = formatTime(segment.start);
        const endTime = formatTime(segment.end);
        const speaker = segment.speaker ? `<div class="speaker">${t('speaker')}: ${segment.speaker}</div>` : '';
        
        const segmentDiv = document.createElement('div');
        segmentDiv.className = 'transcript-segment';
        segmentDiv.setAttribute('data-start', segment.start);
        segmentDiv.setAttribute('data-end', segment.end);
        segmentDiv.innerHTML = `
            <div class="time">${startTime} - ${endTime}</div>
            <div class="text">${segment.text}</div>
            ${speaker}
        `;
        
        // 添加点击事件
        segmentDiv.addEventListener('click', () => {
            const start = parseFloat(segmentDiv.dataset.start);
            if (videoPlayer && !isNaN(start)) {
                videoPlayer.currentTime = start;
            }
        });
        
        return segmentDiv;
    });
    
    // 一次性添加到文档片段
    segmentsArray.forEach(seg => fragment.appendChild(seg));
    transcriptContent.appendChild(fragment);

    // 视频播放时高亮对应文本
    if (videoPlayer) {
        // 移除所有可能存在的 timeupdate 监听器
        const oldHandler = currentTimeUpdateHandler;
        if (oldHandler) {
            try {
                videoPlayer.removeEventListener('timeupdate', oldHandler);
                console.log('[displayTranscript] 已移除旧的视频时间更新监听器');
            } catch (e) {
                console.warn('[displayTranscript] 移除旧监听器时出错:', e);
            }
            currentTimeUpdateHandler = null;
        }
        
        // 重置节流时间
        lastUpdateTime = 0;
        
        // 创建新的事件处理函数（带节流）
        currentTimeUpdateHandler = () => {
            // 节流：限制更新频率
            const now = Date.now();
            if (now - lastUpdateTime < UPDATE_THROTTLE_MS) {
                return;
            }
            lastUpdateTime = now;
            
            // 安全检查
            if (!videoPlayer || !transcriptContent) {
                return;
            }
            
            const currentTime = videoPlayer.currentTime;
            if (isNaN(currentTime) || !isFinite(currentTime)) {
                return;
            }
            
            try {
                const segments = transcriptContent.querySelectorAll('.transcript-segment');
                if (!segments || segments.length === 0) {
                    return;
                }
                
                // 批量更新，避免频繁的 DOM 操作
                segments.forEach(segment => {
                    const start = parseFloat(segment.dataset.start);
                    const end = parseFloat(segment.dataset.end);
                    
                    if (!isNaN(start) && !isNaN(end) && currentTime >= start && currentTime <= end) {
                        if (!segment.classList.contains('active')) {
                            segment.classList.add('active');
                        }
                    } else {
                        if (segment.classList.contains('active')) {
                            segment.classList.remove('active');
                        }
                    }
                });
            } catch (e) {
                console.error('[displayTranscript] 更新高亮时出错:', e);
                // 如果出错，移除监听器防止无限循环
                if (currentTimeUpdateHandler) {
                    try {
                        videoPlayer.removeEventListener('timeupdate', currentTimeUpdateHandler);
                        currentTimeUpdateHandler = null;
                    } catch (removeError) {
                        console.error('[displayTranscript] 移除监听器失败:', removeError);
                    }
                }
            }
        };
        
        // 添加新的事件监听器
        try {
            videoPlayer.addEventListener('timeupdate', currentTimeUpdateHandler, { passive: true });
            console.log('[displayTranscript] 已添加视频时间更新监听器（带节流）');
        } catch (e) {
            console.error('[displayTranscript] 添加监听器时出错:', e);
        }
    }
}

// ============================================================================
// 下载相关函数
// ============================================================================

// 下载字幕
function downloadSubtitle(format, prefix = '') {
    const btnId = prefix ? `${prefix}-download${format === 'srt' ? 'Srt' : 'Vtt'}Btn` : `download${format === 'srt' ? 'Srt' : 'Vtt'}Btn`;
    const btn = getElement(btnId);
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
    const btn = getElement(btnId);
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

// ============================================================================
// 初始化函数
// ============================================================================

// Tab切换处理函数
function handleTabSwitch(btn) {
    const tab = btn.dataset.tab;
    console.log('[Tab切换] 切换到:', tab, '按钮:', btn);
    
    if (!tab) {
        console.warn('[Tab切换] Tab数据属性未找到');
        return;
    }
    
    // 更新Tab按钮状态
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    
    // 更新Tab内容
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
        content.style.display = 'none';
    });
    
    const targetTab = document.getElementById(`${tab}-tab`);
    if (targetTab) {
        targetTab.classList.add('active');
        targetTab.style.display = 'block';
        console.log('[Tab切换] Tab内容已显示:', tab);
    } else {
        console.warn('[Tab切换] Tab内容未找到:', `${tab}-tab`);
    }
}

// 折叠参数组处理函数
function handleCollapsibleToggle(e) {
    const header = e.currentTarget;
    const content = header.nextElementSibling;
    const icon = header.querySelector('.toggle-icon');
    
    if (content && content.classList.contains('collapsible-content')) {
        const isHidden = content.style.display === 'none' || !content.style.display;
        content.style.display = isHidden ? 'block' : 'none';
        
        if (icon) {
            icon.textContent = isHidden ? '▲' : '▼';
        }
        
        console.log('[折叠参数组] 切换状态:', isHidden ? '展开' : '折叠', '元素:', header.querySelector('span')?.textContent);
    }
}

// 初始化事件监听器
function initEventListeners() {
    console.log('[initEventListeners] 开始初始化事件监听器');
    
    // Tab 切换功能 - 直接绑定到每个按钮
    document.querySelectorAll('.tab-btn').forEach((btn, index) => {
        // 先移除旧的事件监听器（如果存在）
        const newBtn = btn.cloneNode(true);
        btn.parentNode.replaceChild(newBtn, btn);
        // 添加新的事件监听器
        newBtn.addEventListener('click', () => handleTabSwitch(newBtn));
        console.log(`[initEventListeners] Tab按钮 ${index + 1} 事件已绑定:`, newBtn.dataset.tab);
    });
    
    if (document.querySelectorAll('.tab-btn').length === 0) {
        console.warn('[initEventListeners] 未找到Tab按钮');
    }
    
    // 折叠参数组功能 - 直接绑定到每个header
    document.querySelectorAll('.collapsible-header').forEach((header, index) => {
        // 克隆节点以移除所有事件监听器
        const newHeader = header.cloneNode(true);
        header.parentNode.replaceChild(newHeader, header);
        // 添加新的事件监听器
        newHeader.addEventListener('click', handleCollapsibleToggle);
        const headerText = newHeader.querySelector('span')?.textContent || `参数组${index + 1}`;
        console.log(`[initEventListeners] 折叠参数组 ${index + 1} 事件已绑定:`, headerText);
    });
    
    const collapsibleHeaders = document.querySelectorAll('.collapsible-header');
    if (collapsibleHeaders.length === 0) {
        console.warn('[initEventListeners] 未找到折叠参数组');
    } else {
        console.log(`[initEventListeners] 找到 ${collapsibleHeaders.length} 个折叠参数组`);
    }
    
    // 文件上传区域 - 使用克隆节点避免重复绑定
    const uploadArea = getElement('uploadArea');
    const fileInput = getElement('fileInput');
    
    if (uploadArea && fileInput) {
        // 克隆节点以移除所有旧的事件监听器
        const newUploadArea = uploadArea.cloneNode(true);
        uploadArea.parentNode.replaceChild(newUploadArea, uploadArea);
        
        const newFileInput = fileInput.cloneNode(true);
        fileInput.parentNode.replaceChild(newFileInput, fileInput);
        
        // 重新绑定事件监听器
        newUploadArea.addEventListener('click', () => {
            if (!isHandlingFile) {
                newFileInput.click();
            }
        });
        newUploadArea.addEventListener('dragover', handleDragOver);
        newUploadArea.addEventListener('dragleave', handleDragLeave);
        newUploadArea.addEventListener('drop', handleDrop);
        newFileInput.addEventListener('change', handleFileSelect);
    }
    
    // URL/YouTube URL 上传类型切换
    const uploadTypeRadios = document.querySelectorAll('input[name="uploadType"]');
    const urlInput = getElement('urlInput');
    const youtubeInput = getElement('youtubeInput');
    
    console.log('[initEventListeners] 找到上传类型单选按钮:', uploadTypeRadios.length);
    console.log('[initEventListeners] urlInput 元素:', !!urlInput);
    console.log('[initEventListeners] youtubeInput 元素:', !!youtubeInput);
    
    // 处理上传类型切换的函数
    const handleUploadTypeChange = (uploadType) => {
        console.log('[handleUploadTypeChange] 切换上传类型:', uploadType);
        if (urlInput && youtubeInput) {
            if (uploadType === 'url') {
                urlInput.style.display = 'block';
                youtubeInput.style.display = 'none';
                console.log('[handleUploadTypeChange] 显示 URL 输入框');
            } else if (uploadType === 'youtube') {
                urlInput.style.display = 'none';
                youtubeInput.style.display = 'block';
                console.log('[handleUploadTypeChange] 显示 YouTube 输入框');
            } else {
                urlInput.style.display = 'none';
                youtubeInput.style.display = 'none';
                console.log('[handleUploadTypeChange] 隐藏所有输入框');
            }
        } else {
            console.warn('[handleUploadTypeChange] urlInput 或 youtubeInput 未找到');
        }
    };
    
    if (uploadTypeRadios.length > 0) {
        uploadTypeRadios.forEach(radio => {
            radio.addEventListener('change', (e) => {
                const uploadType = e.target.value;
                handleUploadTypeChange(uploadType);
            });
        });
        
        // 初始化显示状态
        const checkedRadio = document.querySelector('input[name="uploadType"]:checked');
        if (checkedRadio) {
            const uploadType = checkedRadio.value;
            console.log('[initEventListeners] 初始上传类型:', uploadType);
            handleUploadTypeChange(uploadType);
        } else {
            console.warn('[initEventListeners] 未找到选中的上传类型单选按钮');
            // 默认隐藏所有输入框
            if (urlInput && youtubeInput) {
                urlInput.style.display = 'none';
                youtubeInput.style.display = 'none';
            }
        }
    } else {
        console.warn('[initEventListeners] 未找到上传类型单选按钮');
    }
    
    // 开始处理按钮
    const startProcessBtn = getElement('os-startProcessBtn');
    if (startProcessBtn) {
        startProcessBtn.addEventListener('click', startProcessing);
    }
    
    // 下载按钮
    const osDownloadSrtBtn = getElement('os-downloadSrtBtn');
    const osDownloadVttBtn = getElement('os-downloadVttBtn');
    const osDownloadJsonBtn = getElement('os-downloadJsonBtn');
    
    if (osDownloadSrtBtn) {
        osDownloadSrtBtn.addEventListener('click', () => downloadSubtitle('srt', 'os'));
    }
    if (osDownloadVttBtn) {
        osDownloadVttBtn.addEventListener('click', () => downloadSubtitle('vtt', 'os'));
    }
    if (osDownloadJsonBtn) {
        osDownloadJsonBtn.addEventListener('click', () => downloadJson('os'));
    }
    
    // 语言切换
    const languageSelector = getElement('languageSelector');
    if (languageSelector) {
        languageSelector.addEventListener('change', (e) => {
            if (window.switchLanguage) {
                window.switchLanguage(e.target.value);
            }
        });
    }
    
    console.log('[initEventListeners] 事件监听器初始化完成');
}

// ============================================================================
// DOM 加载完成后初始化
// ============================================================================

// 等待 DOM 加载完成后再初始化
// 使用一个标志防止重复初始化
let eventListenersInitialized = false;

function initializeApp() {
    if (eventListenersInitialized) {
        console.warn('[App.js] 事件监听器已经初始化，跳过重复初始化');
        return;
    }
    
    console.log('[App.js] 准备初始化事件监听器，DOM状态:', document.readyState);
    initEventListeners();
    eventListenersInitialized = true;
    console.log('[App.js] 事件监听器初始化完成');
}

if (document.readyState === 'loading') {
    console.log('[App.js] DOM 正在加载，等待 DOMContentLoaded 事件');
    document.addEventListener('DOMContentLoaded', initializeApp);
} else {
    // DOM 已经加载完成
    console.log('[App.js] DOM 已加载完成，立即初始化');
    initializeApp();
}
