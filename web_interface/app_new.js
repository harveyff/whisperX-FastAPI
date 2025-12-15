// API 基础配置
const API_BASE_URL = 'http://localhost:8000';
let currentTaskId = null;
let currentFile = null;
let currentVideoUrl = null;
let currentSubtitleData = null;
let currentInterface = null;

// 多语言支持
function t(key) {
    return window.t ? window.t(key) : key;
}

// Tab 切换
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const tab = btn.dataset.tab;
        
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
        }
    });
});

// 折叠参数组
document.querySelectorAll('.collapsible-header').forEach(header => {
    header.addEventListener('click', () => {
        const group = header.closest('.param-group');
        const content = group.querySelector('.collapsible-content');
        const isExpanded = group.classList.contains('expanded');
        
        if (isExpanded) {
            group.classList.remove('expanded');
            content.style.display = 'none';
        } else {
            group.classList.add('expanded');
            content.style.display = 'block';
        }
    });
});

// URL输入切换
const useUrlInput = document.getElementById('useUrlInput');
const urlInput = document.getElementById('urlInput');
if (useUrlInput && urlInput) {
    useUrlInput.addEventListener('change', (e) => {
        urlInput.style.display = e.target.checked ? 'block' : 'none';
    });
}

// 接口选择器
document.querySelectorAll('.interface-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const interface = btn.dataset.interface;
        
        // 更新按钮状态
        document.querySelectorAll('.interface-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        
        // 显示接口表单
        currentInterface = interface;
        showInterfaceForm(interface);
    });
});

// 显示接口表单
function showInterfaceForm(interface) {
    const content = document.getElementById('interface-content');
    if (!content) return;
    
    const forms = {
        'transcribe': getTranscribeForm(),
        'align': getAlignForm(),
        'diarize': getDiarizeForm(),
        'combine': getCombineForm(),
        'full': getFullForm(),
        'url': getUrlForm()
    };
    
    content.innerHTML = forms[interface] || '<p>接口表单未实现</p>';
    
    // 重新绑定折叠事件
    content.querySelectorAll('.collapsible-header').forEach(header => {
        header.addEventListener('click', () => {
            const group = header.closest('.param-group');
            const content = group.querySelector('.collapsible-content');
            const isExpanded = group.classList.contains('expanded');
            
            if (isExpanded) {
                group.classList.remove('expanded');
                content.style.display = 'none';
            } else {
                group.classList.add('expanded');
                content.style.display = 'block';
            }
        });
    });
}

// 生成接口表单HTML
function getTranscribeForm() {
    return `
        <div class="interface-form">
            <h3>接口6: 转录音频文件</h3>
            <p class="section-desc">只执行转录步骤，不进行对齐和说话人分离</p>
            
            <div class="section">
                <h4>文件上传</h4>
                <div class="upload-area" id="sbs-uploadArea">
                    <input type="file" id="sbs-fileInput" accept="audio/*,video/*" style="display: none;">
                    <div class="upload-content">
                        <p>点击或拖拽文件到此处上传</p>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h4>参数配置</h4>
                ${getCommonParamsForm('sbs')}
                <button class="btn btn-primary" onclick="startSBSProcess('transcribe')">开始转录</button>
            </div>
            
            <div class="section" id="sbs-result" style="display: none;">
                <h4>结果</h4>
                <div id="sbs-result-content"></div>
            </div>
        </div>
    `;
}

function getAlignForm() {
    return `
        <div class="interface-form">
            <h3>接口7: 对齐转录文本</h3>
            <p class="section-desc">需要上传转录JSON文件和原始音频文件</p>
            
            <div class="section">
                <h4>文件上传</h4>
                <div class="config-item">
                    <label>转录JSON文件</label>
                    <input type="file" id="sbs-transcriptFile" accept=".json">
                </div>
                <div class="config-item">
                    <label>原始音频文件</label>
                    <input type="file" id="sbs-audioFile" accept="audio/*,video/*">
                </div>
            </div>
            
            <div class="section">
                <h4>参数配置</h4>
                ${getAlignParamsForm('sbs')}
                <button class="btn btn-primary" onclick="startSBSProcess('align')">开始对齐</button>
            </div>
            
            <div class="section" id="sbs-result" style="display: none;">
                <h4>结果</h4>
                <div id="sbs-result-content"></div>
            </div>
        </div>
    `;
}

function getDiarizeForm() {
    return `
        <div class="interface-form">
            <h3>接口8: 说话人分离</h3>
            <p class="section-desc">识别音频中的不同说话人</p>
            
            <div class="section">
                <h4>文件上传</h4>
                <div class="upload-area" id="sbs-uploadArea">
                    <input type="file" id="sbs-fileInput" accept="audio/*,video/*" style="display: none;">
                    <div class="upload-content">
                        <p>点击或拖拽文件到此处上传</p>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h4>参数配置</h4>
                ${getDiarizationParamsForm('sbs')}
                <button class="btn btn-primary" onclick="startSBSProcess('diarize')">开始分离</button>
            </div>
            
            <div class="section" id="sbs-result" style="display: none;">
                <h4>结果</h4>
                <div id="sbs-result-content"></div>
            </div>
        </div>
    `;
}

function getCombineForm() {
    return `
        <div class="interface-form">
            <h3>接口9: 合并转录和说话人分离结果</h3>
            <p class="section-desc">需要上传对齐后的转录JSON和说话人分离JSON</p>
            
            <div class="section">
                <h4>文件上传</h4>
                <div class="config-item">
                    <label>对齐转录JSON文件</label>
                    <input type="file" id="sbs-alignedFile" accept=".json">
                </div>
                <div class="config-item">
                    <label>说话人分离JSON文件</label>
                    <input type="file" id="sbs-diarizationFile" accept=".json">
                </div>
            </div>
            
            <div class="section">
                <button class="btn btn-primary" onclick="startSBSProcess('combine')">开始合并</button>
            </div>
            
            <div class="section" id="sbs-result" style="display: none;">
                <h4>结果</h4>
                <div id="sbs-result-content"></div>
            </div>
        </div>
    `;
}

function getFullForm() {
    return `
        <div class="interface-form">
            <h3>接口4: 完整流程处理</h3>
            <p class="section-desc">一次性完成转录、对齐、说话人分离</p>
            
            <div class="section">
                <h4>文件上传</h4>
                <div class="upload-area" id="sbs-uploadArea">
                    <input type="file" id="sbs-fileInput" accept="audio/*,video/*" style="display: none;">
                    <div class="upload-content">
                        <p>点击或拖拽文件到此处上传</p>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h4>参数配置</h4>
                ${getCommonParamsForm('sbs')}
                ${getAlignParamsForm('sbs')}
                ${getDiarizationParamsForm('sbs')}
                <button class="btn btn-primary" onclick="startSBSProcess('full')">开始处理</button>
            </div>
            
            <div class="section" id="sbs-result" style="display: none;">
                <h4>结果</h4>
                <div id="sbs-result-content"></div>
            </div>
        </div>
    `;
}

function getUrlForm() {
    return `
        <div class="interface-form">
            <h3>接口5: 通过URL处理音频文件</h3>
            <p class="section-desc">通过URL地址处理音频文件</p>
            
            <div class="section">
                <h4>URL输入</h4>
                <div class="config-item">
                    <label>音频文件URL</label>
                    <input type="url" id="sbs-urlInput" placeholder="https://example.com/audio.mp3" style="width: 100%;">
                </div>
            </div>
            
            <div class="section">
                <h4>参数配置</h4>
                ${getCommonParamsForm('sbs')}
                ${getAlignParamsForm('sbs')}
                ${getDiarizationParamsForm('sbs')}
                <button class="btn btn-primary" onclick="startSBSProcess('url')">开始处理</button>
            </div>
            
            <div class="section" id="sbs-result" style="display: none;">
                <h4>结果</h4>
                <div id="sbs-result-content"></div>
            </div>
        </div>
    `;
}

// 生成通用参数表单
function getCommonParamsForm(prefix) {
    return `
        <div class="param-group">
            <h4 class="param-group-title">主要参数</h4>
            <div class="config-grid">
                <div class="config-item">
                    <label>语言</label>
                    <select id="${prefix}-language">
                        <option value="zh">中文</option>
                        <option value="en">English</option>
                        <option value="ja">日本語</option>
                    </select>
                </div>
                <div class="config-item">
                    <label>模型</label>
                    <select id="${prefix}-model">
                        <option value="base" selected>Base</option>
                        <option value="small">Small</option>
                        <option value="medium">Medium</option>
                    </select>
                </div>
                <div class="config-item">
                    <label>设备</label>
                    <select id="${prefix}-device">
                        <option value="cuda">GPU (CUDA)</option>
                        <option value="cpu">CPU</option>
                    </select>
                </div>
            </div>
        </div>
        
        <div class="param-group collapsible">
            <h4 class="param-group-title collapsible-header">
                <span>Whisper模型参数</span>
                <span class="toggle-icon">▼</span>
            </h4>
            <div class="collapsible-content" style="display: none;">
                <div class="config-grid">
                    <div class="config-item">
                        <label>任务类型</label>
                        <select id="${prefix}-task">
                            <option value="transcribe" selected>转录</option>
                            <option value="translate">翻译</option>
                        </select>
                    </div>
                    <div class="config-item">
                        <label>批次大小</label>
                        <input type="number" id="${prefix}-batch_size" value="8">
                    </div>
                    <div class="config-item">
                        <label>块大小</label>
                        <input type="number" id="${prefix}-chunk_size" value="20">
                    </div>
                </div>
            </div>
        </div>
        
        <div class="param-group collapsible">
            <h4 class="param-group-title collapsible-header">
                <span>ASR选项</span>
                <span class="toggle-icon">▼</span>
            </h4>
            <div class="collapsible-content" style="display: none;">
                <div class="config-grid">
                    <div class="config-item">
                        <label>束搜索大小</label>
                        <input type="number" id="${prefix}-beam_size" value="5">
                    </div>
                    <div class="config-item">
                        <label>采样温度</label>
                        <input type="number" id="${prefix}-temperatures" value="0.0" step="0.1">
                    </div>
                </div>
            </div>
        </div>
        
        <div class="param-group collapsible">
            <h4 class="param-group-title collapsible-header">
                <span>VAD选项</span>
                <span class="toggle-icon">▼</span>
            </h4>
            <div class="collapsible-content" style="display: none;">
                <div class="config-grid">
                    <div class="config-item">
                        <label>VAD起始阈值</label>
                        <input type="number" id="${prefix}-vad_onset" value="0.500" step="0.001">
                    </div>
                    <div class="config-item">
                        <label>VAD偏移阈值</label>
                        <input type="number" id="${prefix}-vad_offset" value="0.363" step="0.001">
                    </div>
                </div>
            </div>
        </div>
    `;
}

function getAlignParamsForm(prefix) {
    return `
        <div class="param-group collapsible">
            <h4 class="param-group-title collapsible-header">
                <span>对齐参数</span>
                <span class="toggle-icon">▼</span>
            </h4>
            <div class="collapsible-content" style="display: none;">
                <div class="config-grid">
                    <div class="config-item">
                        <label>对齐模型</label>
                        <input type="text" id="${prefix}-align_model" placeholder="留空使用默认">
                    </div>
                    <div class="config-item">
                        <label>插值方法</label>
                        <select id="${prefix}-interpolate_method">
                            <option value="nearest" selected>nearest</option>
                            <option value="linear">linear</option>
                            <option value="ignore">ignore</option>
                        </select>
                    </div>
                </div>
            </div>
        </div>
    `;
}

function getDiarizationParamsForm(prefix) {
    return `
        <div class="param-group collapsible">
            <h4 class="param-group-title collapsible-header">
                <span>说话人分离参数</span>
                <span class="toggle-icon">▼</span>
            </h4>
            <div class="collapsible-content" style="display: none;">
                <div class="config-grid">
                    <div class="config-item">
                        <label>最小说话人数</label>
                        <input type="number" id="${prefix}-min_speakers" placeholder="留空自动检测">
                    </div>
                    <div class="config-item">
                        <label>最大说话人数</label>
                        <input type="number" id="${prefix}-max_speakers" placeholder="留空自动检测">
                    </div>
                </div>
            </div>
        </div>
    `;
}

// 分步调用处理
window.startSBSProcess = async function(interface) {
    console.log('Starting SBS process:', interface);
    // TODO: 实现分步调用逻辑
    alert('分步调用功能开发中...');
};

// 一步生成处理（保留原有逻辑）
// ... 原有的一步生成代码 ...

