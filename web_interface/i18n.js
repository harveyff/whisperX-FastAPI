// 多语言资源
const i18n = {
    'zh-CN': {
        title: '🎙️ WhisperX 语音转文字工具',
        subtitle: '上传视频/音频文件，自动生成字幕并播放',
        uploadHint: '点击或拖拽文件到此处上传',
        fileHint: '支持音频和视频文件 (MP3, WAV, MP4, AVI等)',
        configTitle: '处理配置',
        language: '语言',
        model: '模型',
        device: '设备',
        enableDiarization: '启用说话人分离',
        chineseConversion: '中文转换',
        conversionNone: '不转换',
        conversionT2S: '繁体转简体',
        conversionS2T: '简体转繁体',
        conversionS2TW: '简体转台湾繁体',
        conversionTW2S: '台湾繁体转简体',
        startProcess: '开始处理',
        progressTitle: '处理进度',
        uploading: '正在上传文件...',
        processing: '正在处理中，请稍候...',
        completed: '处理完成！',
        resultTitle: '处理结果',
        downloadSrt: '下载 SRT 字幕',
        downloadVtt: '下载 VTT 字幕',
        downloadJson: '下载 JSON 结果',
        transcriptTitle: '转录文本',
        speaker: '说话人',
        taskId: '任务ID',
        status: '状态',
        taskType: '任务类型',
        duration: '处理时长',
        seconds: '秒',
        fileSelected: '✅ 已选择文件',
        fileSize: '文件大小',
        selectFile: '请先选择文件',
        processingFailed: '处理失败',
        queryFailed: '查询任务状态失败',
        tabOneStep: '一步生成字幕',
        tabStepByStep: '分步调用接口',
        oneStepTitle: '完整工作流 - 一步生成字幕',
        oneStepDesc: '上传文件后自动完成：转录 → 对齐 → 说话人分离 → 生成字幕文件',
        stepByStepTitle: '分步调用接口',
        stepByStepDesc: '按工作流程分步调用各个接口，可以手动编辑中间结果',
        fileUpload: '文件上传',
        useUrl: '或使用URL地址',
        useYouTube: '使用YouTube URL',
        uploadFile: '上传文件',
        mainParams: '主要参数',
        whisperParams: 'Whisper模型参数',
        alignParams: '对齐参数',
        diarizationParams: '说话人分离参数',
        asrParams: 'ASR选项',
        vadParams: 'VAD选项',
        selectInterface: '选择接口',
        interfaceTranscribe: '接口6: 转录',
        interfaceAlign: '接口7: 对齐',
        interfaceDiarize: '接口8: 说话人分离',
        interfaceCombine: '接口9: 合并',
        interfaceFull: '接口4: 完整流程',
        interfaceUrl: '接口5: URL处理',
        enterUrl: '请输入URL地址',
        enterYouTubeUrl: '请输入YouTube URL',
        urlPlaceholder: 'https://example.com/audio.mp3',
        youtubePlaceholder: 'https://www.youtube.com/watch?v=...',
        videoNotSupported: '您的浏览器不支持视频播放',
        subtitle: '字幕',
        taskType: '任务类型',
        taskTranscribe: '转录 (transcribe)',
        taskTranslate: '翻译 (translate)',
        deviceIndex: '设备索引',
        cpuThreads: 'CPU线程数',
        batchSize: '批次大小',
        chunkSize: '块大小',
        computeType: '计算类型',
        alignModel: '对齐模型',
        alignModelPlaceholder: '留空使用默认',
        interpolateMethod: '插值方法',
        returnCharAlignments: '返回字符级对齐',
        minSpeakers: '最小说话人数',
        maxSpeakers: '最大说话人数',
        autoDetectPlaceholder: '留空自动检测',
        beamSize: '束搜索大小',
        bestOf: '保留束数量',
        patience: '耐心值',
        lengthPenalty: '长度惩罚',
        temperatures: '采样温度',
        compressionRatioThreshold: '压缩比阈值',
        logProbThreshold: '对数概率阈值',
        noSpeechThreshold: '无语音阈值',
        initialPrompt: '初始提示',
        initialPromptPlaceholder: '可选，用于提示模型',
        suppressTokens: '抑制标记',
        suppressTokensPlaceholder: '逗号分隔，如: -1,2,3',
        suppressNumerals: '抑制数字符号',
        hotwords: '热词提示',
        hotwordsPlaceholder: '可选，用于提示特定词汇',
        vadOnset: 'VAD起始阈值',
        vadOffset: 'VAD偏移阈值',
        openccNotLoaded: 'OpenCC 库未加载，使用简单转换',
        conversionFailed: '转换失败',
        modelTiny: 'Tiny (最快, 75MB)',
        modelBase: 'Base (推荐, 142MB)',
        modelSmall: 'Small (较好, 466MB)',
        modelMedium: 'Medium (好, 1.4GB)',
        modelLargeV2: 'Large-v2 (很好, 2.9GB)',
        modelLargeV3: 'Large-v3 (最好, 3.1GB)',
        modelLargeV3Turbo: 'Large-v3-turbo (优化版, 3.1GB)',
        deviceGpu: 'GPU (CUDA)',
        deviceCpu: 'CPU'
    },
    'en': {
        title: '🎙️ WhisperX Speech-to-Text Tool',
        subtitle: 'Upload video/audio files, automatically generate subtitles and play',
        uploadHint: 'Click or drag files here to upload',
        fileHint: 'Supports audio and video files (MP3, WAV, MP4, AVI, etc.)',
        configTitle: 'Processing Configuration',
        language: 'Language',
        model: 'Model',
        device: 'Device',
        enableDiarization: 'Enable Speaker Diarization',
        chineseConversion: 'Chinese Conversion',
        conversionNone: 'No Conversion',
        conversionT2S: 'Traditional to Simplified',
        conversionS2T: 'Simplified to Traditional',
        conversionS2TW: 'Simplified to Taiwan Traditional',
        conversionTW2S: 'Taiwan Traditional to Simplified',
        startProcess: 'Start Processing',
        progressTitle: 'Processing Progress',
        uploading: 'Uploading file...',
        processing: 'Processing, please wait...',
        completed: 'Processing completed!',
        resultTitle: 'Processing Result',
        downloadSrt: 'Download SRT Subtitle',
        downloadVtt: 'Download VTT Subtitle',
        downloadJson: 'Download JSON Result',
        transcriptTitle: 'Transcript',
        speaker: 'Speaker',
        taskId: 'Task ID',
        status: 'Status',
        taskType: 'Task Type',
        duration: 'Processing Duration',
        seconds: 'seconds',
        fileSelected: '✅ File Selected',
        fileSize: 'File Size',
        selectFile: 'Please select a file first',
        processingFailed: 'Processing failed',
        queryFailed: 'Failed to query task status',
        tabOneStep: 'One-Step Generation',
        tabStepByStep: 'Step-by-Step API',
        oneStepTitle: 'Complete Workflow - One-Step Subtitle Generation',
        oneStepDesc: 'Automatically complete: transcription → alignment → diarization → subtitle generation',
        stepByStepTitle: 'Step-by-Step API Calls',
        stepByStepDesc: 'Call each API step by step, can manually edit intermediate results',
        fileUpload: 'File Upload',
        useUrl: 'Or use URL address',
        useYouTube: 'Use YouTube URL',
        uploadFile: 'Upload File',
        mainParams: 'Main Parameters',
        whisperParams: 'Whisper Model Parameters',
        alignParams: 'Alignment Parameters',
        diarizationParams: 'Diarization Parameters',
        asrParams: 'ASR Options',
        vadParams: 'VAD Options',
        selectInterface: 'Select Interface',
        interfaceTranscribe: 'API 6: Transcribe',
        interfaceAlign: 'API 7: Align',
        interfaceDiarize: 'API 8: Diarize',
        interfaceCombine: 'API 9: Combine',
        interfaceFull: 'API 4: Full Process',
        interfaceUrl: 'API 5: URL Process',
        enterUrl: 'Please enter URL address',
        enterYouTubeUrl: 'Please enter YouTube URL',
        urlPlaceholder: 'https://example.com/audio.mp3',
        youtubePlaceholder: 'https://www.youtube.com/watch?v=...',
        videoNotSupported: 'Your browser does not support video playback',
        subtitle: 'Subtitle',
        taskType: 'Task Type',
        taskTranscribe: 'Transcribe',
        taskTranslate: 'Translate',
        deviceIndex: 'Device Index',
        cpuThreads: 'CPU Threads',
        batchSize: 'Batch Size',
        chunkSize: 'Chunk Size',
        computeType: 'Compute Type',
        alignModel: 'Alignment Model',
        alignModelPlaceholder: 'Leave empty to use default',
        interpolateMethod: 'Interpolate Method',
        returnCharAlignments: 'Return Character Alignments',
        minSpeakers: 'Min Speakers',
        maxSpeakers: 'Max Speakers',
        autoDetectPlaceholder: 'Leave empty for auto detection',
        beamSize: 'Beam Size',
        bestOf: 'Best Of',
        patience: 'Patience',
        lengthPenalty: 'Length Penalty',
        temperatures: 'Temperatures',
        compressionRatioThreshold: 'Compression Ratio Threshold',
        logProbThreshold: 'Log Probability Threshold',
        noSpeechThreshold: 'No Speech Threshold',
        initialPrompt: 'Initial Prompt',
        initialPromptPlaceholder: 'Optional, for prompting the model',
        suppressTokens: 'Suppress Tokens',
        suppressTokensPlaceholder: 'Comma separated, e.g.: -1,2,3',
        suppressNumerals: 'Suppress Numerals',
        hotwords: 'Hotwords',
        hotwordsPlaceholder: 'Optional, for prompting specific words',
        vadOnset: 'VAD Onset Threshold',
        vadOffset: 'VAD Offset Threshold',
        openccNotLoaded: 'OpenCC library not loaded, using simple conversion',
        conversionFailed: 'Conversion failed',
        modelTiny: 'Tiny (Fastest, 75MB)',
        modelBase: 'Base (Recommended, 142MB)',
        modelSmall: 'Small (Good, 466MB)',
        modelMedium: 'Medium (Better, 1.4GB)',
        modelLargeV2: 'Large-v2 (Very Good, 2.9GB)',
        modelLargeV3: 'Large-v3 (Best, 3.1GB)',
        modelLargeV3Turbo: 'Large-v3-turbo (Optimized, 3.1GB)',
        deviceGpu: 'GPU (CUDA)',
        deviceCpu: 'CPU'
    },
    'ja': {
        title: '🎙️ WhisperX 音声文字変換ツール',
        subtitle: '動画/音声ファイルをアップロードし、自動的に字幕を生成して再生',
        uploadHint: 'クリックまたはファイルをドラッグしてアップロード',
        fileHint: '音声・動画ファイルをサポート (MP3, WAV, MP4, AVI等)',
        configTitle: '処理設定',
        language: '言語',
        model: 'モデル',
        device: 'デバイス',
        enableDiarization: '話者分離を有効化',
        chineseConversion: '中国語変換',
        conversionNone: '変換なし',
        conversionT2S: '繁体字から簡体字',
        conversionS2T: '簡体字から繁体字',
        conversionS2TW: '簡体字から台湾繁体字',
        conversionTW2S: '台湾繁体字から簡体字',
        startProcess: '処理開始',
        progressTitle: '処理進捗',
        uploading: 'ファイルをアップロード中...',
        processing: '処理中、お待ちください...',
        completed: '処理完了！',
        resultTitle: '処理結果',
        downloadSrt: 'SRT字幕をダウンロード',
        downloadVtt: 'VTT字幕をダウンロード',
        downloadJson: 'JSON結果をダウンロード',
        transcriptTitle: '転写テキスト',
        speaker: '話者',
        taskId: 'タスクID',
        status: 'ステータス',
        taskType: 'タスクタイプ',
        duration: '処理時間',
        seconds: '秒',
        fileSelected: '✅ ファイルを選択しました',
        fileSize: 'ファイルサイズ',
        selectFile: 'まずファイルを選択してください',
        processingFailed: '処理に失敗しました',
        queryFailed: 'タスクステータスの照会に失敗しました',
        tabOneStep: 'ワンステップ生成',
        tabStepByStep: 'ステップバイステップAPI',
        oneStepTitle: '完全ワークフロー - ワンステップ字幕生成',
        oneStepDesc: '自動的に完了：転写 → 整列 → 話者分離 → 字幕生成',
        stepByStepTitle: 'ステップバイステップAPI呼び出し',
        stepByStepDesc: '各APIを段階的に呼び出し、中間結果を手動で編集可能',
        fileUpload: 'ファイルアップロード',
        useUrl: 'またはURLアドレスを使用',
        useYouTube: 'YouTube URLを使用',
        uploadFile: 'ファイルをアップロード',
        mainParams: '主要パラメータ',
        whisperParams: 'Whisperモデルパラメータ',
        alignParams: '整列パラメータ',
        diarizationParams: '話者分離パラメータ',
        asrParams: 'ASRオプション',
        vadParams: 'VADオプション',
        selectInterface: 'インターフェース選択',
        interfaceTranscribe: 'API 6: 転写',
        interfaceAlign: 'API 7: 整列',
        interfaceDiarize: 'API 8: 話者分離',
        interfaceCombine: 'API 9: 結合',
        interfaceFull: 'API 4: 完全処理',
        interfaceUrl: 'API 5: URL処理',
        enterUrl: 'URLアドレスを入力してください',
        enterYouTubeUrl: 'YouTube URLを入力してください',
        urlPlaceholder: 'https://example.com/audio.mp3',
        youtubePlaceholder: 'https://www.youtube.com/watch?v=...',
        videoNotSupported: 'お使いのブラウザは動画再生をサポートしていません',
        subtitle: '字幕',
        taskType: 'タスクタイプ',
        taskTranscribe: '転写 (transcribe)',
        taskTranslate: '翻訳 (translate)',
        deviceIndex: 'デバイスインデックス',
        cpuThreads: 'CPUスレッド数',
        batchSize: 'バッチサイズ',
        chunkSize: 'チャンクサイズ',
        computeType: '計算タイプ',
        alignModel: '整列モデル',
        alignModelPlaceholder: '空欄の場合はデフォルトを使用',
        interpolateMethod: '補間方法',
        returnCharAlignments: '文字レベル整列を返す',
        minSpeakers: '最小話者数',
        maxSpeakers: '最大話者数',
        autoDetectPlaceholder: '空欄の場合は自動検出',
        beamSize: 'ビーム検索サイズ',
        bestOf: '保持ビーム数',
        patience: '忍耐値',
        lengthPenalty: '長さペナルティ',
        temperatures: 'サンプリング温度',
        compressionRatioThreshold: '圧縮比しきい値',
        logProbThreshold: '対数確率しきい値',
        noSpeechThreshold: '無音しきい値',
        initialPrompt: '初期プロンプト',
        initialPromptPlaceholder: 'オプション、モデルへのプロンプト用',
        suppressTokens: '抑制トークン',
        suppressTokensPlaceholder: 'カンマ区切り、例: -1,2,3',
        suppressNumerals: '数字記号を抑制',
        hotwords: 'ホットワード',
        hotwordsPlaceholder: 'オプション、特定の語彙へのプロンプト用',
        vadOnset: 'VAD開始しきい値',
        vadOffset: 'VADオフセットしきい値',
        openccNotLoaded: 'OpenCCライブラリが読み込まれていません。簡単な変換を使用します',
        conversionFailed: '変換に失敗しました',
        modelTiny: 'Tiny (最速, 75MB)',
        modelBase: 'Base (推奨, 142MB)',
        modelSmall: 'Small (良好, 466MB)',
        modelMedium: 'Medium (良い, 1.4GB)',
        modelLargeV2: 'Large-v2 (非常に良い, 2.9GB)',
        modelLargeV3: 'Large-v3 (最高, 3.1GB)',
        modelLargeV3Turbo: 'Large-v3-turbo (最適化版, 3.1GB)',
        deviceGpu: 'GPU (CUDA)',
        deviceCpu: 'CPU'
    }
};

// 当前语言
let currentLang = 'zh-CN';

// 检测浏览器语言
function detectBrowserLanguage() {
    const browserLang = navigator.language || navigator.userLanguage;
    
    // 支持的语言列表
    const supportedLangs = ['zh-CN', 'en', 'ja'];
    
    // 精确匹配
    if (supportedLangs.includes(browserLang)) {
        return browserLang;
    }
    
    // 语言代码匹配（如 zh-TW -> zh-CN）
    const langCode = browserLang.split('-')[0];
    if (langCode === 'zh') {
        return 'zh-CN';
    }
    
    // 默认返回英文
    return 'en';
}

// 初始化语言
function initLanguage() {
    // 从本地存储读取
    const savedLang = localStorage.getItem('preferredLanguage');
    if (savedLang && i18n[savedLang]) {
        currentLang = savedLang;
    } else {
        // 使用浏览器语言
        currentLang = detectBrowserLanguage();
    }
    
    applyLanguage();
}

// 应用语言
function applyLanguage() {
    const lang = i18n[currentLang];
    if (!lang) return;
    
    // 更新所有文本
    const h1 = document.querySelector('h1');
    const subtitle = document.querySelector('.subtitle');
    if (h1) h1.textContent = lang.title;
    if (subtitle) subtitle.textContent = lang.subtitle;
    
    // 更新所有带 data-i18n 属性的元素
    document.querySelectorAll('[data-i18n]').forEach(el => {
        const key = el.getAttribute('data-i18n');
        if (lang[key]) {
            if (el.tagName === 'OPTION') {
                el.textContent = lang[key];
            } else if (el.tagName === 'SPAN' && el.parentElement.tagName === 'LABEL') {
                // label内的span，只更新文本
                el.textContent = lang[key];
            } else {
                el.textContent = lang[key];
            }
        }
    });
    
    // 更新上传区域（仅在未选择文件时）
    const uploadContent = document.querySelector('.upload-content');
    if (uploadContent && !uploadContent.textContent.includes('✅') && !uploadContent.textContent.includes('File Selected')) {
        const svg = uploadContent.querySelector('svg');
        const existingP = uploadContent.querySelectorAll('p');
        if (existingP.length >= 2) {
            // 更新现有元素
            if (existingP[0].hasAttribute('data-i18n')) {
                existingP[0].textContent = lang.uploadHint;
            }
            if (existingP[1].hasAttribute('data-i18n')) {
                existingP[1].textContent = lang.fileHint;
            }
        }
    }
    
    // 更新语言选择器
    updateLanguageSelector();
    
    // 更新HTML lang属性
    document.documentElement.lang = currentLang;
    
    // 触发自定义事件，通知其他脚本语言已切换
    window.dispatchEvent(new CustomEvent('languageChanged', { detail: currentLang }));
}

// 更新语言选择器
function updateLanguageSelector() {
    const selector = document.getElementById('languageSelector');
    if (selector) {
        selector.value = currentLang;
    }
}

// 切换语言
function switchLanguage(lang) {
    if (!i18n[lang]) return;
    
    currentLang = lang;
    localStorage.setItem('preferredLanguage', lang);
    applyLanguage();
}

// 导出到全局，供其他脚本使用
window.switchLanguage = switchLanguage;
window.t = t;
window.currentLang = () => currentLang;

// 获取翻译文本
function t(key) {
    return i18n[currentLang]?.[key] || key;
}

// 页面加载时初始化
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initLanguage);
} else {
    initLanguage();
}

