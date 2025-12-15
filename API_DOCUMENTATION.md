# WhisperX FastAPI 接口文档

本文档详细说明了 WhisperX FastAPI 项目的所有 API 接口，包括接口含义、请求参数和返回示例。

## 目录

1. [健康检查接口](#健康检查接口)
2. [语音转文字接口](#语音转文字接口)
3. [语音转文字服务接口](#语音转文字服务接口)
4. [任务管理接口](#任务管理接口)

## 接口对比说明

为了避免混淆，以下是相关接口的区别说明：

### 完整处理 vs 分步处理

- **接口4 (`/speech-to-text`)** 和 **接口5 (`/speech-to-text-url`)**：完整处理流程
  - 一次性完成：转录 → 对齐 → 说话人分离 → 合并
  - 适合需要完整结果的场景
  
- **接口6-9 (`/service/*`)**：分步处理
  - 可以单独执行：转录、对齐、说话人分离、合并
  - 适合需要分步控制或只执行部分步骤的场景

### URL 处理接口

- **接口5 (`/speech-to-text-url`)**：通过URL处理音频文件
  - 适用于直接链接到音频文件的 URL（如 `https://example.com/audio.mp3`）
  - 系统会自动下载文件并处理

---

## 健康检查接口

### 1. 简单健康检查

**接口路径:** `GET /health`

**接口说明:** 检查服务是否正常运行。就像问"你还好吗？"，返回"我很好"。

**请求参数:** 无

**请求示例:**
```bash
curl -X GET "http://localhost:8000/health"
```

**返回示例:**
```json
{
  "status": "ok",
  "message": "Service is running"
}
```

---

### 2. 存活检查

**接口路径:** `GET /health/live`

**接口说明:** 检查服务是否还活着。会返回当前时间，用于监控系统判断服务是否正常运行。

**请求参数:** 无

**请求示例:**
```bash
curl -X GET "http://localhost:8000/health/live"
```

**返回示例:**
```json
{
  "status": "ok",
  "timestamp": 1703123456.789,
  "message": "Application is live"
}
```

---

### 3. 就绪检查

**接口路径:** `GET /health/ready`

**接口说明:** 检查服务是否准备好处理任务。会检查数据库等依赖是否正常，如果都正常就返回成功，如果有问题就返回失败。

**请求参数:** 无

**请求示例:**
```bash
curl -X GET "http://localhost:8000/health/ready"
```

**返回示例（成功）:**
```json
{
  "status": "ok",
  "database": "connected",
  "message": "Application is ready to accept requests"
}
```

**返回示例（失败）:**
```json
{
  "status": "error",
  "database": "disconnected",
  "message": "Application is not ready due to an internal error."
}
```

---

## 语音转文字接口

### 4. 上传音频文件进行语音转文字

**接口路径:** `POST /speech-to-text`

**接口说明:** 上传一个音频或视频文件，系统会自动完成所有处理：把语音转成文字、给每个字加上时间戳、识别不同说话人。处理完成后可以通过任务ID查询结果。

**与接口6的区别:** 此接口执行完整处理流程（转录+对齐+说话人分离+合并），而接口6只执行转录步骤。如果需要完整结果，使用此接口；如果需要分步控制，使用接口6-9。

**请求参数:**

- `file` (multipart/form-data, 必需): 要处理的音频/视频文件
- `language` (query, 可选, 默认: "en"): 转录语言代码
- `task` (query, 可选, 默认: "transcribe"): 任务类型，可选值: "transcribe" 或 "translate"
- `model` (query, 可选, 默认: "tiny"): Whisper 模型名称
- `device` (query, 可选, 默认: "cuda"): 设备类型，可选值: "cuda" 或 "cpu"
- `device_index` (query, 可选, 默认: 0): 设备索引
- `threads` (query, 可选, 默认: 0): CPU 推理使用的线程数
- `batch_size` (query, 可选, 默认: 8): 推理的首选批次大小
- `chunk_size` (query, 可选, 默认: 20): 合并 VAD 段的块大小
- `compute_type` (query, 可选, 默认: "float16"): 计算类型，可选值: "float16", "float32", "int8"
- `align_model` (query, 可选): 用于对齐的音素级 ASR 模型名称
- `interpolate_method` (query, 可选, 默认: "nearest"): 插值方法，可选值: "nearest", "linear", "ignore"
- `return_char_alignments` (query, 可选, 默认: false): 是否返回字符级对齐
- `min_speakers` (query, 可选): 音频文件中的最小说话人数量
- `max_speakers` (query, 可选): 音频文件中的最大说话人数量
- `beam_size` (query, 可选, 默认: 5): 束搜索中的束数量
- `best_of` (query, 可选, 默认: 5): 束搜索中保留的束数量
- `patience` (query, 可选, 默认: 1.0): 束解码中的耐心值
- `length_penalty` (query, 可选, 默认: 1.0): 标记长度惩罚系数
- `temperatures` (query, 可选, 默认: 0.0): 采样温度
- `compression_ratio_threshold` (query, 可选, 默认: 2.4): gzip 压缩比阈值
- `log_prob_threshold` (query, 可选, 默认: -1.0): 平均对数概率阈值
- `no_speech_threshold` (query, 可选, 默认: 0.6): 无语音阈值
- `initial_prompt` (query, 可选): 初始提示文本
- `suppress_tokens` (query, 可选, 默认: "-1"): 要抑制的标记 ID 列表（逗号分隔）
- `suppress_numerals` (query, 可选, 默认: false): 是否抑制数字符号
- `hotwords` (query, 可选): 热词提示
- `vad_onset` (query, 可选, 默认: 0.500): VAD 起始阈值
- `vad_offset` (query, 可选, 默认: 0.363): VAD 偏移阈值

**请求示例:**
```bash
curl -X POST "http://localhost:8000/speech-to-text?language=en&model=base&device=cuda" \
  -F "file=@audio.mp3"
```

**返回示例:**
```json
{
  "identifier": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Task queued"
}
```

---

### 5. 通过 URL 处理音频文件

**接口路径:** `POST /speech-to-text-url`

**接口说明:** 通过URL地址处理音频文件。你只需要提供一个音频文件的下载链接，系统会自动下载并处理。适合处理网络上的音频文件，不需要先下载到本地。

**请求参数:**

- `url` (form-data, 必需): 音频文件的 URL
- 其他参数与 `/speech-to-text` 接口相同

**请求示例:**
```bash
curl -X POST "http://localhost:8000/speech-to-text-url?language=en&model=base" \
  -F "url=https://example.com/audio.mp3"
```

**返回示例:**
```json
{
  "identifier": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Task queued"
}
```

---

## 语音转文字服务接口

### 6. 转录音频文件

**接口路径:** `POST /service/transcribe`

**接口说明:** 把音频文件里的语音转成文字。这是第一步，只做转录，不做时间对齐和说话人识别。适合只需要文字内容的场景，或者想分步处理的场景。

**与接口4的区别:** 接口4会一次性完成所有处理（转录+对齐+说话人分离），这个接口只做转录。如果你需要完整结果，用接口4；如果想分步控制，用这个接口配合后面的接口。

**请求参数:**

- `file` (multipart/form-data, 必需): 要转录的音频/视频文件
- `language` (query, 可选, 默认: "en"): 转录语言代码
- `task` (query, 可选, 默认: "transcribe"): 任务类型
- `model` (query, 可选, 默认: "tiny"): Whisper 模型名称
- `device` (query, 可选, 默认: "cuda"): 设备类型
- 其他 ASR 和 VAD 相关参数（与 `/speech-to-text` 相同）

**请求示例:**
```bash
curl -X POST "http://localhost:8000/service/transcribe?language=en&model=base" \
  -F "file=@audio.mp3"
```

**返回示例:**
```json
{
  "identifier": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Task queued"
}
```

**任务完成后的结果示例（通过 GET /task/{identifier} 获取）:**
```json
{
  "status": "completed",
  "result": {
    "language": "en",
    "segments": [
      {
        "start": 2.858,
        "end": 10.299,
        "text": " This is a test audio file of about phone line quality in English."
      }
    ]
  },
  "metadata": {
    "task_type": "transcription",
    "task_params": {...},
    "language": "en",
    "file_name": "audio.mp3",
    "audio_duration": 10.299
  },
  "error": null
}
```

---

### 7. 对齐转录文本

**接口路径:** `POST /service/align`

**接口说明:** 给转录出来的文字加上精确的时间戳。比如"苹果"这个词在音频的第0.5秒到第1秒。这样就能知道每个字是什么时候说的，可以用来做字幕。

**具体作用:**
- 给每个字或词标上开始时间和结束时间
- 告诉你每个字在音频中的准确位置
- 可以用来生成字幕文件（SRT格式）
- **注意**: 需要同时上传转录结果和原始音频文件，因为要对音频进行分析

**使用场景:**
1. **制作字幕文件**: 需要精确的单词级时间戳来生成 SRT/VTT 字幕
2. **音频标注**: 需要知道每个单词在音频中的精确位置
3. **分步处理流程**: 先转录（接口6）→ 再对齐（接口7）→ 再说话人分离（接口8）→ 最后合并（接口9）
4. **质量控制**: 需要检查转录的准确性，单词级对齐可以帮助识别问题
5. **手动编辑后重新对齐**: 拿到接口6的转录结果后，可以手动编辑转录文本（修正错误），然后使用接口7重新对齐

**输入要求:**
- 必须提供转录 JSON 文件（可以是接口6的结果，也可以是手动编辑后的转录文件）
- 必须提供原始的音频/视频文件（用于对齐分析）

**手动编辑流程:**
1. 使用接口6进行转录，获取 `transcript.json`
2. 手动编辑 `transcript.json`，修正转录错误或调整文本
3. 使用接口7，上传编辑后的 `transcript.json` 和原始音频文件
4. 接口7会基于编辑后的文本重新进行对齐，生成新的单词级时间戳

**实际操作示例:**

假设接口6返回的结果如下：
```json
{
  "status": "completed",
  "result": {
    "segments": [
      {
        "text": "苹果十年都干不成凭啥你们三年能干成最少需要100亿美元如果大家觉得我合适为了小米我愿意挺身而出",
        "start": 0.031,
        "end": 11.978
      }
    ],
    "language": "zh"
  }
}
```

**步骤1: 提取并格式化转录JSON**

⚠️ **重要**: 接口7需要的JSON格式与接口6的返回格式不同！

从接口6的结果中，**只提取 `result` 部分**，创建 `transcript.json` 文件。

**接口6返回的完整格式（不要直接上传这个）:**
```json
{
  "status": "completed",
  "result": {
    "segments": [...],
    "language": "zh"
  },
  "metadata": {...}
}
```

**接口7需要的格式（只提取result部分）:**
```json
{
  "language": "zh",
  "segments": [
    {
      "start": 0.031,
      "end": 11.978,
      "text": "苹果十年都干不成凭啥你们三年能干成最少需要100亿美元如果大家觉得我合适为了小米我愿意挺身而出"
    }
  ]
}
```

**正确的提取方法:**

从接口6的返回结果中，提取 `result` 对象的内容：
- `result.language` → `language`
- `result.segments` → `segments`（每个segment只需要 `start`, `end`, `text` 三个字段）

**错误示例（会导致报错）:**
```json
{
  "status": "completed",  // ❌ 不要包含这些字段
  "result": {...},       // ❌ 不要嵌套result
  "metadata": {...}      // ❌ 不要包含metadata
}
```

**错误示例2（会导致 JSONDecodeError）:**
```json
{
  "segments": [...],
  "language": "zh"
},                      // ❌ 不要有多余的逗号
"metadata": {...}       // ❌ 不要包含metadata字段
```

**正确示例（接口7需要的格式）:**
```json
{
  "language": "zh",
  "segments": [
    {
      "start": 0.031,
      "end": 11.978,
      "text": "苹果十年都干不成，凭啥你们三年能干成？最少需要100亿美元。如果大家觉得我合适，为了小米，我愿意挺身而出"
    }
  ]
}
```

**关键点:**
- ✅ 顶层只有两个字段：`language` 和 `segments`
- ✅ 不要包含 `metadata`、`status`、`result` 等任何其他字段
- ✅ JSON 文件必须以 `{` 开始，以 `}` 结束，中间不要有多余的逗号
- ✅ `segments` 数组中每个对象只有三个字段：`start`、`end`、`text`

**步骤2: 手动编辑转录文本**

可以添加标点符号、修正错误、调整文本等。例如：
```json
{
  "language": "zh",
  "segments": [
    {
      "start": 0.031,
      "end": 11.978,
      "text": "苹果十年都干不成，凭啥你们三年能干成？最少需要100亿美元。如果大家觉得我合适，为了小米，我愿意挺身而出。"
    }
  ]
}
```

**注意**: 
- `start` 和 `end` 时间戳可以保持不变（接口7会重新计算单词级时间戳）
- `text` 字段可以自由编辑（添加标点、修正错误、调整措辞等）
- `language` 字段必须与原始音频语言一致

**步骤3: 调用接口7进行对齐**

使用编辑后的 `transcript.json` 和原始音频文件调用接口7：
```bash
curl -X POST "http://localhost:8000/service/align?device=cuda" \
  -F "transcript=@transcript.json" \
  -F "file=@leijun15s.wav"
```

**步骤4: 获取对齐结果**

等待任务完成，然后获取结果：
```bash
curl -X GET "http://localhost:8000/task/{identifier}"
```

对齐后的结果会包含每个单词的精确时间戳。例如你得到的结果：

```json
{
  "status": "completed",
  "result": {
    "segments": [
      {
        "start": 0.031,
        "end": 11.998,
        "text": "苹果十年都干不成，凭啥你们三年能干成？最少需要100亿美元。如果大家觉得我合适，为了小米，我愿意挺身而出。",
        "words": [
          {
            "word": "苹",
            "start": 0.031,
            "end": 0.512,
            "score": 0.875
          },
          {
            "word": "果",
            "start": 0.512,
            "end": 0.532,
            "score": 0.974
          }
          // ... 更多单词
        ]
      }
    ],
    "word_segments": [
      // 所有单词的平铺列表
    ]
  }
}
```

## 对齐结果的使用

### 1. 保存对齐结果（用于后续接口9）

**提取对齐后的转录数据：**

从接口7的结果中，提取 `result` 部分，保存为 `aligned_transcript.json`：

```json
{
  "segments": [
    {
      "start": 0.031,
      "end": 11.998,
      "text": "苹果十年都干不成，凭啥你们三年能干成？最少需要100亿美元。如果大家觉得我合适，为了小米，我愿意挺身而出。",
      "words": [
        {
          "word": "苹",
          "start": 0.031,
          "end": 0.512,
          "score": 0.875
        }
        // ... 更多单词
      ]
    }
  ],
  "word_segments": [
    // ... 所有单词
  ]
}
```

**Python 脚本提取：**
```python
import json

# 接口7的返回结果
api7_result = {
    "status": "completed",
    "result": {
        "segments": [...],
        "word_segments": [...]
    }
}

# 提取对齐后的转录（用于接口9）
aligned_transcript = {
    "segments": api7_result["result"]["segments"],
    "word_segments": api7_result["result"]["word_segments"]
}

# 保存为文件
with open("aligned_transcript.json", "w", encoding="utf-8") as f:
    json.dump(aligned_transcript, f, ensure_ascii=False, indent=2)

print("✅ aligned_transcript.json 已保存，可用于接口9")
```

### 2. 下一步操作选项

**选项A: 继续说话人分离流程（推荐）**

如果你需要区分不同说话人，继续执行：
1. **接口8**: 说话人分离 → 得到 `diarization.json`
2. **接口9**: 合并对齐转录和说话人分离结果 → 得到最终带说话人标签的结果

**选项B: 直接使用对齐结果**

对齐结果已经可以用于：
- 生成字幕文件（SRT/VTT）
- 制作时间轴标注
- 音频标注和分析
- 提取特定时间段的文本

### 3. 生成字幕文件示例

```python
import json

# 读取对齐结果
with open("aligned_transcript.json", "r", encoding="utf-8") as f:
    aligned_data = json.load(f)

# 生成 SRT 字幕
def generate_srt(aligned_data):
    srt_content = []
    index = 1
    
    for segment in aligned_data["segments"]:
        start_time = format_time(segment["start"])
        end_time = format_time(segment["end"])
        text = segment["text"].strip()
        
        srt_content.append(f"{index}")
        srt_content.append(f"{start_time} --> {end_time}")
        srt_content.append(text)
        srt_content.append("")
        index += 1
    
    return "\n".join(srt_content)

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

# 生成并保存
srt_text = generate_srt(aligned_data)
with open("subtitle.srt", "w", encoding="utf-8") as f:
    f.write(srt_text)

print("✅ subtitle.srt 已生成")
```

**请求参数:**

- `transcript` (multipart/form-data, 必需): Whisper 格式的转录 JSON 文件
- `file` (multipart/form-data, 必需): 已转录的音频/视频文件
- `device` (query, 可选, 默认: "cuda"): PyTorch 推理设备
- `align_model` (query, 可选): 对齐模型名称
- `interpolate_method` (query, 可选, 默认: "nearest"): 插值方法
- `return_char_alignments` (query, 可选, 默认: false): 是否返回字符级对齐

**请求示例:**
```bash
curl -X POST "http://localhost:8000/service/align?device=cuda" \
  -F "transcript=@transcript.json" \
  -F "file=@audio.mp3"
```

**转录文件格式示例 (transcript.json):**

⚠️ **格式要求**: 必须是以下格式，不能包含 `status`、`metadata` 等字段！

```json
{
  "language": "zh",
  "segments": [
    {
      "start": 0.031,
      "end": 11.978,
      "text": "苹果十年都干不成，凭啥你们三年能干成？最少需要100亿美元。如果大家觉得我合适，为了小米，我愿意挺身而出。"
    }
  ]
}
```

**字段说明:**
- `language` (必需): 语言代码，必须与音频语言一致
- `segments` (必需): 数组，包含转录段
  - `start` (必需): 段的开始时间（秒）
  - `end` (必需): 段的结束时间（秒）
  - `text` (必需): 转录文本内容

**常见错误:**
1. ❌ 直接上传接口6的完整返回结果（包含status、metadata等）
2. ❌ 嵌套了 `result` 对象
3. ❌ 缺少必需的字段（language、segments、start、end、text）

**返回示例:**
```json
{
  "identifier": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Task queued"
}
```

**任务完成后的结果示例:**
```json
{
  "status": "completed",
  "result": {
    "segments": [
      {
        "start": 2.878,
        "end": 10.199,
        "text": " This is a test audio file of about phone line quality in English.",
        "words": [
          {
            "word": "This",
            "start": 2.878,
            "end": 3.059,
            "score": 0.676
          },
          {
            "word": "is",
            "start": 3.119,
            "end": 3.159,
            "score": 0.104
          }
        ]
      }
    ],
    "word_segments": [
      {
        "word": "This",
        "start": 2.878,
        "end": 3.059,
        "score": 0.676
      }
    ]
  },
  "metadata": {
    "task_type": "transcription_alignment",
    "language": "en",
    "file_name": "audio.mp3"
  },
  "error": null
}
```

---

### 8. 说话人分离（Diarization）

**接口路径:** `POST /service/diarize`

**接口说明:** 识别音频里有几个人在说话，以及每个人在什么时候说话。比如会议录音，能区分出是张三在0-5秒说话，李四在5-10秒说话。

**请求参数:**

- `file` (multipart/form-data, 必需): 要处理的音频/视频文件
- `device` (query, 可选, 默认: "cuda"): PyTorch 推理设备
- `min_speakers` (query, 可选): 最小说话人数量
- `max_speakers` (query, 可选): 最大说话人数量

**请求示例:**
```bash
curl -X POST "http://localhost:8000/service/diarize?device=cuda&min_speakers=1&max_speakers=3" \
  -F "file=@audio.mp3"
```

**返回示例:**
```json
{
  "identifier": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Task queued"
}
```

**任务完成后的结果示例:**
```json
{
  "status": "completed",
  "result": [
    {
      "label": "0",
      "speaker": "SPEAKER_00",
      "start": 2.8607809847198644,
      "end": 4.847198641765704
    },
    {
      "label": "0",
      "speaker": "SPEAKER_00",
      "start": 5.882852292020374,
      "end": 7.190152801358234
    },
    {
      "label": "1",
      "speaker": "SPEAKER_01",
      "start": 7.699490662139219,
      "end": 9.02376910016978
    }
  ],
  "metadata": {
    "task_type": "diarization",
    "file_name": "audio.mp3"
  },
  "error": null
}
```

---

### 9. 合并转录和说话人分离结果

**接口路径:** `POST /service/combine`

**接口说明:** 把文字内容和说话人信息合并在一起。比如把"苹果十年都干不成"这段文字标记为"SPEAKER_00说的"。这是最后一步，把前面步骤的结果组合起来。

**具体作用:**
- 把带时间戳的文字和说话人信息匹配起来
- 给每段文字标上是谁说的（SPEAKER_00、SPEAKER_01等）
- 得到最终结果：每段文字+时间+说话人
- **注意**: 这个接口不需要音频文件，只需要上传两个JSON文件（对齐结果和说话人分离结果）

**使用场景:**
1. **会议记录**: 需要知道谁在什么时候说了什么
2. **访谈转录**: 区分采访者和被采访者的发言
3. **多说话人音频处理**: 任何需要区分不同说话人的场景
4. **分步处理流程的最后一步**: 
   - 步骤1: 使用接口6进行转录
   - 步骤2: 使用接口7进行对齐（可选，但推荐）
   - 步骤3: 使用接口8进行说话人分离
   - 步骤4: 使用接口9合并结果
5. **手动编辑后重新合并**: 拿到接口7和接口8的结果后，可以手动编辑（调整说话人标签、修正时间戳等），然后使用接口9重新合并

**输入要求:**
- 必须提供对齐后的转录 JSON 文件（来自接口7的结果，或包含 words 字段的对齐转录）
- 必须提供说话人分离结果 JSON 文件（来自接口8的结果）
- **不需要音频文件**（纯数据处理）

**输出结果:**
- 每个转录段都包含 `speaker` 字段，标识该段文本是由哪个说话人说的

**手动编辑流程:**
1. 使用接口7获取对齐后的转录 `aligned_transcript.json`
2. 使用接口8获取说话人分离结果 `diarization.json`
3. 手动编辑这两个 JSON 文件（如修正说话人标签、调整时间段等）
4. 使用接口9，上传编辑后的两个 JSON 文件
5. 接口9会基于编辑后的数据重新合并，生成最终结果

**请求参数:**

- `aligned_transcript` (multipart/form-data, 必需): 对齐后的转录 JSON 文件
- `diarization_result` (multipart/form-data, 必需): 说话人分离结果 JSON 文件

**请求示例:**
```bash
curl -X POST "http://localhost:8000/service/combine" \
  -F "aligned_transcript=@aligned_transcript.json" \
  -F "diarization_result=@diarization.json"
```

**对齐转录文件格式示例 (aligned_transcript.json):**

⚠️ **格式要求**: 必须从接口7的返回结果中提取 `result` 部分，不能包含 `status`、`metadata` 等字段！

从接口7的返回结果中，**只提取 `result` 对象的内容**：

```json
{
  "segments": [
    {
      "start": 0.031,
      "end": 11.998,
      "text": "苹果十年都干不成，凭啥你们三年能干成？最少需要100亿美元。如果大家觉得我合适，为了小米，我愿意挺身而出。",
      "words": [
        {
          "word": "苹",
          "start": 0.031,
          "end": 0.512,
          "score": 0.875
        },
        {
          "word": "果",
          "start": 0.512,
          "end": 0.532,
          "score": 0.974
        }
        // ... 更多单词
      ]
    }
  ],
  "word_segments": [
    {
      "word": "苹",
      "start": 0.031,
      "end": 0.512,
      "score": 0.875
    },
    {
      "word": "果",
      "start": 0.512,
      "end": 0.532,
      "score": 0.974
    }
    // ... 所有单词的平铺列表
  ]
}
```

**字段说明:**
- `segments` (必需): 数组，包含对齐后的转录段
  - `start` (必需): 段的开始时间（秒）
  - `end` (必需): 段的结束时间（秒）
  - `text` (必需): 转录文本内容
  - `words` (必需): 数组，包含该段中每个单词/字符的详细信息
    - `word` (必需): 单词或字符
    - `start` (必需): 单词的开始时间（秒）
    - `end` (必需): 单词的结束时间（秒）
    - `score` (必需): 置信度分数（0-1之间）
- `word_segments` (必需): 数组，所有单词/字符的平铺列表（与segments中的words内容相同，但格式更扁平）

**错误示例（会导致报错）:**
```json
{
  "status": "completed",  // ❌ 不要包含这些字段
  "result": {...},        // ❌ 不要嵌套result
  "metadata": {...}      // ❌ 不要包含metadata
}
```

**正确提取方法:**

从接口7的返回结果中，提取 `result` 对象：
```python
import json

# 接口7的返回结果
api7_result = {
    "status": "completed",
    "result": {
        "segments": [
            {
                "start": 0.031,
                "end": 11.998,
                "text": "苹果十年都干不成，凭啥你们三年能干成？最少需要100亿美元。如果大家觉得我合适，为了小米，我愿意挺身而出。",
                "words": [...]
            }
        ],
        "word_segments": [...]
    },
    "metadata": {...}
}

# 提取result对象（用于接口9）
aligned_transcript = {
    "segments": api7_result["result"]["segments"],
    "word_segments": api7_result["result"]["word_segments"]
}

# 保存为文件
with open("aligned_transcript.json", "w", encoding="utf-8") as f:
    json.dump(aligned_transcript, f, ensure_ascii=False, indent=2)

print("✅ aligned_transcript.json 已保存，可用于接口9")
```

**说话人分离结果文件格式示例 (diarization.json):**

⚠️ **格式要求**: 必须是数组格式，不能包含 `status`、`metadata` 等字段！

从接口8的返回结果中，**只提取 `result` 数组部分**：

```json
[
  {
    "label": "SPEAKER_00",
    "speaker": "SPEAKER_00",
    "start": 0.03096875,
    "end": 7.7428437500000005
  },
  {
    "label": "SPEAKER_00",
    "speaker": "SPEAKER_00",
    "start": 8.164718750000002,
    "end": 9.09284375
  },
  {
    "label": "SPEAKER_00",
    "speaker": "SPEAKER_00",
    "start": 9.632843750000003,
    "end": 11.978468750000001
  }
]
```

**错误示例（会导致报错）:**
```json
{
  "status": "completed",  // ❌ 不要包含这些字段
  "result": [...],        // ❌ 不要嵌套result
  "metadata": {...}      // ❌ 不要包含metadata
}
```

**正确提取方法:**

从接口8的返回结果中，提取 `result` 数组：
```python
import json

# 接口8的返回结果
api8_result = {
    "status": "completed",
    "result": [
        {
            "label": "SPEAKER_00",
            "speaker": "SPEAKER_00",
            "start": 0.03096875,
            "end": 7.7428437500000005
        }
    ],
    "metadata": {...}
}

# 提取result数组（用于接口9）
diarization_data = api8_result["result"]

# 保存为文件
with open("diarization.json", "w", encoding="utf-8") as f:
    json.dump(diarization_data, f, ensure_ascii=False, indent=2)

print("✅ diarization.json 已保存，可用于接口9")
```

**返回示例:**
```json
{
  "identifier": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Task queued"
}
```

**任务完成后的结果示例:**
```json
{
  "status": "completed",
  "result": {
    "segments": [
      {
        "start": 2.878,
        "end": 4.847,
        "text": " This is a test",
        "speaker": "SPEAKER_00"
      },
      {
        "start": 5.882,
        "end": 7.190,
        "text": " audio file.",
        "speaker": "SPEAKER_01"
      }
    ]
  },
  "metadata": {
    "task_type": "combine_transcript&diarization"
  },
  "error": null
}
```

---

## 任务管理接口

### 10. 获取所有任务状态

**接口路径:** `GET /task/all`

**接口说明:** 查看所有处理任务的状态。会返回一个列表，显示每个任务是否完成、处理了哪个文件等信息。

**请求参数:** 无

**请求示例:**
```bash
curl -X GET "http://localhost:8000/task/all"
```

**返回示例:**
```json
{
  "tasks": [
    {
      "identifier": "550e8400-e29b-41d4-a716-446655440000",
      "status": "completed",
      "task_type": "full_process",
      "file_name": "audio.mp3",
      "url": null,
      "audio_duration": 120.5,
      "language": "en",
      "error": null,
      "duration": 45.2,
      "start_time": "2024-01-15T10:00:00Z",
      "end_time": "2024-01-15T10:00:45Z"
    },
    {
      "identifier": "660e8400-e29b-41d4-a716-446655440001",
      "status": "processing",
      "task_type": "transcription",
      "file_name": "video.mp4",
      "url": null,
      "audio_duration": 300.0,
      "language": "zh",
      "error": null,
      "duration": null,
      "start_time": "2024-01-15T10:05:00Z",
      "end_time": null
    },
    {
      "identifier": "770e8400-e29b-41d4-a716-446655440002",
      "status": "failed",
      "task_type": "diarization",
      "file_name": "audio.wav",
      "url": null,
      "audio_duration": 60.0,
      "language": null,
      "error": "File format not supported",
      "duration": 5.0,
      "start_time": "2024-01-15T10:10:00Z",
      "end_time": "2024-01-15T10:10:05Z"
    }
  ]
}
```

---

### 11. 获取特定任务状态

**接口路径:** `GET /task/{identifier}`

**接口说明:** 查看某个具体任务的处理结果。输入任务ID，就能看到这个任务是否完成、处理结果是什么、有没有出错等详细信息。

**请求参数:**

- `identifier` (path, 必需): 任务标识符（UUID）

**请求示例:**
```bash
curl -X GET "http://localhost:8000/task/550e8400-e29b-41d4-a716-446655440000"
```

**返回示例（处理中）:**
```json
{
  "status": "processing",
  "result": null,
  "metadata": {
    "task_type": "full_process",
    "task_params": {
      "language": "en",
      "model": "base",
      "device": "cuda"
    },
    "language": "en",
    "file_name": "audio.mp3",
    "url": null,
    "duration": null,
    "audio_duration": 120.5,
    "start_time": "2024-01-15T10:00:00Z",
    "end_time": null
  },
  "error": null
}
```

**返回示例（已完成）:**
```json
{
  "status": "completed",
  "result": {
    "segments": [
      {
        "start": 0.0,
        "end": 5.5,
        "text": "Hello, this is a test transcription.",
        "speaker": "SPEAKER_00"
      },
      {
        "start": 5.5,
        "end": 10.2,
        "text": "This is another segment from a different speaker.",
        "speaker": "SPEAKER_01"
      }
    ]
  },
  "metadata": {
    "task_type": "full_process",
    "task_params": {
      "language": "en",
      "model": "base",
      "device": "cuda"
    },
    "language": "en",
    "file_name": "audio.mp3",
    "url": null,
    "duration": 45.2,
    "audio_duration": 120.5,
    "start_time": "2024-01-15T10:00:00Z",
    "end_time": "2024-01-15T10:00:45Z"
  },
  "error": null
}
```

**返回示例（失败）:**
```json
{
  "status": "failed",
  "result": null,
  "metadata": {
    "task_type": "transcription",
    "task_params": {...},
    "language": "en",
    "file_name": "audio.mp3",
    "url": null,
    "duration": 5.0,
    "audio_duration": 120.5,
    "start_time": "2024-01-15T10:00:00Z",
    "end_time": "2024-01-15T10:00:05Z"
  },
  "error": "File format not supported or corrupted"
}
```

---

### 12. 删除任务

**接口路径:** `DELETE /task/{identifier}/delete`

**接口说明:** 删除某个任务记录。输入任务ID，就能把这个任务从系统中删除。

**请求参数:**

- `identifier` (path, 必需): 任务标识符（UUID）

**请求示例:**
```bash
curl -X DELETE "http://localhost:8000/task/550e8400-e29b-41d4-a716-446655440000/delete"
```

**返回示例:**
```json
{
  "identifier": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Task deleted"
}
```

**错误返回示例（任务不存在）:**
```json
{
  "detail": "Task with identifier 550e8400-e29b-41d4-a716-446655440000 not found"
}
```

---

## 任务状态说明

- **processing**: 任务正在处理中
- **completed**: 任务已完成
- **failed**: 任务处理失败

## 任务类型说明

- **transcription**: 仅转录
- **transcription_alignment**: 转录和对齐
- **diarization**: 说话人分离
- **combine_transcript&diarization**: 合并转录和说话人分离结果
- **full_process**: 完整处理（转录、对齐、说话人分离、合并）

## 支持的音频/视频格式

根据配置，支持以下文件扩展名：
- 音频格式: `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg` 等
- 视频格式: `.mp4`, `.avi`, `.mov`, `.flv` 等

具体支持的格式请参考配置文件中的 `AUDIO_EXTENSIONS` 和 `VIDEO_EXTENSIONS`。

## 分步处理流程示例

以下是一个完整的分步处理流程，展示了如何使用接口6-9来逐步处理音频：

### 场景：处理一个会议录音，需要区分不同说话人

**步骤1: 转录** (接口6)
```bash
# 上传音频文件进行转录
curl -X POST "http://localhost:8000/service/transcribe?language=en&model=base" \
  -F "file=@meeting.mp3"
# 返回: {"identifier": "task-1", "message": "Task queued"}
```

**步骤2: 获取转录结果**
```bash
# 等待任务完成，然后获取结果
curl -X GET "http://localhost:8000/task/task-1"
# 返回的 result 字段包含转录文本，保存为 transcript.json
```

**步骤3: 对齐** (接口7)
```bash
# 将对齐转录文本，为每个单词添加时间戳
curl -X POST "http://localhost:8000/service/align?device=cuda" \
  -F "transcript=@transcript.json" \
  -F "file=@meeting.mp3"
# 返回: {"identifier": "task-2", "message": "Task queued"}
```

**步骤4: 获取对齐结果**
```bash
# 等待任务完成，获取对齐后的转录
curl -X GET "http://localhost:8000/task/task-2"
# 返回的 result 包含带单词级时间戳的转录，保存为 aligned_transcript.json
```

**步骤5: 说话人分离** (接口8)
```bash
# 识别音频中的不同说话人
curl -X POST "http://localhost:8000/service/diarize?device=cuda&min_speakers=2&max_speakers=5" \
  -F "file=@meeting.mp3"
# 返回: {"identifier": "task-3", "message": "Task queued"}
```

**步骤6: 获取说话人分离结果**
```bash
# 等待任务完成，获取说话人时间段
curl -X GET "http://localhost:8000/task/task-3"
# 返回的 result 包含说话人时间段，保存为 diarization.json
```

**步骤7: 合并结果** (接口9)
```bash
# 将对齐的转录和说话人分离结果合并
curl -X POST "http://localhost:8000/service/combine" \
  -F "aligned_transcript=@aligned_transcript.json" \
  -F "diarization_result=@diarization.json"
# 返回: {"identifier": "task-4", "message": "Task queued"}
```

**步骤8: 获取最终结果**
```bash
# 获取最终的带说话人标签的转录结果
curl -X GET "http://localhost:8000/task/task-4"
# 返回的 result 包含每个转录段及其对应的说话人
```

**最终结果示例:**
```json
{
  "segments": [
    {
      "start": 0.0,
      "end": 5.5,
      "text": "大家好，欢迎参加今天的会议。",
      "speaker": "SPEAKER_00"
    },
    {
      "start": 5.5,
      "end": 12.3,
      "text": "谢谢，我们今天要讨论项目进度。",
      "speaker": "SPEAKER_01"
    }
  ]
}
```

### 接口7 vs 接口9 的对比

虽然接口7和接口9都可以在拿到中间结果后手动编辑再执行，但它们有重要区别：

| 特性 | 接口7 (对齐) | 接口9 (合并) |
|------|-------------|-------------|
| **输入** | 转录JSON + **音频文件** | 对齐转录JSON + 说话人分离JSON |
| **是否需要音频** | ✅ 是（必须） | ❌ 否 |
| **处理类型** | 音频分析（对齐） | 纯数据处理（合并） |
| **可编辑性** | 可编辑转录文本后重新对齐 | 可编辑两个JSON后重新合并 |
| **计算复杂度** | 高（需要音频分析） | 低（纯数据匹配） |

**关键区别:**
- **接口7需要音频文件**：因为它要分析音频波形来对齐文本，即使你编辑了转录文本，也需要原始音频来重新计算时间戳
- **接口9不需要音频文件**：它只是将两个JSON结果进行时间匹配和合并，是纯数据处理操作

### 何时使用分步处理 vs 完整处理

**使用分步处理（接口6-9）的情况:**
- 需要中间结果（如只要转录，不需要说话人信息）
- 需要自定义处理流程（如只对齐，不进行说话人分离）
- 需要重用中间结果（如多个音频使用同一个对齐模型）
- 需要调试或质量控制（检查每一步的结果）
- **需要手动编辑中间结果**（如修正转录错误、调整说话人标签等）

**使用完整处理（接口4或5）的情况:**
- 需要完整的最终结果（转录+对齐+说话人分离）
- 不需要中间步骤的结果
- 希望一次性完成所有处理
- 处理流程固定，不需要自定义
- 不需要手动编辑中间结果

## 字幕文件生成指南

### 字幕文件格式说明

#### 1. SRT 格式（SubRip Subtitle）

**格式说明:**
- 最常用的字幕格式
- 支持大多数视频播放器
- 格式简单，易于编辑

**SRT 文件结构:**
```
序号
开始时间 --> 结束时间
字幕文本
（空行）
```

**时间格式:** `HH:MM:SS,mmm`（小时:分钟:秒,毫秒）

**示例文件 (subtitle.srt):**
```
1
00:00:00,031 --> 00:00:11,998
苹果十年都干不成，凭啥你们三年能干成？最少需要100亿美元。如果大家觉得我合适，为了小米，我愿意挺身而出。

2
00:00:12,000 --> 00:00:20,500
这是第二段字幕内容。
```

#### 2. VTT 格式（WebVTT）

**格式说明:**
- 用于网页视频播放
- 支持HTML5 video标签
- 可以包含样式和定位信息

**VTT 文件结构:**
```
WEBVTT

序号
开始时间 --> 结束时间
字幕文本
```

**时间格式:** `HH:MM:SS.mmm`（小时:分钟:秒.毫秒）

**示例文件 (subtitle.vtt):**
```
WEBVTT

1
00:00:00.031 --> 00:00:11.998
苹果十年都干不成，凭啥你们三年能干成？最少需要100亿美元。如果大家觉得我合适，为了小米，我愿意挺身而出。

2
00:00:12.000 --> 00:00:20.500
这是第二段字幕内容。
```

### 从对齐结果生成字幕文件

#### Python 脚本：生成 SRT 字幕

```python
import json

def format_time_srt(seconds):
    """将秒数转换为SRT时间格式 (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def generate_srt_from_aligned(aligned_data):
    """从对齐结果生成SRT字幕"""
    srt_content = []
    index = 1
    
    for segment in aligned_data["segments"]:
        start_time = format_time_srt(segment["start"])
        end_time = format_time_srt(segment["end"])
        text = segment["text"].strip()
        
        # SRT格式：序号、时间轴、文本、空行
        srt_content.append(f"{index}")
        srt_content.append(f"{start_time} --> {end_time}")
        srt_content.append(text)
        srt_content.append("")
        index += 1
    
    return "\n".join(srt_content)

# 使用示例
# 从接口7的结果中提取
api7_result = {
    "status": "completed",
    "result": {
        "segments": [
            {
                "start": 0.031,
                "end": 11.998,
                "text": "苹果十年都干不成，凭啥你们三年能干成？最少需要100亿美元。如果大家觉得我合适，为了小米，我愿意挺身而出。",
                "words": [...]
            }
        ]
    }
}

# 提取对齐数据
aligned_data = {
    "segments": api7_result["result"]["segments"]
}

# 生成SRT
srt_text = generate_srt_from_aligned(aligned_data)

# 保存文件
with open("subtitle.srt", "w", encoding="utf-8") as f:
    f.write(srt_text)

print("✅ subtitle.srt 已生成")
```

#### Python 脚本：生成 VTT 字幕

```python
def format_time_vtt(seconds):
    """将秒数转换为VTT时间格式 (HH:MM:SS.mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

def generate_vtt_from_aligned(aligned_data):
    """从对齐结果生成VTT字幕"""
    vtt_content = ["WEBVTT", ""]  # VTT文件必须以WEBVTT开头
    index = 1
    
    for segment in aligned_data["segments"]:
        start_time = format_time_vtt(segment["start"])
        end_time = format_time_vtt(segment["end"])
        text = segment["text"].strip()
        
        # VTT格式：序号、时间轴、文本
        vtt_content.append(f"{index}")
        vtt_content.append(f"{start_time} --> {end_time}")
        vtt_content.append(text)
        vtt_content.append("")
        index += 1
    
    return "\n".join(vtt_content)

# 生成VTT
vtt_text = generate_vtt_from_aligned(aligned_data)

# 保存文件
with open("subtitle.vtt", "w", encoding="utf-8") as f:
    f.write(vtt_text)

print("✅ subtitle.vtt 已生成")
```

#### 高级功能：按单词分割生成字幕

如果需要更精细的字幕（每个单词一行），可以使用 `word_segments`：

```python
def generate_word_level_srt(aligned_data, max_words_per_line=8):
    """生成单词级别的SRT字幕（每行最多N个单词）"""
    srt_content = []
    index = 1
    current_words = []
    current_start = None
    current_end = None
    
    for segment in aligned_data["segments"]:
        for word_info in segment.get("words", []):
            word = word_info["word"]
            start = word_info["start"]
            end = word_info["end"]
            
            if current_start is None:
                current_start = start
            
            current_words.append(word)
            current_end = end
            
            # 当达到最大单词数或遇到标点符号时，生成一行字幕
            if len(current_words) >= max_words_per_line or word in "。，！？；：":
                text = "".join(current_words)
                srt_content.append(f"{index}")
                srt_content.append(f"{format_time_srt(current_start)} --> {format_time_srt(current_end)}")
                srt_content.append(text)
                srt_content.append("")
                index += 1
                current_words = []
                current_start = None
        
        # 处理剩余的单词
        if current_words:
            text = "".join(current_words)
            srt_content.append(f"{index}")
            srt_content.append(f"{format_time_srt(current_start)} --> {format_time_srt(current_end)}")
            srt_content.append(text)
            srt_content.append("")
            index += 1
            current_words = []
            current_start = None
    
    return "\n".join(srt_content)
```

### 完整示例脚本

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从WhisperX对齐结果生成字幕文件
支持SRT和VTT格式
"""

import json
import sys

def format_time_srt(seconds):
    """SRT时间格式: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def format_time_vtt(seconds):
    """VTT时间格式: HH:MM:SS.mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

def generate_srt(aligned_data):
    """生成SRT字幕"""
    srt_content = []
    index = 1
    
    for segment in aligned_data["segments"]:
        start_time = format_time_srt(segment["start"])
        end_time = format_time_srt(segment["end"])
        text = segment["text"].strip()
        
        srt_content.append(f"{index}")
        srt_content.append(f"{start_time} --> {end_time}")
        srt_content.append(text)
        srt_content.append("")
        index += 1
    
    return "\n".join(srt_content)

def generate_vtt(aligned_data):
    """生成VTT字幕"""
    vtt_content = ["WEBVTT", ""]
    index = 1
    
    for segment in aligned_data["segments"]:
        start_time = format_time_vtt(segment["start"])
        end_time = format_time_vtt(segment["end"])
        text = segment["text"].strip()
        
        vtt_content.append(f"{index}")
        vtt_content.append(f"{start_time} --> {end_time}")
        vtt_content.append(text)
        vtt_content.append("")
        index += 1
    
    return "\n".join(vtt_content)

def main():
    # 读取对齐结果JSON文件
    if len(sys.argv) < 2:
        print("用法: python generate_subtitle.py <aligned_transcript.json>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    with open(input_file, "r", encoding="utf-8") as f:
        aligned_data = json.load(f)
    
    # 生成SRT
    srt_text = generate_srt(aligned_data)
    srt_file = input_file.replace(".json", ".srt")
    with open(srt_file, "w", encoding="utf-8") as f:
        f.write(srt_text)
    print(f"✅ {srt_file} 已生成")
    
    # 生成VTT
    vtt_text = generate_vtt(aligned_data)
    vtt_file = input_file.replace(".json", ".vtt")
    with open(vtt_file, "w", encoding="utf-8") as f:
        f.write(vtt_text)
    print(f"✅ {vtt_file} 已生成")

if __name__ == "__main__":
    main()
```

**使用方法:**
```bash
# 从接口7的结果中提取并保存 aligned_transcript.json
# 然后运行脚本
python generate_subtitle.py aligned_transcript.json
```

### 字幕文件使用

**SRT文件:**
- 可以直接导入到视频编辑软件（Premiere、Final Cut Pro等）
- 大多数视频播放器都支持（VLC、PotPlayer等）
- 可以上传到视频平台（YouTube、Bilibili等）

**VTT文件:**
- 主要用于网页视频播放
- 可以在HTML中使用：
```html
<video controls>
  <source src="video.mp4" type="video/mp4">
  <track kind="subtitles" src="subtitle.vtt" srclang="zh" label="中文">
</video>
```

## Whisper 模型说明

### 模型列表

系统支持以下 Whisper 模型：

#### 标准模型（多语言）

| 模型名称 | 参数量 | 模型大小 | 速度 | 准确度 | 适用场景 |
|---------|--------|---------|------|--------|---------|
| `tiny` | 39M | ~75MB | 最快 | 较低 | 快速测试、资源受限环境 |
| `base` | 74M | ~142MB | 快 | 中等 | 日常使用、平衡速度和准确度 |
| `small` | 244M | ~466MB | 中等 | 较好 | 一般生产环境 |
| `medium` | 769M | ~1.4GB | 较慢 | 好 | 高质量转录需求 |
| `large` | 1550M | ~2.9GB | 慢 | 最好 | 最高质量转录（已弃用，建议用large-v2/v3） |
| `large-v1` | 1550M | ~2.9GB | 慢 | 很好 | 高质量转录（旧版本） |
| `large-v2` | 1550M | ~2.9GB | 慢 | 很好 | 高质量转录（推荐） |
| `large-v3` | 1550M | ~3.1GB | 慢 | 最好 | 最新版本，最高准确度 |
| `large-v3-turbo` | 1550M | ~3.1GB | 较快 | 最好 | 优化版本，速度更快 |

#### 英语专用模型（仅支持英语，但速度更快）

| 模型名称 | 参数量 | 模型大小 | 速度 | 准确度 | 适用场景 |
|---------|--------|---------|------|--------|---------|
| `tiny.en` | 39M | ~75MB | 最快 | 较低（仅英语） | 英语快速转录 |
| `base.en` | 74M | ~142MB | 快 | 中等（仅英语） | 英语日常转录 |
| `small.en` | 244M | ~466MB | 中等 | 较好（仅英语） | 英语高质量转录 |
| `medium.en` | 769M | ~1.4GB | 较慢 | 好（仅英语） | 英语最高质量转录 |

#### 蒸馏模型（Distilled，更小更快）

| 模型名称 | 参数量 | 模型大小 | 速度 | 准确度 | 适用场景 |
|---------|--------|---------|------|--------|---------|
| `distil-small.en` | ~ | ~ | 快 | 中等（仅英语） | 英语快速转录 |
| `distil-medium.en` | ~ | ~ | 中等 | 较好（仅英语） | 英语平衡转录 |
| `distil-large-v2` | ~ | ~ | 中等 | 好 | 多语言快速转录 |
| `distil-large-v3` | ~ | ~ | 中等 | 好 | 多语言快速转录（最新） |

#### 自定义模型

| 模型名称 | 说明 |
|---------|------|
| `nyrahealth/faster_CrisperWhisper` | 第三方优化模型，速度更快 |

### 模型选择建议

**根据需求选择：**

1. **快速测试/开发**：使用 `tiny` 或 `base`
   - 速度快，资源占用少
   - 适合快速验证功能

2. **日常使用**：使用 `base` 或 `small`
   - 平衡速度和准确度
   - 适合大多数场景

3. **高质量转录**：使用 `medium` 或 `large-v3`
   - 准确度更高
   - 适合正式生产环境

4. **仅英语内容**：使用 `.en` 后缀的模型
   - 速度更快
   - 准确度可能更高（仅限英语）

5. **资源受限**：使用 `tiny`、`base` 或蒸馏模型
   - 内存占用小
   - 适合CPU或低配置GPU

**根据硬件选择：**

- **CPU 或 低配置 GPU（<4GB显存）**：`tiny`、`base`、`small`
- **中等配置 GPU（4-8GB显存）**：`small`、`medium`
- **高配置 GPU（>8GB显存）**：`medium`、`large-v2`、`large-v3`

**注意：**
- 模型会在首次使用时自动下载
- 模型大小是下载后的磁盘占用
- 运行时需要额外的GPU显存（通常是模型大小的1.5-2倍）
- 使用 `large-v3-turbo` 可以获得接近 `large-v3` 的准确度，但速度更快

## 注意事项

1. 所有处理任务都是异步的，接口会立即返回任务标识符，需要通过任务管理接口查询处理状态。
2. 文件上传接口有大小限制，请根据服务器配置调整。
3. 某些功能（如说话人分离）可能需要特定的模型文件，请确保已正确配置。
4. 使用 GPU（cuda）可以显著提高处理速度，但需要确保系统已正确配置 CUDA 环境。
5. **分步处理时注意**: 接口7需要接口6的结果，接口9需要接口7和接口8的结果，请确保按顺序执行。
6. **字幕生成注意**: 
   - SRT格式使用逗号分隔毫秒（`00:00:00,031`）
   - VTT格式使用点分隔毫秒（`00:00:00.031`）
   - 确保文件编码为UTF-8，以支持中文等多语言字符
7. **模型选择注意**:
   - 模型越大，准确度越高，但速度越慢，需要更多显存
   - 首次使用某个模型时会自动下载，需要网络连接
   - 建议根据实际需求选择合适的模型，不要盲目使用最大的模型

8. **中文简繁体问题**:
   - Whisper 模型的 `zh` 语言代码不区分简体和繁体中文
   - 模型训练数据中包含了简体和繁体中文，所以输出可能混合出现
   - 这是正常现象，因为模型会根据音频内容、上下文等因素自动选择
   - **解决方案**：如果需要统一为简体或繁体，可以在后处理时使用转换工具（如 OpenCC）

**中文简繁体转换示例（Python）:**

如果需要将转录结果统一转换为简体或繁体，可以使用 OpenCC 库：

```python
# 安装: pip install opencc-python-reimplemented
import opencc

# 创建转换器
# s2t: 简体转繁体
# t2s: 繁体转简体
# s2tw: 简体转台湾繁体
# tw2s: 台湾繁体转简体
# s2hk: 简体转香港繁体
# hk2s: 香港繁体转简体

converter = opencc.OpenCC('t2s')  # 繁体转简体

# 转换转录结果
transcript_data = {
    "segments": [
        {
            "text": "蘋果十年都幹不成，憑啥你們三年能幹成？",  # 繁体
            "start": 0.031,
            "end": 11.978
        }
    ]
}

# 转换文本
for segment in transcript_data["segments"]:
    segment["text"] = converter.convert(segment["text"])

# 结果: "苹果十年都干不成，凭啥你们三年能干成？"  # 简体
```

**在 Web 界面中添加简繁转换:**

可以在 `web_interface/app.js` 中添加转换功能：

```javascript
// 使用 opencc-js 库（需要在 HTML 中引入）
// <script src="https://cdn.jsdelivr.net/npm/opencc-js@1.0.3/dist/umd/opencc.min.js"></script>

function convertToSimplified(text) {
    const converter = new OpenCC('t2s');
    return converter.convert(text);
}

function convertToTraditional(text) {
    const converter = new OpenCC('s2t');
    return converter.convert(text);
}

// 在显示转录结果时使用
function displayTranscript(segments) {
    const convertToSimplified = true; // 是否转换为简体
    
    transcriptContent.innerHTML = segments.map((segment, index) => {
        let text = segment.text;
        if (convertToSimplified) {
            // 使用 opencc-js 转换
            text = convertToSimplified(text);
        }
        // ... 显示逻辑
    }).join('');
}
```

