# 中文简繁体转换工具

## 工具说明

这些脚本用于将 WhisperX 转录结果中的中文统一转换为简体或繁体。

## 安装依赖

```bash
pip install opencc-python-reimplemented
```

## 使用方法

### 1. 单文件转换 (convert_chinese.py)

转换单个 JSON 文件：

```bash
# 繁体转简体（默认，推荐）
python scripts/convert_chinese.py input.json output.json

# 简体转繁体
python scripts/convert_chinese.py input.json output.json --type s2t

# 简体转台湾繁体
python scripts/convert_chinese.py input.json output.json --type s2tw
```

**支持的转换类型：**
- `t2s`: 繁体转简体（默认）
- `s2t`: 简体转繁体
- `s2tw`: 简体转台湾繁体
- `tw2s`: 台湾繁体转简体
- `s2hk`: 简体转香港繁体
- `hk2s`: 香港繁体转简体

**使用示例：**

```bash
# 转换接口6的转录结果
python scripts/convert_chinese.py transcript.json transcript_simplified.json

# 转换接口7的对齐结果
python scripts/convert_chinese.py aligned_transcript.json aligned_transcript_simplified.json

# 转换接口9的最终结果
python scripts/convert_chinese.py final_result.json final_result_simplified.json
```

### 2. 批量转换 (convert_chinese_batch.py)

批量转换目录中的所有 JSON 文件：

```bash
# 转换目录中所有文件（繁体转简体）
python scripts/convert_chinese_batch.py ./results

# 指定转换类型
python scripts/convert_chinese_batch.py ./results s2t
```

**注意：**
- 会创建 `.bak` 备份文件
- 直接修改原文件

## 支持的格式

工具支持以下 JSON 格式：

1. **接口返回的完整格式**（包含 status、result、metadata）
2. **对齐后的转录格式**（segments + word_segments）
3. **最终合并结果格式**（带 speaker 标签的 segments）

## 示例

### 转换前（繁体）：
```json
{
  "segments": [
    {
      "text": "蘋果十年都幹不成，憑啥你們三年能幹成？",
      "start": 0.031,
      "end": 11.978
    }
  ]
}
```

### 转换后（简体）：
```json
{
  "segments": [
    {
      "text": "苹果十年都干不成，凭啥你们三年能干成？",
      "start": 0.031,
      "end": 11.978
    }
  ]
}
```

## 注意事项

1. 转换会保留所有其他字段（时间戳、说话人标签等）
2. 只转换文本内容，不影响时间戳和元数据
3. 建议在转换前备份原文件
4. 批量转换会自动创建 `.bak` 备份文件

