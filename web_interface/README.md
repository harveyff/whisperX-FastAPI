# WhisperX Web 界面

这是一个完整的 Web 界面，用于上传视频/音频文件，自动生成字幕并播放。

## 功能特性

- ✅ 拖拽上传文件（支持音频和视频）
- ✅ 实时处理进度显示
- ✅ 自动生成字幕（SRT/VTT格式）
- ✅ 视频播放器集成字幕显示
- ✅ 转录文本展示和交互
- ✅ 点击文本跳转到对应时间点
- ✅ 播放时自动高亮当前文本段
- ✅ 下载字幕和JSON结果
- ✅ **中文简繁转换**（自动统一为简体或繁体）

## 使用方法

### 1. 启动后端服务

确保 WhisperX FastAPI 服务正在运行：

```bash
# 在项目根目录
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 2. 打开 Web 界面

直接在浏览器中打开 `index.html` 文件，或者使用本地服务器：

```bash
# 使用 Python 简单服务器
cd web_interface
python -m http.server 8080

# 然后访问 http://localhost:8080
```

### 3. 配置 API 地址

如果后端服务不在 `http://localhost:8000`，请修改 `app.js` 文件中的 `API_BASE_URL`：

```javascript
const API_BASE_URL = 'http://your-api-url:8000';
```

## 使用流程

1. **上传文件**：点击或拖拽文件到上传区域
2. **配置选项**：
   - 选择语言（中文、英文等）
   - 选择模型（推荐使用 base 或 small）
   - 选择设备（GPU 或 CPU）
   - 是否启用说话人分离
   - **中文转换**（繁体转简体、简体转繁体等）
3. **开始处理**：点击"开始处理"按钮
4. **等待完成**：系统会自动处理并显示进度
5. **查看结果**：
   - 视频播放器会自动加载字幕
   - 右侧显示转录文本
   - 可以下载字幕文件和JSON结果

## 文件结构

```
web_interface/
├── index.html      # 主页面
├── style.css       # 样式文件
├── app.js          # JavaScript 逻辑
└── README.md       # 说明文档
```

## 注意事项

1. 确保后端 API 服务正常运行
2. 首次使用某个模型时会自动下载，需要网络连接
3. 处理大文件或使用大模型时可能需要较长时间
4. 建议使用 Chrome 或 Edge 浏览器以获得最佳体验

## 自定义配置

### 修改 API 地址

在 `app.js` 中修改：

```javascript
const API_BASE_URL = 'http://your-api-url:8000';
```

### 添加更多语言选项

在 `index.html` 的 `<select id="language">` 中添加更多选项。

### 修改样式

编辑 `style.css` 文件来自定义界面样式。

