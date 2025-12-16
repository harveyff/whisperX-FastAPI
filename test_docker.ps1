# Docker 容器测试脚本
# 用于测试构建成功的 whisperx-service 镜像

$IMAGE_NAME = "whisperx-service"
$CONTAINER_NAME = "whisperx-test"

Write-Host "`n=== Docker 容器测试脚本 ===" -ForegroundColor Green
Write-Host ""

# 检查 .env 文件
if (-not (Test-Path .env)) {
    Write-Host "错误: 未找到 .env 文件！" -ForegroundColor Red
    Write-Host "请创建 .env 文件，包含以下内容：" -ForegroundColor Yellow
    Write-Host "HF_TOKEN=你的HuggingFace Token"
    Write-Host "WHISPER_MODEL=base"
    Write-Host "LOG_LEVEL=info"
    exit 1
}

# 检查镜像是否存在
Write-Host "1. 检查镜像是否存在..." -ForegroundColor Cyan
$imageExists = docker images --format "{{.Repository}}" | Select-String -Pattern "^${IMAGE_NAME}$"
if (-not $imageExists) {
    Write-Host "错误: 镜像 ${IMAGE_NAME} 不存在！" -ForegroundColor Red
    Write-Host "请先运行: docker build -t ${IMAGE_NAME} ." -ForegroundColor Yellow
    exit 1
}
Write-Host "✓ 镜像存在" -ForegroundColor Green

# 停止并删除旧容器（如果存在）
Write-Host "`n2. 清理旧容器（如果存在）..." -ForegroundColor Cyan
docker stop $CONTAINER_NAME 2>$null
docker rm $CONTAINER_NAME 2>$null
Write-Host "✓ 清理完成" -ForegroundColor Green

# 启动容器
Write-Host "`n3. 启动容器..." -ForegroundColor Cyan
docker run -d --name $CONTAINER_NAME --gpus all -p 8000:8000 --env-file .env $IMAGE_NAME
if ($LASTEXITCODE -ne 0) {
    Write-Host "错误: 容器启动失败！" -ForegroundColor Red
    exit 1
}
Write-Host "✓ 容器已启动" -ForegroundColor Green

# 等待容器启动
Write-Host "`n4. 等待容器启动（10秒）..." -ForegroundColor Cyan
Start-Sleep -Seconds 10

# 检查容器状态
Write-Host "`n5. 检查容器状态..." -ForegroundColor Cyan
$containerStatus = docker ps -a --filter "name=$CONTAINER_NAME" --format "{{.Status}}"
Write-Host "容器状态: $containerStatus" -ForegroundColor Yellow

# 检查容器是否正在运行
$isRunning = docker ps --filter "name=$CONTAINER_NAME" --format "{{.Names}}"
if (-not $isRunning) {
    Write-Host "警告: 容器可能未正常运行！" -ForegroundColor Yellow
    Write-Host "查看日志：" -ForegroundColor Yellow
    docker logs --tail 50 $CONTAINER_NAME
    exit 1
}

# 查看日志
Write-Host "`n6. 查看容器日志（最后50行）..." -ForegroundColor Cyan
docker logs --tail 50 $CONTAINER_NAME

# 测试关键包
Write-Host "`n7. 测试关键包..." -ForegroundColor Cyan
Write-Host "测试 gunicorn..." -ForegroundColor Yellow
docker exec $CONTAINER_NAME python3 -c "import gunicorn; print(f'✓ gunicorn={gunicorn.__version__}')" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "警告: gunicorn 测试失败" -ForegroundColor Yellow
}

Write-Host "测试 uvicorn..." -ForegroundColor Yellow
docker exec $CONTAINER_NAME python3 -c "import uvicorn; print(f'✓ uvicorn={uvicorn.__version__}')" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "警告: uvicorn 测试失败" -ForegroundColor Yellow
}

Write-Host "测试 pydantic..." -ForegroundColor Yellow
docker exec $CONTAINER_NAME python3 -c "import pydantic; print(f'✓ pydantic={pydantic.__version__}')" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "警告: pydantic 测试失败" -ForegroundColor Yellow
}

# 测试 API 端点
Write-Host "`n8. 测试 API 端点..." -ForegroundColor Cyan
Write-Host "测试 /docs 端点..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/docs" -Method Head -TimeoutSec 5 -ErrorAction Stop
    Write-Host "✓ /docs 端点响应: $($response.StatusCode)" -ForegroundColor Green
} catch {
    Write-Host "✗ /docs 端点测试失败: $_" -ForegroundColor Red
}

Write-Host "测试 /web 端点..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/web" -Method Head -TimeoutSec 5 -ErrorAction Stop
    Write-Host "✓ /web 端点响应: $($response.StatusCode)" -ForegroundColor Green
} catch {
    Write-Host "✗ /web 端点测试失败: $_" -ForegroundColor Red
}

# 测试 gunicorn 包装脚本
Write-Host "`n9. 测试 gunicorn 包装脚本..." -ForegroundColor Cyan
docker exec $CONTAINER_NAME /usr/local/bin/gunicorn --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ gunicorn 包装脚本工作正常" -ForegroundColor Green
} else {
    Write-Host "✗ gunicorn 包装脚本测试失败" -ForegroundColor Red
}

# 总结
Write-Host "`n=== 测试完成 ===" -ForegroundColor Green
Write-Host ""
Write-Host "访问以下地址测试服务：" -ForegroundColor Cyan
Write-Host "  - API 文档: http://localhost:8000/docs" -ForegroundColor Yellow
Write-Host "  - Web 界面: http://localhost:8000/web" -ForegroundColor Yellow
Write-Host ""
Write-Host "查看实时日志：" -ForegroundColor Cyan
Write-Host "  docker logs -f $CONTAINER_NAME" -ForegroundColor Yellow
Write-Host ""
Write-Host "停止并删除测试容器：" -ForegroundColor Cyan
Write-Host "  docker stop $CONTAINER_NAME" -ForegroundColor Yellow
Write-Host "  docker rm $CONTAINER_NAME" -ForegroundColor Yellow

