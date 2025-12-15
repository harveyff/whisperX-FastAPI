# Docker 构建和校验指南

## 快速构建和测试

### 方法 1: 使用脚本（Linux/macOS）

```bash
# 赋予执行权限
chmod +x build_and_test.sh

# 运行构建和测试脚本
./build_and_test.sh
```

### 方法 2: 手动步骤（Windows/Linux/macOS）

#### 1. 构建镜像

```bash
docker build -t whisperx-service .
```

#### 2. 验证镜像是否存在

```bash
docker images | grep whisperx-service
```

#### 3. 启动容器进行测试

```bash
# 启动容器（确保 .env 文件存在）
docker run -d --name whisperx-test --gpus all -p 8000:8000 --env-file .env whisperx-service

# 等待容器启动
sleep 10

# 查看容器状态
docker ps -a | grep whisperx-test
```

#### 4. 检查容器日志

```bash
# 查看最后 50 行日志
docker logs --tail 50 whisperx-test

# 实时查看日志
docker logs -f whisperx-test
```

#### 5. 验证关键包是否安装

```bash
# 检查 gunicorn 是否可用
docker exec whisperx-test python3 -m gunicorn --version

# 检查所有关键包
docker exec whisperx-test python3 -c "import gunicorn; import uvicorn; import pydantic; import pydantic_settings; print(f'✓ gunicorn={gunicorn.__version__}, uvicorn={uvicorn.__version__}, pydantic={pydantic.__version__}, pydantic_settings={pydantic_settings.__version__}')"
```

#### 6. 测试 API 端点

```bash
# 测试 /docs 端点
curl -I http://localhost:8000/docs

# 测试 /web 端点
curl -I http://localhost:8000/web

# 测试根路径（应该重定向到 /docs）
curl -I http://localhost:8000/
```

#### 7. 验证 gunicorn 包装脚本

```bash
# 检查包装脚本是否存在
docker exec whisperx-test ls -la /usr/local/bin/gunicorn

# 测试包装脚本
docker exec whisperx-test /usr/local/bin/gunicorn --version
```

#### 8. 清理测试容器

```bash
# 停止并删除测试容器
docker stop whisperx-test
docker rm whisperx-test
```

## 常见问题排查

### 问题 1: `gunicorn: not found`

**检查步骤：**
```bash
# 1. 检查 gunicorn 模块是否安装
docker exec whisperx-test python3 -c "import gunicorn; print(gunicorn.__version__)"

# 2. 检查包装脚本是否存在
docker exec whisperx-test ls -la /usr/local/bin/gunicorn

# 3. 检查 PATH 环境变量
docker exec whisperx-test echo $PATH
```

### 问题 2: `No module named 'pydantic'` 或 `No module named 'pydantic_settings'`

**检查步骤：**
```bash
# 检查所有关键包
docker exec whisperx-test python3 -c "import gunicorn; import uvicorn; import pydantic; import pydantic_settings; print('All packages OK')"
```

### 问题 3: 容器启动后立即退出

**检查步骤：**
```bash
# 查看容器退出代码
docker inspect whisperx-test | grep -A 10 "State"

# 查看完整日志
docker logs whisperx-test
```

## 完整验证清单

- [ ] 镜像构建成功
- [ ] 容器可以启动
- [ ] gunicorn 模块可以导入
- [ ] uvicorn 模块可以导入
- [ ] pydantic 模块可以导入
- [ ] pydantic_settings 模块可以导入
- [ ] `/usr/local/bin/gunicorn` 包装脚本存在
- [ ] `python3 -m gunicorn --version` 可以执行
- [ ] `/usr/local/bin/gunicorn --version` 可以执行
- [ ] `/docs` 端点可以访问
- [ ] `/web` 端点可以访问
- [ ] 容器日志中没有错误信息

## 使用 Docker Compose

```bash
# 构建并启动
docker-compose up --build

# 查看日志
docker-compose logs -f

# 停止
docker-compose down
```

