#!/bin/bash
# Docker 容器测试脚本
# 用于测试构建成功的 whisperx-service 镜像

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

IMAGE_NAME="whisperx-service"
CONTAINER_NAME="whisperx-test"

echo -e "${GREEN}=== Docker 容器测试脚本 ===${NC}"
echo ""

# 检查 .env 文件
if [ ! -f .env ]; then
    echo -e "${RED}错误: 未找到 .env 文件！${NC}"
    echo -e "${YELLOW}请创建 .env 文件，包含以下内容：${NC}"
    echo "HF_TOKEN=你的HuggingFace Token"
    echo "WHISPER_MODEL=base"
    echo "LOG_LEVEL=info"
    exit 1
fi

# 检查镜像是否存在
echo -e "${CYAN}1. 检查镜像是否存在...${NC}"
if ! docker images --format "{{.Repository}}" | grep -q "^${IMAGE_NAME}$"; then
    echo -e "${RED}错误: 镜像 ${IMAGE_NAME} 不存在！${NC}"
    echo -e "${YELLOW}请先运行: docker build -t ${IMAGE_NAME} .${NC}"
    exit 1
fi
echo -e "${GREEN}✓ 镜像存在${NC}"

# 停止并删除旧容器（如果存在）
echo -e "\n${CYAN}2. 清理旧容器（如果存在）...${NC}"
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true
echo -e "${GREEN}✓ 清理完成${NC}"

# 启动容器
echo -e "\n${CYAN}3. 启动容器...${NC}"
if docker run -d --name $CONTAINER_NAME --gpus all -p 8000:8000 --env-file .env $IMAGE_NAME; then
    echo -e "${GREEN}✓ 容器已启动${NC}"
else
    echo -e "${RED}错误: 容器启动失败！${NC}"
    exit 1
fi

# 等待容器启动
echo -e "\n${CYAN}4. 等待容器启动（10秒）...${NC}"
sleep 10

# 检查容器状态
echo -e "\n${CYAN}5. 检查容器状态...${NC}"
CONTAINER_STATUS=$(docker ps -a --filter "name=$CONTAINER_NAME" --format "{{.Status}}")
echo -e "${YELLOW}容器状态: $CONTAINER_STATUS${NC}"

# 检查容器是否正在运行
if ! docker ps --filter "name=$CONTAINER_NAME" --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
    echo -e "${YELLOW}警告: 容器可能未正常运行！${NC}"
    echo -e "${YELLOW}查看日志：${NC}"
    docker logs --tail 50 $CONTAINER_NAME
    exit 1
fi

# 查看日志
echo -e "\n${CYAN}6. 查看容器日志（最后50行）...${NC}"
docker logs --tail 50 $CONTAINER_NAME

# 测试关键包
echo -e "\n${CYAN}7. 测试关键包...${NC}"
echo -e "${YELLOW}测试 gunicorn...${NC}"
if docker exec $CONTAINER_NAME python3 -c "import gunicorn; print(f'✓ gunicorn={gunicorn.__version__}')" 2>&1; then
    echo -e "${GREEN}✓ gunicorn 测试通过${NC}"
else
    echo -e "${YELLOW}警告: gunicorn 测试失败${NC}"
fi

echo -e "${YELLOW}测试 uvicorn...${NC}"
if docker exec $CONTAINER_NAME python3 -c "import uvicorn; print(f'✓ uvicorn={uvicorn.__version__}')" 2>&1; then
    echo -e "${GREEN}✓ uvicorn 测试通过${NC}"
else
    echo -e "${YELLOW}警告: uvicorn 测试失败${NC}"
fi

echo -e "${YELLOW}测试 pydantic...${NC}"
if docker exec $CONTAINER_NAME python3 -c "import pydantic; print(f'✓ pydantic={pydantic.__version__}')" 2>&1; then
    echo -e "${GREEN}✓ pydantic 测试通过${NC}"
else
    echo -e "${YELLOW}警告: pydantic 测试失败${NC}"
fi

# 测试 API 端点
echo -e "\n${CYAN}8. 测试 API 端点...${NC}"
echo -e "${YELLOW}测试 /docs 端点...${NC}"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/docs || echo "000")
if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "307" ]; then
    echo -e "${GREEN}✓ /docs 端点响应: $HTTP_CODE${NC}"
else
    echo -e "${RED}✗ /docs 端点测试失败: HTTP $HTTP_CODE${NC}"
fi

echo -e "${YELLOW}测试 /web 端点...${NC}"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/web || echo "000")
if [ "$HTTP_CODE" = "200" ]; then
    echo -e "${GREEN}✓ /web 端点响应: $HTTP_CODE${NC}"
else
    echo -e "${RED}✗ /web 端点测试失败: HTTP $HTTP_CODE${NC}"
fi

# 测试 gunicorn 包装脚本
echo -e "\n${CYAN}9. 测试 gunicorn 包装脚本...${NC}"
if docker exec $CONTAINER_NAME /usr/local/bin/gunicorn --version 2>&1; then
    echo -e "${GREEN}✓ gunicorn 包装脚本工作正常${NC}"
else
    echo -e "${RED}✗ gunicorn 包装脚本测试失败${NC}"
fi

# 总结
echo -e "\n${GREEN}=== 测试完成 ===${NC}"
echo ""
echo -e "${CYAN}访问以下地址测试服务：${NC}"
echo -e "${YELLOW}  - API 文档: http://localhost:8000/docs${NC}"
echo -e "${YELLOW}  - Web 界面: http://localhost:8000/web${NC}"
echo ""
echo -e "${CYAN}查看实时日志：${NC}"
echo -e "${YELLOW}  docker logs -f $CONTAINER_NAME${NC}"
echo ""
echo -e "${CYAN}停止并删除测试容器：${NC}"
echo -e "${YELLOW}  docker stop $CONTAINER_NAME${NC}"
echo -e "${YELLOW}  docker rm $CONTAINER_NAME${NC}"

