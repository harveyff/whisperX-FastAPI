#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

IMAGE_NAME="whisperx-service"
CONTAINER_NAME="whisperx-test"

echo -e "${GREEN}=== Building Docker Image ===${NC}"
docker build -t ${IMAGE_NAME} .

echo -e "\n${GREEN}=== Verifying Image ===${NC}"
echo "Checking if image exists..."
docker images | grep ${IMAGE_NAME} || (echo -e "${RED}ERROR: Image not found!${NC}" && exit 1)

echo -e "\n${GREEN}=== Testing Container Startup ===${NC}"
echo "Starting container in detached mode..."
docker run -d --name ${CONTAINER_NAME} --gpus all -p 8000:8000 --env-file .env ${IMAGE_NAME} || true

echo "Waiting 10 seconds for container to start..."
sleep 10

echo -e "\n${GREEN}=== Checking Container Status ===${NC}"
docker ps -a | grep ${CONTAINER_NAME} || echo -e "${YELLOW}Warning: Container not found${NC}"

echo -e "\n${GREEN}=== Checking Container Logs ===${NC}"
echo "Last 50 lines of logs:"
docker logs --tail 50 ${CONTAINER_NAME} || echo -e "${YELLOW}Warning: Could not get logs${NC}"

echo -e "\n${GREEN}=== Testing Gunicorn Installation ===${NC}"
echo "Checking if gunicorn is available in container..."
docker exec ${CONTAINER_NAME} python3 -m gunicorn --version || echo -e "${RED}ERROR: gunicorn not found!${NC}"

echo -e "\n${GREEN}=== Testing Critical Packages ===${NC}"
docker exec ${CONTAINER_NAME} python3 -c "import gunicorn; import uvicorn; import pydantic; import pydantic_settings; print(f'✓ All packages available: gunicorn={gunicorn.__version__}, uvicorn={uvicorn.__version__}, pydantic={pydantic.__version__}, pydantic_settings={pydantic_settings.__version__}')" || echo -e "${RED}ERROR: Some packages missing!${NC}"

echo -e "\n${GREEN}=== Testing API Endpoint ===${NC}"
echo "Testing /docs endpoint..."
curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/docs || echo -e "${YELLOW}Warning: Could not reach /docs endpoint${NC}"

echo -e "\n${GREEN}=== Testing /web Endpoint ===${NC}"
echo "Testing /web endpoint..."
curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/web || echo -e "${YELLOW}Warning: Could not reach /web endpoint${NC}"

echo -e "\n${GREEN}=== Cleanup ===${NC}"
echo "Stopping and removing test container..."
docker stop ${CONTAINER_NAME} 2>/dev/null || true
docker rm ${CONTAINER_NAME} 2>/dev/null || true

echo -e "\n${GREEN}=== Build and Test Complete ===${NC}"

