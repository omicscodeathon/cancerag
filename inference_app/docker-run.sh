#!/bin/bash
# Quick start script for CancerAg Inference App Docker deployment

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}CancerAg Inference App - Docker Deployment${NC}"
echo "=========================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi

# Check if docker-compose is available
if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
elif docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
else
    echo -e "${RED}Error: docker-compose is not installed${NC}"
    exit 1
fi

# Check if models exist
if [ ! -f "../results/models/random_forest.pkl" ]; then
    echo -e "${YELLOW}Warning: Model file not found at ../results/models/random_forest.pkl${NC}"
    echo -e "${YELLOW}Please ensure models are trained before deploying${NC}"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if preprocessing artifacts exist
if [ ! -f "../data/processed/ml_preprocessed/scaler.pkl" ]; then
    echo -e "${YELLOW}Warning: Preprocessing artifacts not found${NC}"
    echo -e "${YELLOW}Please ensure preprocessing is complete before deploying${NC}"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Build and run
echo -e "${GREEN}Building Docker image...${NC}"
$COMPOSE_CMD build

echo -e "${GREEN}Starting container...${NC}"
$COMPOSE_CMD up -d

echo -e "${GREEN}Waiting for app to start...${NC}"
sleep 5

# Check if container is running
if docker ps | grep -q cancerag-inference; then
    echo -e "${GREEN}✅ Container is running!${NC}"
    echo ""
    echo -e "${GREEN}App is available at: http://localhost:7860${NC}"
    echo ""
    echo "Useful commands:"
    echo "  View logs:    docker-compose logs -f"
    echo "  Stop app:     docker-compose down"
    echo "  View status:  docker ps | grep cancerag"
else
    echo -e "${RED}❌ Container failed to start${NC}"
    echo "Check logs with: docker-compose logs"
    exit 1
fi

