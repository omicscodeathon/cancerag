# Makefile for CancerAg Inference App Docker Deployment

.PHONY: build run stop logs clean help

# Docker image name
IMAGE_NAME = cancerag-inference
IMAGE_TAG = latest

# Default port
PORT = 7860

help:
	@echo "CancerAg Inference App - Docker Commands"
	@echo ""
	@echo "Available commands:"
	@echo "  make build          - Build Docker image"
	@echo "  make run            - Run container using docker-compose"
	@echo "  make stop           - Stop running container"
	@echo "  make logs           - View container logs"
	@echo "  make clean          - Remove container and image"
	@echo "  make shell           - Open shell in running container"
	@echo "  make test           - Test if app is running"
	@echo ""

build:
	@echo "Building Docker image..."
	docker build -f Dockerfile -t $(IMAGE_NAME):$(IMAGE_TAG) ..

run:
	@echo "Starting container with docker-compose..."
	docker-compose up -d
	@echo "App should be available at http://localhost:$(PORT)"

stop:
	@echo "Stopping container..."
	docker-compose down

logs:
	@echo "Viewing logs..."
	docker-compose logs -f

clean:
	@echo "Cleaning up..."
	docker-compose down -v
	docker rmi $(IMAGE_NAME):$(IMAGE_TAG) || true

shell:
	@echo "Opening shell in container..."
	docker-compose exec inference-app /bin/bash || docker exec -it cancerag-inference /bin/bash

test:
	@echo "Testing app health..."
	@curl -s http://localhost:$(PORT) > /dev/null && echo "✅ App is running!" || echo "❌ App is not responding"

rebuild:
	@echo "Rebuilding image..."
	docker-compose build --no-cache
	@echo "Restarting container..."
	docker-compose up -d

